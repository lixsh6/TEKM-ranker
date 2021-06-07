# coding=utf-8
import time
import torch.optim as optim
from config import *
from data.generator import *
from metrics.rank_evaluations import *
from model.Ranker import *
from model.VAE import *

from utils import *
best_result = 0.0
from tensorboardX import SummaryWriter

def toVisdomY(Y):
    if type(Y) == torch.Tensor:
        return Y.view(1,).cpu()
    else:
        return np.array([Y])

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class TrainingModel(object):
    def __init__(self,args, config):

        self.__dict__.update(config)
        self.config = config
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        if use_cuda:
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.cuda.set_device(args.gpu)

        #torch.backends.cudnn.benchmark = False
        #torch.backends.cudnn.deterministic = True


        self.message = args.m
        self.data_generator = DataGenerator(self.config)
        self.vocab_size = self.data_generator.vocab_size
        self.ent_size = self.data_generator.ent_size

        self.model_name = 'IERM'


        if args.m != "":
            self.saveModeladdr = './trainModel/checkpoint_%s_%s.pkl' % (self.model_name,args.m)
        else:
            self.saveModeladdr = './trainModel/' + args.save


        self.model = Ranker(self.vocab_size, self.ent_size, self.config)
        self.VAE_model = VAE(self.vocab_size, self.ent_size,self.model.word_emb,self.model.ent_emb, self.config)

        if use_cuda:
            self.model.cuda()
            self.VAE_model.cuda()

        vae_lr = self.config['pretrain_lr'] if config['pretrain_step'] > 0 else config['vae_lr']
        self.vae_optimizer = getOptimizer(config['vae_optim'], self.VAE_model.parameters(),
                                             lr=vae_lr,betas=(0.99, 0.99))
        self.ranker_optimizer = getOptimizer(config['ranker_optim'], self.model.parameters(),
                                             lr=config['ranker_lr'],weight_decay=config['weight_decay'])

        vae_model_size = sum(p.numel() for p in self.VAE_model.parameters())
        ranker_size = sum(p.numel() for p in self.model.parameters())
        #print 'Model size: ', vae_model_size, ranker_size
        #exit(-1)
        if args.resume and os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            #print checkpoint.keys()
            self.model.load_state_dict(checkpoint['rank_state_dict'])
            self.VAE_model.load_state_dict(checkpoint['vae_state_dict'])
            self.vae_optimizer.load_state_dict(checkpoint['vae_optimizer'])
            self.ranker_optimizer.load_state_dict(checkpoint['rank_optimizer'])
        else:
            print("Creating a new model")

        self.timings = defaultdict(list) #record the loss iterations
        self.evaluator = rank_eval()
        self.epoch = 0
        self.step = 0

        self.kl_weight = 1

        if args.visual:
            self.config['visual'] = True
            self.writer = SummaryWriter('runs/' + args.m)
        else:
            self.config['visual'] = False
        self.reconstr_loss = nn.MSELoss()

    def add_values(self,iter, value_dict):
        for key in value_dict:
            self.writer.add_scalar(key,value_dict[key],iter)

    def adjust_learning_rate(self,optimizer, lr, decay_rate=.5):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * decay_rate

    def kl_anneal_function(self,anneal_function, step, k=0.0025, x0=2500):
        if anneal_function == 'logistic':
            return float(1 / (1 + np.exp(-k * (step - x0))))
        elif anneal_function == 'linear':
            return min(1, step / x0)

    def vae_loss(self,input_qw,reconstr_w,input_qe,reconstr_e, prior_mean, prior_var,
              posterior_mean, posterior_var, posterior_log_var):
        # Reconstruction term
        if self.config['reconstruct'] != 'entity':
            input_qw_bow = to_bow(input_qw, self.vocab_size)
            input_qw_bow = Tensor2Varible(torch.tensor(input_qw_bow).float())
            #reconstr_w = torch.log_softmax(reconstr_w + 1e-10,dim=1)
            #RL_w = -torch.sum(input_qw_bow * reconstr_w , dim=1)
            #RL_w = self.reconstr_loss(reconstr_w,input_qw_bow)
            RL_w = -torch.sum(input_qw_bow * reconstr_w + (1-input_qw_bow) * torch.log(1 - torch.exp(reconstr_w)), dim=1)
        else:
            RL_w = Tensor2Varible(torch.tensor([0]).float())
        if self.config['reconstruct'] != 'word':
            input_qe_bow = to_bow(input_qe, self.ent_size)
            input_qe_bow = Tensor2Varible(torch.tensor(input_qe_bow).float())
            #RL_e = -torch.sum(input_qe_bow * reconstr_e, dim=1)
            #RL_e = self.reconstr_loss(reconstr_e,input_qe_bow)
            RL_e = -torch.sum(input_qe_bow * reconstr_e + (1-input_qe_bow) * torch.log(1 - torch.exp(reconstr_e)), dim=1)
        else:
            RL_e = Tensor2Varible(torch.tensor([0]).float())

        # KL term
        # var division term
        var_division = torch.sum(posterior_var / prior_var, dim=1)
        # diff means term
        diff_means = prior_mean - posterior_mean
        diff_term = torch.sum(
            (diff_means * diff_means) / prior_var, dim=1)
        # logvar det division term
        logvar_det_division = \
            prior_var.log().sum() - posterior_log_var.sum(dim=1)
        # combine terms
        KL = 0.5 * (
                var_division + diff_term - self.model.intent_num + logvar_det_division)

        loss = self.kl_weight * KL + RL_w + RL_e
        #loss = 0.001 * KL + RL_w + RL_e

        return loss.sum(), KL.sum(), RL_w.sum(), RL_e.sum()

    def pretraining(self):
        if self.pretrain_step <= 0:
            return

        train_start_time = time.time()
        data_reader = self.data_generator.pretrain_reader(self.pretrain_bs)
        total_loss = 0.;total_KL_loss = 0.
        total_RLw_loss = 0.
        total_RLe_loss = 0.
        for step in xrange(self.pretrain_step):
            input_qw,input_qe = next(data_reader)
            #self.kl_weight = self.kl_anneal_function('logistic', step)
            topic_e, vae_loss, kl_loss, rl_w_loss, rl_e_loss = self.train_VAE(input_qw,input_qe)
            vae_loss.backward()
            torch.nn.utils.clip_grad_value_(self.VAE_model.parameters(), self.clip_grad)  # clip_grad_norm(, )

            self.vae_optimizer.step()

            vae_loss = vae_loss.data

            #print ('VAE loss: %.3f\tKL: %.3f\tRL_w:%.3f\tRL_e:%.3f' % (vae_loss, kl_loss, rl_w_loss, rl_e_loss))

            if torch.isnan(vae_loss):
                print("Got NaN cost .. skipping")
                exit(-1)
                continue

            #if self.config['visual']:
            #    self.add_values(step, {'vae_loss': vae_loss, 'kl_loss': kl_loss, 'rl_w_loss': rl_w_loss,
            #                          'rl_e_loss': rl_e_loss, 'kl_weight': self.kl_weight})

            total_loss += vae_loss
            total_KL_loss += kl_loss
            total_RLw_loss += rl_w_loss
            total_RLe_loss += rl_e_loss

            if step != 0 and step % self.pretrain_freq == 0:
                total_loss /= self.pretrain_freq
                total_KL_loss /= self.pretrain_freq
                total_RLw_loss /= self.pretrain_freq
                total_RLe_loss /= self.pretrain_freq
                print ('Step: %d\t Elapsed:%.2f' % (step, time.time() - train_start_time))
                print ('Pretrain VAE loss: %.3f\tKL: %.3f\tRL_w:%.3f\tRL_e:%.3f' % (total_loss,total_KL_loss,total_RLw_loss,total_RLe_loss))
                if self.config['visual']:
                    self.add_values(step, {'vae_loss': total_loss, 'kl_loss': total_KL_loss, 'rl_w_loss': total_RLw_loss,
                                           'rl_e_loss': total_RLe_loss, 'kl_weight': self.kl_weight})
                total_loss = 0.
                total_KL_loss = 0.
                total_RLw_loss = 0.
                total_RLe_loss = 0.
                print '=============================================='
                #self.generate_beta_phi_3(show_topic_limit=5)

        self.save_checkpoint(message=self.message + '-pretraining')
        print ('Pretraining end')
        #recovering the learning rate
        self.adjust_learning_rate(self.vae_optimizer,self.config['vae_lr'],1)


    def trainIters(self,):
        self.step = 0
        train_start_time = time.time()
        patience = self.patience

        best_ndcg10 = 0.0
        last_ndcg10 = 0.0

        data_reader = self.data_generator.pair_reader(self.batch_size)
        total_loss = 0.0
        total_rank_loss = 0.
        total_vae_loss = 0.
        total_KL_loss = 0.
        total_RLw_loss = 0.
        total_RLe_loss = 0.

        for step in xrange(self.steps):
            out = next(data_reader)
            input_qw,input_qe,input_dw_pos,input_de_pos,input_dw_neg,input_de_neg = out
            rank_loss, vae_total_loss, KL_loss, RL_w_loss, RL_e_loss \
                = self.train(input_qw,input_qe,input_dw_pos,input_de_pos,input_dw_neg,input_de_neg)

            cur_total_loss = rank_loss + vae_total_loss
            if torch.isnan(cur_total_loss):
                print("Got NaN cost .. skipping")
                continue
            self.step += 1
            total_loss += cur_total_loss
            total_rank_loss += rank_loss
            total_vae_loss += vae_total_loss
            total_KL_loss += KL_loss
            total_RLw_loss += RL_w_loss
            total_RLe_loss += RL_e_loss

            if self.eval_freq != -1 and self.step % self.eval_freq == 0:
                with torch.no_grad():
                    valid_performance = self.test(valid_or_test='valid',source=self.config['click_model'])
                    current_ndcg10 = valid_performance['ndcg@10']

                    if current_ndcg10 > best_ndcg10:
                        print 'Got better result, save to %s' % self.saveModeladdr
                        best_ndcg10 = current_ndcg10
                        patience = self.patience
                        self.save_checkpoint(message=self.message)

                        #self.generate_beta_phi_3(show_topic_limit=5)
                    elif current_ndcg10 <= last_ndcg10 * self.cost_threshold:
                        patience -= 1
                    last_ndcg10 = current_ndcg10

            if self.step % self.train_freq == 0:
                total_loss /= self.train_freq
                total_rank_loss /= self.train_freq
                total_vae_loss /= self.train_freq
                total_KL_loss /= self.train_freq
                total_RLw_loss /= self.train_freq
                total_RLe_loss /= self.train_freq

                self.timings['train'].append(total_loss)
                print ('Step: %d\t Elapsed:%.2f' % (step, time.time() - train_start_time))
                print ('Train total loss: %.3f\tRank loss: %.3f\tVAE loss: %.3f' % (total_loss, total_rank_loss, total_vae_loss))
                print ('KL loss: %.3f\tRL W: %.3f\tRL E: %.3f' % (total_KL_loss, total_RLw_loss, total_RLe_loss))
                print ('Patience left: %d' % patience)

                if self.config['visual']:
                    self.add_values(step, {'Train vae_loss': total_loss, 'Train kl_loss': total_KL_loss, 'Train rl_w_loss': total_RLw_loss,
                                           'Train rl_e_loss': total_RLe_loss, 'Train Rank loss': total_rank_loss})

                total_loss = 0
                total_rank_loss = 0.
                total_vae_loss = 0.
                total_KL_loss = 0.
                total_RLw_loss = 0.
                total_RLe_loss = 0.

            if patience < 0:
                print 'patience runs out...'
                break


        print 'Patience___: ',patience
        print ("All done, exiting...")

    def test(self,valid_or_test,source):

        predicted = []
        results = defaultdict(list)

        if valid_or_test == 'valid':
            is_test = False
            data_addr = self.valid_rank_addr
            data_source = self.data_generator.pointwise_reader_evaluation(data_addr, is_test=is_test,
                                                                          label_type=source)
        elif valid_or_test == 'ntcir13' or valid_or_test == 'ntcir14':
            is_test = True
            data_source = self.data_generator.pointwise_ntcir_generator(valid_or_test)
            source = 'HUMAN'
        else:
            is_test = True
            data_addr = self.test_rank_addr
            data_source = self.data_generator.pointwise_reader_evaluation(data_addr, is_test=is_test,
                                                                          label_type=source)
        start = time.clock()
        count = 0
        for out in data_source:
            (qid, dids, input_qw,input_qe,input_dw,input_de,gt_rels) = out
            gt_rels = map(lambda t: score2cutoff(source, t), gt_rels)
            rels_predicted = self.predict(input_qw,input_qe,input_dw,input_de).view(-1).cpu().numpy()

            result = self.evaluator.eval(gt_rels, rels_predicted)
            for did, gt,pred in zip(dids, gt_rels, rels_predicted):
                predicted.append((qid,did, pred, gt))

            for k,v in result.items():
                results[k].append(v)
            count += 1
        elapsed = (time.clock() - start)
        print('Elapsed:%.3f\tAvg:%.3f' % (elapsed, elapsed / count))
        performances = {}

        for k, v in results.items():
            performances[k] = np.mean(v)


        print '------Source: %s\tPerformance-------:' % source
        print 'Validating...' if valid_or_test == 'valid' else 'Testing'
        print 'Message: %s' % self.message
        print 'Source: %s' % source
        print performances


        if valid_or_test != 'valid':
            path = './results/' + self.message + '_' + valid_or_test + '_' + source
            if not os.path.exists(path):
                os.makedirs(path)
            out_file = open('%s/%s.predicted.txt'%(path,self.model_name),'w')
            for qid,did, pred, gt in predicted:
                print >> out_file, '\t'.join([qid,did, str(pred), str(gt)])

        return performances

    def get_text(self,input,map_fun):
        text_list = []
        for element in input:
            if element == 0:
                break
            text_list.append(map_fun(element))
        return ' '.join(text_list)

    def generate_beta_phi_3(self, topK=10, show_topic_limit=-1):
        beta,phi = self.VAE_model.infer_topic_dis(topK)
        topics = defaultdict(list)
        topics_ents = defaultdict(list)
        show_topic_num = self.config['intent_num'] if show_topic_limit == -1 else show_topic_limit

        for i in range(show_topic_num):
            idxs = beta[i]
            eidxs = phi[i]
            component_words = [self.data_generator.id2word[idx]
                               for idx in idxs.cpu().numpy()]
            component_ents = [self.data_generator.id2ent[self.data_generator.new2old[idx]]
                              for idx in eidxs.cpu().numpy()]
            topics[i] = component_words
            topics_ents[i] = component_ents

        print '--------Topic-Word-------'
        prefix = ('./topic/%s/' % args.m)
        if not os.path.exists(prefix):
            os.makedirs(prefix)
        outfile = open(prefix + 'topic-words.txt','w')
        for k in topics:
            print >> outfile, (str(k) + ' : ' + ' '.join(topics[k]))
            print >> outfile, (str(k) + ' : ' + ' '.join(topics_ents[k]))
        return topics,topics_ents


    def run_test_topic(self,out_file_name, topK, topicNum):
        topics_words, topics_ents = self.generate_beta_phi_3(topK)
        data_addr = self.test_rank_addr
        data_source = self.data_generator.pointwise_reader_evaluation(data_addr, is_test=True,
                                                                      label_type=self.config['click_model'])
        out_file = open(out_file_name, 'w')
        with torch.no_grad():
            self.VAE_model.eval()
            self.model.eval()
            for i, out in enumerate(data_source):
                (qid, dids, input_qw, input_qe, input_dw, input_de, gt_rels) = out
                theta = self.VAE_model.get_theta(input_qw, input_qe)
                input_qw = input_qw[0]
                input_qe = input_qe[0]

                input_w = self.get_text(input_qw, lambda w: self.data_generator.id2word[w])
                input_e = self.get_text(input_qe, lambda e: self.data_generator.id2ent[self.data_generator.new2old[e]])

                theta = theta[0].data.cpu().numpy()
                top_indices = np.argsort(theta)[::-1][:3]

                #print '========================='
                print >> out_file, 'Query: ', input_w
                print >> out_file, 'Entity: ', input_e
                for j,k in enumerate(top_indices):
                    ws = topics_words[k]
                    es = topics_ents[k]
                    print >> out_file, '%d Word Topic %d: %s' % (j, k, ' '.join(ws))
                    print >> out_file, '%d Entity Topic %d: %s' % (j, k, ' '.join(es))

    def generate_topic_word_ent(self,out_file, topK=10):
        print 'Visualizing ...'
        data_addr = self.test_rank_addr
        data_source = self.data_generator.pointwise_reader_evaluation(data_addr, is_test=True,
                                                                      label_type=self.config['click_model'])
        out_file = open(out_file,'w')
        with torch.no_grad():
            self.VAE_model.eval()
            self.model.eval()
            for i,out in enumerate(data_source):
                (input_qw,input_qe,input_dw,input_de,gt_rels) = out
                _, word_indices, ent_indices = self.VAE_model.get_topic_words(input_qw,input_qe,topK=topK)
                word_indices = word_indices[0].data.cpu().numpy()
                ent_indices = ent_indices[0].data.cpu().numpy()

                #print 'ent_indices: ', ent_indices
                #print 'word_indices: ', word_indices
                input_qw = input_qw[0]
                input_qe = input_qe[0]

                input_w = self.get_text(input_qw, lambda w: self.data_generator.id2word[w])
                input_e = self.get_text(input_qe, lambda e: self.data_generator.id2ent[self.data_generator.new2old[e]])
                reconstuct_w = self.get_text(word_indices,lambda w:self.data_generator.id2word[w])
                reconstuct_e = self.get_text(ent_indices, lambda e:self.data_generator.id2ent[self.data_generator.new2old[e]] )

                print >> out_file, ('%d: Word: %s\tRecons: %s' % (i+1, input_w,reconstuct_w))
                print >> out_file, ('%d: Ent: %s\tRecons: %s' % (i+1, input_e, reconstuct_e))


    def train_VAE(self,input_qw,input_qe):
        self.VAE_model.train()
        self.VAE_model.zero_grad()
        self.vae_optimizer.zero_grad()

        topic_embeddings, logPw, logPe, prior_mean, prior_variance,\
            poster_mu, poster_sigma, poster_log_sigma = self.VAE_model(input_qw,input_qe)

        vae_total_loss, KL, RL_w, RL_e = self.vae_loss(input_qw,logPw,input_qe,logPe, prior_mean, prior_variance,
              poster_mu, poster_sigma, poster_log_sigma)

        #vae_total_loss.backward(retain_graph=True)
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        #torch.nn.utils.clip_grad_value_(self.VAE_model.parameters(), self.clip_grad)  # clip_grad_norm(, )
        #self.vae_optimizer.step()

        return topic_embeddings, vae_total_loss, KL.data, RL_w.data, RL_e.data


    def train(self,input_qw,input_qe,input_dw_pos,input_de_pos,input_dw_neg,input_de_neg):
        # Turn on training mode which enables dropout.
        self.model.train()
        self.model.zero_grad()
        self.ranker_optimizer.zero_grad()

        topic_embeddings,vae_total_loss,KL_loss,RL_w_loss,RL_e_loss = self.train_VAE(input_qw,input_qe)

        score_pos, orth_loss_1 = self.model(input_qw,input_qe,input_dw_pos,input_de_pos,topic_embeddings)
        score_neg, orth_loss_2 = self.model(input_qw,input_qe,input_dw_neg,input_de_neg,topic_embeddings)

        rank_loss = torch.sum(torch.clamp(1.0 - score_pos + score_neg, min=0))
        vae_weight = self.config['intent_lambda']

        orth_loss = (orth_loss_1 + orth_loss_2) / 2
        total_loss = rank_loss + vae_weight * vae_total_loss + orth_loss
        total_loss.backward()

        ## update parameters
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_value_(self.VAE_model.parameters(), self.clip_grad)  # clip_grad_norm(, )
        torch.nn.utils.clip_grad_value_(self.model.parameters(),self.clip_grad) #clip_grad_norm(, )

        self.ranker_optimizer.step()
        self.vae_optimizer.step()

        return rank_loss.data, vae_total_loss.data, KL_loss, RL_w_loss, RL_e_loss


    def predict(self,input_qw,input_qe,input_dw,input_de):
        # Turn on evaluation mode which disables dropout.
        with torch.no_grad():
            self.VAE_model.eval()
            self.model.eval()
            topic_embeddings = self.VAE_model(input_qw,input_qe)
            rels_predicted, _ = self.model(input_qw,input_qe,input_dw,input_de,topic_embeddings)

        return rels_predicted


    def save_checkpoint(self,message):
        filePath = os.path.join(self.saveModeladdr)
        #if not os.path.exists(filePath):
        #    os.makedirs(filePath)
        torch.save({
            'vae_state_dict':self.VAE_model.state_dict(),
            'rank_state_dict': self.model.state_dict(),
            'vae_optimizer': self.vae_optimizer.state_dict(),
            'rank_optimizer':self.ranker_optimizer.state_dict()
        }, filePath)


    def get_embeddings(self):
        word_embeddings = self.model.word_emb.weight.detach().cpu().numpy()
        ent_embeddings = self.model.ent_emb.weight.detach().cpu().numpy()
        topic_embeddings = self.model.topic_embedding.detach().cpu().numpy()

        print 'Topic size: ', topic_embeddings.shape[0]
        cPickle.dump((word_embeddings,ent_embeddings,topic_embeddings),open('./topic_analysis/w_e_t_embedding.pkl','w'))
        print 'saved'
        return



def main(args,config):
    train_model = TrainingModel(args,config)
    #train_model.get_embeddings()
    #return
    if not args.eval:
        train_model.pretraining()
        train_model.trainIters()
        #train_model.generate_beta_phi_3()
    if args.eval:
        print 'only evaluation'


    performances = train_model.test(valid_or_test='test', source='HUMAN')
    #performances = train_model.test(valid_or_test='test', source='PSCM')
    #performances = train_model.test(valid_or_test='test', source='UBM')
    #performances = train_model.test(valid_or_test='test', source='DBN')
    #ntcir13_performance = train_model.test(valid_or_test='ntcir13',source='HUMAN')
    #ntcir14_performance = train_model.test(valid_or_test='ntcir14', source='HUMAN')

    if config['reconstruct'] == 'both':
        prefix = ('./topic/%s/' % args.m)
        if not os.path.exists(prefix):
            os.makedirs(prefix)
        train_model.run_test_topic(prefix + 'test_topic-w-e.txt',20,config['intent_num'])

    #print performances

if __name__ == '__main__':
    args = load_arguments()
    config_state = eval(args.prototype)()
    main(args, config_state)
