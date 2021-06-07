import torch,cPickle
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from utils import *



def Tensor2Varible(tensor_):
    var = Variable(tensor_)
    var = var.cuda() if use_cuda else var
    return var

class Ranker(nn.Module):
    def __init__(self, vocab_size,ent_size, config):
        super(Ranker, self).__init__()
        self.vocab_size = vocab_size
        self.ent_size = ent_size
        self.batch_size = config['batch_size']
        self.config = config
        self.feature_size = config['feature_size']
        self.intent_num = config['intent_num']
        self.intent_emb_size = config['intent_emb_size']
        self.out_hidden_num = config['out_hidden_num']

        self.word_emb = nn.Embedding(self.vocab_size, self.config['word_emb_size'])
        self.ent_emb = nn.Embedding(ent_size, self.config['ent_emb_size'])

        self.load_embedding(self.word_emb, self.config['word_emb'])
        self.load_embedding(self.ent_emb, self.config['ent_emb'],name='entity embedding')

        self.topic_embedding = torch.Tensor(self.intent_num, self.intent_emb_size)

        self.out_linear = nn.Linear(self.config['n_bins'] * 4, 1)  # + self.intent_emb_size

        tensor_mu = torch.FloatTensor(self.config['mu'])
        tensor_sigma = torch.FloatTensor(self.config['sigma'])

        if use_cuda:
            tensor_mu = tensor_mu.cuda()
            tensor_sigma = tensor_sigma.cuda()
            self.topic_embedding = self.topic_embedding.cuda()

        self.topic_embedding = nn.Parameter(self.topic_embedding)
        nn.init.orthogonal_(self.topic_embedding)

        self.kernel_mu = Variable(tensor_mu, requires_grad=False).view(1, 1, 1, self.config['n_bins'])
        self.kernel_sigma = Variable(tensor_sigma, requires_grad=False).view(1, 1, 1, self.config['n_bins'])

    def load_embedding(self,emb,addr,name='word embedding'):
        pre_embeds = cPickle.load(open(addr))
        print 'emb: ',emb.weight.size()
        print name, ': ', pre_embeds.shape
        emb.weight = nn.Parameter(torch.FloatTensor(pre_embeds))

    def get_intersect_matrix(self, q_embed, d_embed,q_mask=None,d_mask=None,use_q_mask=True):
        d_embed = d_embed.transpose(1, 2)
        batch_size = q_embed.size()[0]
        sim = torch.bmm(q_embed, d_embed)#(bs,ql,dl)
        sim = sim.view(batch_size, q_embed.size()[1], d_embed.size()[2], 1)

        pooling_value = torch.exp((- ((sim - self.kernel_mu) ** 2) / (self.kernel_sigma ** 2) / 2)) \
                        * d_mask.view(batch_size, 1, -1, 1) #(bs,ql,dl,K)

        pooling_sum = torch.sum(pooling_value, 2)#(bs,ql,K)
        log_pooling_sum = torch.log(torch.clamp(pooling_sum, min=1e-10)) * 0.01
        if use_q_mask:
            log_pooling_sum = log_pooling_sum * q_mask.view(batch_size, -1, 1)
        log_pooling_sum = torch.sum(log_pooling_sum, 1)  # (bs,K)

        return log_pooling_sum

    def forward(self,input_qw,input_qe,input_dw,input_de,theta):
        input_qw = Tensor2Varible(torch.LongTensor(input_qw))
        input_dw = Tensor2Varible(torch.LongTensor(input_dw))
        input_qe = Tensor2Varible(torch.LongTensor(input_qe))
        input_de = Tensor2Varible(torch.LongTensor(input_de))

        mask_qw = torch.ne(input_qw, 0).float()  # (bs,ql)
        mask_dw = torch.ne(input_dw, 0).float()  # (bs,dl)
        mask_qe = torch.ne(input_qe, 0).float()
        mask_de = torch.ne(input_de, 0).float()

        input_qw_embed = self.word_emb(input_qw)
        input_dw_embed = self.word_emb(input_dw)
        input_qe_embed = self.ent_emb(input_qe)
        input_de_embed = self.ent_emb(input_de)

        log_pooling_sum_ent = self.get_intersect_matrix(input_qe_embed, input_de_embed, mask_qe, mask_de)# (bs,K)
        log_pooling_sum_wrd = self.get_intersect_matrix(input_qw_embed, input_dw_embed, mask_qw, mask_dw)# (bs,K)

        topic_embeddings = theta.unsqueeze(-1) * self.topic_embedding.view(1, self.intent_num, -1)  # (bs,T,I_embed)
        #topic_embeddings = torch.sum(topic_embeddings, dim=1)

        orth_loss = torch.mm(self.topic_embedding, self.topic_embedding.t()).sum() - Tensor2Varible(
            torch.eye(self.topic_embedding.size(0))).sum()

        if self.config['theta_weight']:
            log_pooling_sum_ent_intent = self.get_intersect_matrix(topic_embeddings, input_de_embed,q_mask=theta, d_mask=mask_de)  # (bs,K)
            log_pooling_sum_wrd_intent = self.get_intersect_matrix(topic_embeddings, input_dw_embed,q_mask=theta, d_mask=mask_dw)  # (bs,K)
        else:
            log_pooling_sum_ent_intent = self.get_intersect_matrix(topic_embeddings, input_de_embed, d_mask=mask_de,use_q_mask=False)# (bs,K)
            log_pooling_sum_wrd_intent = self.get_intersect_matrix(topic_embeddings, input_dw_embed, d_mask=mask_dw,use_q_mask=False)  # (bs,K)

        hidden = torch.cat([log_pooling_sum_ent,log_pooling_sum_wrd,log_pooling_sum_wrd_intent,log_pooling_sum_ent_intent], dim=-1)
        score = self.out_linear(hidden)

        return score, orth_loss