import argparse


def load_arguments():
    parser = argparse.ArgumentParser(description='Evidence Reading Model')

    parser.add_argument('--resume', type=str, default="",
                        help='Resume training from that state')
    parser.add_argument("--prototype", type=str, help="Use the prototype", default='basic_config')
    parser.add_argument("--eval", action="store_true", help="only evaluation")

    #parser.add_argument('--visdom', action="store_true",
    #                    help='Use visdom for loss visualization')
    parser.add_argument('--m', type=str, default="",
                        help='Message in visualizationn window (Flag)')
    parser.add_argument('--gpu', type=int, default=1,
                        help="# of GPU running on")
    parser.add_argument('--save', type=str, default="checkpoint.pth.tar",
                        help='save filename for training model')
    parser.add_argument('--visual', action="store_true",
                        help='Visualize the topic words and ents')
    args = parser.parse_args()

    return args


def basic_config():
    state = {}
    state['min_score_diff'] = 0.25
    state['dataName'] = 'ST'

    state['pairwise'] = True
    state['click_model'] = 'PSCM'

    original_addr = ''

    state['train_rank_addr'] = original_addr + '/ad-hoc-udoc/train/'
    state['valid_rank_addr'] = original_addr + '/ad-hoc-udoc/valid/'
    state['test_rank_addr'] = original_addr + '/ad-hoc-udoc/test/'

    state['local_data_addr'] = '../../GraRanker/data/'
    state['vocab_dict_file'] = original_addr + '/GraRanker/data/vocab.dict.9W.pkl'
    state['ent_dict_file'] = 'the_prefix/entity/raw-xlore.ent_dict.pkl'
    state['entID_map_dict_file'] = '../baselines/data/newold_eid_dict.pkl'
    state['qd2id_dict_file'] = 'the_prefix/data/qdid2eid-xlore.pkl'
    state['pretrain_qid_addr'] = './lda/train_queries_qid.txt'

    state['word_emb'] = original_addr + '/GraRanker/data/emb50_9W.pkl'
    state['ent_emb'] = '../baselines/data/ent_embedding_50.pkl'

    state['topic_output'] = './data/test_topic_word_ent.txt'
    state['drate'] = 0.8
    state['seed'] = 1234

    state['batch_size'] = 80
    state['steps'] = 3000000
    state['lr'] = 0.001#0.01
    state['weight_decay'] = 0#1e-3
    state['clip_grad'] = 0.5
    state['optim'] = 'adam'  # 'sgd, adadelta*' adadelta0.1, adam0.005

    state['mask_id'] = 0

    state['word_emb_size'] = 50
    state['ent_emb_size'] = 50

    state['cost_threshold'] = 0.997
    state['patience'] = 5
    state['train_freq'] = 50  # 200
    state['eval_freq'] = 500  # 5000
    state['value_loss_coef'] = 0.5

    state['prate'] = 0.2  # exploration applied to layers (0 = noexploration)
    return state



def edrm_config():
    state = basic_config()

    state['model_name'] = 'edrm'
    state['batch_size'] = 80

    state['optim'] = 'adam'#adagrad
    state['lr'] = 0.001
    state['weight_decay'] = 1e-5

    state['drate'] = 0

    state['max_query_length'] = 12
    state['max_doc_length'] = 15

    state['win_size'] = 3

    state['sigma'] = [1e-3] + [0.1] * 10
    state['mu'] = [1.0, 0.9, 0.7, 0.5, 0.3, 0.1, -0.1, -0.3, -0.5, -0.7, -0.9]
    state['n_bins'] = len(state['mu'])

    return state

def ierm_config():
    state = basic_config()

    state['model_name'] = 'ierm'
    state['batch_size'] = 80#80
    state['train_freq'] = 50  # 200
    state['eval_freq'] = 500  # 5000

    state['vae_optim'] = 'adam'
    state['ranker_optim'] = 'adam'#adam
    state['vae_lr'] = 1e-5
    state['ranker_lr'] = 0.001
    state['weight_decay'] = 1e-5

    state['drate'] = 0.3#0.3
    state['max_query_length'] = 12
    state['max_doc_length'] = 15
    state['max_query_ent'] = 8

    state['kernel_size'] = 3

    state['sigma'] = [1e-3] + [0.1] * 10
    state['mu'] = [1.0, 0.9, 0.7, 0.5, 0.3, 0.1, -0.1, -0.3, -0.5, -0.7, -0.9]
    state['n_bins'] = len(state['mu'])

    state['intent_num'] = 128#128
    state['feature_size'] = 128
    state['intent_emb_size'] = 50#128
    state['out_hidden_num'] = 128

    state['window_sizes'] = [1,2,3]#[1,2,3]
    state['ent_window_sizes'] = [1,2,3]#[1,2,3]
    state['learn_priors'] = False

    state['pretrain_lr'] = 2e-3#2e-3
    state['pretrain_freq'] = 200
    state['pretrain_step'] = 8000
    state['pretrain_bs'] = 10
    state['intent_lambda'] = 1  # 0.005

    state['reconstruct'] = 'both' #both,word,entity
    state['LDA_model_type'] = 'LDA' #prodLDA
    state['theta_weight'] = False
    #state['steps'] = 300
    return state

def origin_config():
    state = ierm_config()
    state['vae_lr'] = 0
    state['pretrain_lr'] = 0
    state['pretrain_step'] = 0
    state['intent_lambda'] = 0
    return state


def ierm2_config():
    state = ierm_config()
    state['pretrain_step'] = 10000
    state['steps'] = 100

    state['pretrain_lr'] = 5e-3
    return state


def ierm3_config():
    state = ierm_config()
    state['pretrain_step'] = 12000
    state['pretrain_lr'] = 2e-3
    state['pretrain_bs'] = 32
    state['reconstruct'] = 'both'
    state['epoches'] = 10
    state['steps'] = 300000

    #state['entID_map_dict_file'] = '../baselines/data/newold_eid_dict_new.pkl'
    #state['ent_emb'] = '../baselines/data/ent_embedding_50_new.pkl'

    state['theta_weight'] = False
    state['LDA_model_type'] = 'prodLDA'  # prodLDA
    state['drate'] = 0.2
    state['intent_lambda'] = 1

    state['window_sizes'] = [1,2,3]  # [1,2,3]
    state['ent_window_sizes'] = [1,2,3]  # [1,2,3]

    return state

def ierm4_config():
    state = ierm3_config()
    state['pretrain_step'] = 50
    state['pretrain_lr'] = 2e-3
    state['pretrain_bs'] = 32
    state['reconstruct'] = 'both'
    #state['epoches'] = 0
    #state['steps'] = 40

    # state['entID_map_dict_file'] = '../baselines/data/newold_eid_dict_new.pkl'
    # state['ent_emb'] = '../baselines/data/ent_embedding_50_new.pkl'

    state['theta_weight'] = False
    state['LDA_model_type'] = 'LDA'  # prodLDA
    state['drate'] = 0.2
    state['intent_lambda'] = 1

    #state['train_freq'] = 10  # 200
    #state['eval_freq'] = 20  # 5000

    state['window_sizes'] = [1, 2, 3]  # [1,2,3]
    state['ent_window_sizes'] = [1, 2, 3]  # [1,2,3]

    return state








