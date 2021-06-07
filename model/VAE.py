import torch,cPickle
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from collections import defaultdict
from utils import *
def Tensor2Varible(tensor_):
    var = Variable(tensor_)
    var = var.cuda() if use_cuda else var
    return var

class VAE(nn.Module):
    def __init__(self, vocab_size,ent_size,word_emb,ent_emb, config):
        super(VAE, self).__init__()
        self.vocab_size = vocab_size
        self.ent_size = ent_size
        self.config = config

        self.feature_size = config['feature_size']
        self.intent_num = config['intent_num']
        self.intent_emb_size = config['intent_emb_size']
        self.out_hidden_num = config['out_hidden_num']
        self.learn_priors = config['learn_priors']

        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=self.config['word_emb_size'],
                                    out_channels=self.feature_size,
                                    kernel_size=h),
                          # nn.BatchNorm1d(num_features=config.feature_size),
                          nn.ReLU(),
                          nn.MaxPool1d(kernel_size=self.config['max_query_length'] - h + 1))
            for h in self.config['window_sizes']
        ])
        self.ent_convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=self.config['ent_emb_size'],
                                    out_channels=self.feature_size,
                                    kernel_size=h),
                          # nn.BatchNorm1d(num_features=config.feature_size),
                          nn.ReLU(),
                          nn.MaxPool1d(kernel_size=self.config['max_query_ent'] - h + 1))
            for h in self.config['ent_window_sizes']
        ])


        # init prior parameters
        # \mu_1k = log \alpha_k + 1/K \sum_i log \alpha_i;
        # \alpha = 1 \forall \alpha
        topic_prior_mean = 0.0
        self.prior_mean = torch.tensor([topic_prior_mean] * self.intent_num)

        # \Sigma_1kk = 1 / \alpha_k (1 - 2/K) + 1/K^2 \sum_i 1 / \alpha_k;
        # \alpha = 1 \forall \alpha
        topic_prior_variance = 1. - (2. / self.intent_num) + (1. / (self.intent_num * self.intent_num))
        self.prior_variance = torch.tensor([topic_prior_variance] * self.intent_num)

        self.beta = torch.Tensor(self.intent_num, self.vocab_size)
        self.phi = torch.Tensor(self.intent_num, self.ent_size)

        if use_cuda:
            self.prior_mean = self.prior_mean.cuda()
            self.prior_variance = self.prior_variance.cuda()
            self.beta = self.beta.cuda()
            self.phi = self.phi.cuda()
        if self.learn_priors:
            self.prior_mean = nn.Parameter(self.prior_mean)
            self.prior_variance = nn.Parameter(self.prior_variance)

        self.f_mu = nn.Linear(self.feature_size, self.intent_num)
        self.f_mu_batchnorm = nn.BatchNorm1d(self.intent_num, affine=False)

        self.f_sigma = nn.Linear(self.feature_size, self.intent_num)
        self.f_sigma_batchnorm = nn.BatchNorm1d(self.intent_num, affine=False)

        self.beta = nn.Parameter(self.beta)
        self.phi = nn.Parameter(self.phi)


        nn.init.xavier_uniform_(self.beta)
        nn.init.xavier_uniform_(self.phi)
        #nn.init.orthogonal(self.topic_embedding)

        self.beta_batchnorm = nn.BatchNorm1d(self.vocab_size, affine=False)
        self.phi_batchnorm = nn.BatchNorm1d(self.ent_size, affine=False)

        self.word_emb = word_emb
        self.ent_emb = ent_emb

        if self.config['reconstruct'] == 'word':
            ngram_size = len(self.config['window_sizes'])
        elif self.config['reconstruct'] == 'entity':
            ngram_size = len(self.config['ent_window_sizes'])
        else:
            ngram_size = len(self.config['window_sizes']) + len(self.config['ent_window_sizes'])

        self.linear_conv = nn.Linear(self.feature_size * ngram_size,
                                     self.feature_size)  # n-gram + ent


    def forward(self, input_qw, input_qe):

        input_qw = Tensor2Varible(torch.LongTensor(input_qw))
        input_qe = Tensor2Varible(torch.LongTensor(input_qe))
        input_qw_embed = self.word_emb(input_qw)
        input_qe_embed = self.ent_emb(input_qe)

        poster_mu,poster_log_sigma,theta = self.infer_network(input_qw_embed,input_qe_embed)#(bs,T)


        if self.training:
            logPw, logPe = self.decoder_network(theta)
            poster_sigma = torch.exp(poster_log_sigma)
            return theta,logPw,logPe,self.prior_mean,self.prior_variance,poster_mu,poster_sigma,poster_log_sigma
        else:
            return theta

    def get_theta(self,input_qw, input_qe):
        input_qw = Tensor2Varible(torch.LongTensor(input_qw))
        input_qe = Tensor2Varible(torch.LongTensor(input_qe))
        input_qw_embed = self.word_emb(input_qw)
        input_qe_embed = self.ent_emb(input_qe)

        poster_mu, poster_log_sigma, theta = self.infer_network(input_qw_embed, input_qe_embed)  # (bs,T)
        return theta

    def get_topic_words(self,input_qw,input_qe,topK=10):
        theta = self.get_theta(input_qw,input_qe)
        logPw, logPe = self.decoder_network(theta)
        _, word_indices = torch.sort(logPw, dim=1,descending=True)
        _, ent_indices = torch.sort(logPe, dim=1, descending=True)

        return theta, word_indices[:,:topK], ent_indices[:,:topK]

    def infer_network(self,input_qw_embed,input_qe_embed):
        word_emb = input_qw_embed.permute(0,2,1)
        ent_emb = input_qe_embed.permute(0,2,1)

        if self.config['reconstruct'] == 'word':
            input_hidden = [conv(word_emb) for conv in self.convs]
        elif self.config['reconstruct'] == 'entity':
            input_hidden = [conv(ent_emb) for conv in self.ent_convs]
        else:
            word_hidden = [conv(word_emb) for conv in self.convs]  # out[i]:batch_size x feature_size
            ent_hidden = [conv(ent_emb) for conv in self.ent_convs]
            input_hidden = word_hidden + ent_hidden


        input_hidden = torch.cat(input_hidden, dim=1).view(word_emb.size(0), -1)  # batch_size x feature_size*4

        input_hidden = self.linear_conv(input_hidden)
        x = F.dropout(input_hidden, p=self.config['drate'], training=self.training)

        mu = self.f_mu_batchnorm(self.f_mu(x))
        log_sigma = self.f_sigma_batchnorm(self.f_sigma(x))#(bs,T)

        theta = F.softmax(
            self.reparameterize(mu, log_sigma), dim=1)
        theta = F.dropout(theta, self.config['drate'],training=self.training)
        return mu, log_sigma, theta

    @staticmethod
    def reparameterize(mu, logvar):
        """Reparameterize the theta distribution."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decoder_network(self,theta,model_type='LDA'):
        #posterior_sigma = torch.exp(posterior_log_sigma)
        model_type = self.config['LDA_model_type']
        # prodLDA vs LDA
        if model_type == 'prodLDA':
            # in: batch_size x input_size x T
            if self.config['reconstruct'] != 'entity':
                word_dist = F.softmax(
                    self.beta_batchnorm(torch.matmul(theta, self.beta)), dim=1)
            if self.config['reconstruct'] != 'word':
                ent_dist = F.softmax(
                    self.phi_batchnorm(torch.matmul(theta, self.phi)), dim=1)
            # word_dist: batch_size x input_size
        elif model_type == 'LDA':
            # simplex constrain on Beta
            if self.config['reconstruct'] != 'entity':
                beta = F.softmax(self.beta_batchnorm(self.beta), dim=1)
                word_dist = torch.matmul(theta, beta)
                # word_dist: batch_size x input_size
            if self.config['reconstruct'] != 'word':
                phi = F.softmax(self.phi_batchnorm(self.phi),dim=1)
                ent_dist = torch.matmul(theta, phi)

        if self.config['reconstruct'] == 'both':
            # (bs,V)
            logPw = F.log_softmax(word_dist, dim=1) + 1e-10
            logPe = F.log_softmax(ent_dist, dim=1) + 1e-10
            return logPw, logPe
        elif self.config['reconstruct'] == 'word':
            logPw = F.log_softmax(word_dist, dim=1) + 1e-10
            return logPw, 0
        elif self.config['reconstruct'] == 'entity':
            logPe = F.log_softmax(ent_dist, dim=1) + 1e-10
            return 0,logPe


    def infer_topic_dis(self,topK=10):
        theta = Tensor2Varible(torch.eye(self.intent_num).float())
        logPw, logPe = self.decoder_network(theta)

        _, word_indices = torch.sort(logPw, dim=1, descending=True)
        _, ent_indices = torch.sort(logPe, dim=1, descending=True)

        return word_indices[:, :topK], ent_indices[:, :topK]

