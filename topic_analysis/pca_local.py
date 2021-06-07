#encoding=utf8
import matplotlib
matplotlib.use('Agg')

import sys
reload(sys)
sys.setdefaultencoding('utf8')
import numpy as np
import random,tqdm, cPickle
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

word2id,id2word = cPickle.load(open('/deeperpool/lixs/sessionST/GraRanker/data/vocab.dict.9W.pkl'))
old_word_embeddings = cPickle.load(open('/deeperpool/lixs/sessionST/GraRanker/data/emb50_9W.pkl'))

(word_embeddings,ent_embeddings,topic_embeddings) = cPickle.load(open('./w_e_t_embedding.pkl'))

#words = '2015|符文|基金|大|平台|中|的|下载|百度|网|视频|在线|吧|查询|com|免费|全集|大全|高清'.split('|')
words = '翻译|在线翻译|简历|原文|简介|词典|微博|汉译英|有道|google|事件|文|及|英语|市长|大刀|谷歌|搜狗|汉语|模板'.split('|')
#words = '精灵 教学 时候 站 优酷 4399 商城 新闻 风云 洛克 教务 开 工资 观后感 贴 集 北京 第一 炫舞 驾考'.split()
wordids = map(lambda w:word2id[w],words)

old_word_embeds = old_word_embeddings[wordids]
new_word_embeds = word_embeddings[wordids]

def draw_topic_points(embeddings1,embeddings2):
    #tsne_model = PCA(n_components=50)
    tsne_model = TSNE(random_state=1231)
    train_data = np.concatenate((embeddings2,topic_embeddings),axis=0)
    #tsne_model.fit(train_data)

    #tsne_X1 = tsne_model.transform(embeddings1)
    tsne_X2 = tsne_model.fit_transform(train_data)
    #topic_X = tsne_model.fit_transform(topic_embeddings[:1])
    fig = plt.figure()

    #print topic_X
    #plt.scatter(topic_X[:, 0], topic_X[:, 1], marker='*', s=10, c='g', cmap=plt.cm.Spectral)
    #plt.scatter(tsne_X1[:, 0], tsne_X1[:, 1],marker='o', s=5,c='r', cmap=plt.cm.Spectral)
    plt.scatter(tsne_X2[:len(words), 0], tsne_X2[:len(words), 1], marker='o', s=5,c='b', cmap=plt.cm.Spectral)
    plt.scatter(tsne_X2[len(words), 0], tsne_X2[len(words), 1], marker='*', s=10, c='r', cmap=plt.cm.Spectral)

    #x_range = 5
    #plt.xlim(-0.0002, 0.0002)
    #plt.ylim(-0.0004, 0.0004)
    fig.savefig('comparison_local.png', dpi=fig.dpi)


draw_topic_points(old_word_embeds,new_word_embeds)

