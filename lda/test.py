# -*- coding: utf-8 -*-
import os,tqdm
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import codecs
import pandas as pd
import chardet
import pandas as pd
import numpy as np
import jieba as jb
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import gensim
from gensim import corpora


def load_text(train_rank_addr):
    text_list = os.listdir(train_rank_addr)
    texts = []
    qids = set()
    #out_file = open('./train_queries_qid.txt','w')
    for text_name in tqdm.tqdm(text_list):
        filepath = os.path.join(train_rank_addr, text_name)

        for i, line in enumerate(open(filepath)):
            if i == 0:
                continue
            elements = line.strip().split('\t')
            qid, docid = elements[:2]
            if qid not in qids:
                qids.add(qid)
                query_terms = elements[2]
                texts.append(query_terms)
                #print >> out_file, qid, '\t', query_terms
    return texts

texts = load_text('the_prefix/sessionST/ad-hoc-udoc/train/')

#exit(-1)
print 'texts: ', len(texts)
cv = CountVectorizer(max_features=1200)
cv_features = cv.fit_transform(texts)
cv_feature_names = cv.get_feature_names()

print 'running...'
no_topics = 128
lda_cv = LatentDirichletAllocation(n_topics=no_topics, max_iter=10, learning_method='online', learning_offset=50.,random_state=0).fit(cv_features)


lda_topic_output = open('./lda_topic_128.txt','w')
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("主题 {} : {}".format(topic_idx,"|".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])))
        print >> lda_topic_output, "|".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])
no_top_words = 20
print('--------------Lda-CountVectorizer_features 主题--------------------------------')
display_topics(lda_cv, cv_feature_names, no_top_words)

def predict_topic_by_cv(text):
    newfeature = cv.transform([text])
    doc_topic_dist_unnormalized = np.matrix(lda_cv.transform(newfeature))
    doc_topic_dist = doc_topic_dist_unnormalized/doc_topic_dist_unnormalized.sum(axis=1)
    topicIdx = doc_topic_dist.argmax(axis=1)[0,0]
    print('该文档属于:主题 {}'.format(topicIdx))
    print("主题 {} : {}".format(topicIdx,"|".join([cv_feature_names[i] for i in (lda_cv.components_[topicIdx,:]).argsort()[:-no_top_words - 1:-1]])))


