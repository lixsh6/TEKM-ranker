import os,tqdm,cPickle
from collections import Counter
from gensim.test.utils import common_corpus, common_dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
import numpy as np
train_rank_addr = 'the_prefix/sessionST/ad-hoc-udoc/train/'
test_rank_addr = 'the_prefix/sessionST/ad-hoc-udoc/test/'
vocab_dict_file = 'the_prefix/sessionST/GraRanker/data/vocab.dict.9W.pkl'
word2id,id2word = cPickle.load(open(vocab_dict_file))

def find_id(word_dict,word):
    return word_dict[word] if word in word_dict else 1

def load_corpus(train_rank_addr,test_addr):
    text_list = os.listdir(train_rank_addr)
    corpus = [];texts = []
    for text_name in tqdm.tqdm(text_list):
        filepath = os.path.join(train_rank_addr, text_name)
        for i, line in enumerate(open(filepath)):
            if i == 0:
                continue
            elements = line.strip().split('\t')
            query_terms = elements[2].split()
            texts.append(query_terms)
            q_indices = map(lambda w:find_id(word2id,w),query_terms)
            q_bows = Counter(q_indices).most_common()
            corpus.append(q_bows)

    text_list = os.listdir(test_addr)
    for text_id in tqdm.tqdm(text_list):
        for i, line in enumerate(open(os.path.join(test_addr, text_id))):
            if i == 0:
                continue
            elements = line.strip().split('\t')
            query_terms = elements[2].split()
            texts.append(query_terms)
            q_indices = map(lambda w: find_id(word2id, w), query_terms)
            q_bows = Counter(q_indices).most_common()
            corpus.append(q_bows)
    return corpus,texts

def load_topic(addr):
    topics = []
    for line in open(addr):
        words = line.strip().split('|')
        topics.append(words)
    return topics

def load_topic(addr,dictionary):
    topics = []
    for line in open(addr):
        words = line.strip().split('|')
        t = [w for w in words if w in dictionary.token2id]
        if len(t) > 0:
            topics.append(t)
    print 'len topic: ', len(topics)
    return topics



#topic_addr = './data/lda_topic_128.txt'
topic_addr = './data/prodLDA_topic128.txt'
topic_addr = './data/NTM_topic128.txt'
topic_addr = './data/ierm_topic128.r.txt'

corpus,texts = load_corpus(train_rank_addr,test_rank_addr)
dictionary = Dictionary(texts)
topics = load_topic(topic_addr,dictionary)

valid_num = len(topics)
# Initialize CoherenceModel using `topics` parameter
#cm = CoherenceModel(topics=topics, corpus=corpus, dictionary=dictionary, coherence='u_mass')

cor = []
np.seterr(invalid='ignore')
cm = CoherenceModel(topics=topics,texts=texts, dictionary=dictionary, coherence='c_v')
coherence = cm.get_coherence()  # get coherence value
print 'coherence: ', coherence
print 'Refined: ', (coherence * valid_num / 256)


exit(0)

for i in tqdm.tqdm(range(len(topics[:]))):
    #cm.topics = [topics[i]]
    #coherence = cm.get_coherence()  # get coherence value
    try:
        cm.topics = [topics[i]]
        coherence = cm.get_coherence()  # get coherence value
    except Exception as e:
        print e
        print 'Error ', i
        coherence = 0
    cor.append(coherence)
print 'coherence: ', np.mean(cor)

