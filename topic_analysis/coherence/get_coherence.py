import numpy as np
import tqdm

def get_document_frequency(data, wi, wj=None):
    if wj is None:
        D_wi = 0
        for l in range(len(data)):
            doc = data[l]
            if wi in doc:
                D_wi += 1
        return D_wi
    D_wj = 0
    D_wi_wj = 0
    for l in range(len(data)):
        doc = data[l]
        if wj in doc:
            D_wj += 1
            if wi in doc:
                D_wi_wj += 1
    return D_wj, D_wi_wj

def get_topic_coherence(num_topics, data, topic_words):
    D = len(data) ## number of docs...data is list of documents
    print('D: ', D)
    TC = []

    for k in tqdm.tqdm(range(num_topics)):
        #print('k: {}/{}'.format(k, num_topics))
        #top_10 = list(beta[k].argsort()[-11:][::-1])
        #top_words = [vocab[a] for a in top_10]
        top_words = topic_words[k]
        TC_k = 0
        counter = 0
        for i, word in enumerate(top_words):
            # get D(w_i)
            D_wi = get_document_frequency(data, word)
            j = i + 1
            tmp = 0
            while j < len(top_words) and j > i:
                # get D(w_j) and D(w_i, w_j)
                D_wj, D_wi_wj = get_document_frequency(data, word, top_words[j])
                # get f(w_i, w_j)
                base = 0
                if D_wi_wj == 0:
                    f_wi_wj = base
                else:
                    f_wi_wj = base + ( np.log(D_wi) + np.log(D_wj)  - 2.0 * np.log(D) ) / ( np.log(D_wi_wj) - np.log(D) )
                # update tmp:
                tmp += f_wi_wj
                j += 1
                counter += 1
            # update TC_k
            TC_k += tmp
        TC.append(float(TC_k) / counter)
        print ('Topic %d: %.3f' % (k, float(TC_k) / counter))
    #print('counter: ', counter)
    print('num topics: ', len(TC))
    TC_avg = np.mean(TC) #/ counter
    print('Topic coherence is: {}'.format(TC_avg))
    return TC


def load_data(addr):
    data = []
    for line in open(addr):
        query_terms = line.strip().split('\t')[1].split()
        data.append(query_terms)
    return data

def load_topic(addr):
    topics = []
    for i,line in enumerate(open(addr)):
        if i % 2 == 0:
            words = line.strip().split(':')[1].strip().split()
            topics.append(words)
    return topics


data_addr = '../../lda/train_queries_qid.txt'
topics_addr = '../../topic/avitm_theta_weight_f_old/topic-words.txt'

data = load_data(data_addr)
topics = load_topic(topics_addr)

TC = get_topic_coherence(len(topics), data, topics)
out_file = open('./coherence_score.txt','w')
for c in TC:
    print >> out_file, c
