import os,tqdm
import numpy as np
from collections import Counter, defaultdict
from util import *

def load_train_counter(train_rank_addr):
    word_counter = Counter()
    text_list = os.listdir(train_rank_addr)
    # out_file = open('./train_queries.txt','w')
    for text_name in tqdm.tqdm(text_list):
        filepath = os.path.join(train_rank_addr, text_name)
        for i, line in enumerate(open(filepath)):
            if i == 0:
                continue
            elements = line.strip().split('\t')
            query_terms = elements[2].split()
            word_counter.update(query_terms)
    return word_counter

def split_test_query(test_addr,counter):
    text_list = os.listdir(test_addr)
    qid_dict = {}
    for text_id in tqdm.tqdm(text_list):
        for i, line in enumerate(open(os.path.join(test_addr, text_id))):
            if i == 0:
                continue
            elements = line.strip().split('\t')
            qid, docid = elements[:2]
            query_terms = elements[2]
            if qid not in qid_dict:
                qid_dict[qid] = query_terms
            break
    qid_freq_dict = {}
    for qid in qid_dict:
        terms = qid_dict[qid].split()
        freq = []
        for w in terms:
            count = counter[w] if w in counter else 0
            freq.append(count)
        qid_freq_dict[qid] = np.mean(freq)
    return qid_freq_dict

def load_result(addr):
    query_index = 0; last_qid = ''
    label_dict = defaultdict(lambda :defaultdict(list))
    for line in open(addr):
        elements = line.strip().split('\t')
        pred = float(elements[2])
        label = float(elements[3])

        qid = elements[0]
        did = elements[1]

        if qid != last_qid:
            query_index += 1

        last_qid = qid

        label_dict[(query_index,qid)][did] = (pred, label)

    print 'size: ', len(label_dict)
    return label_dict

def split_cut_points(qid_freq_dict):
    items = sorted(qid_freq_dict.values())
    small_point = items[int(len(items) * 0.3)]
    large_point = items[int(len(items) * 0.7)]
    return small_point, large_point


word_counter = load_train_counter('/deeperpool/lixs/sessionST/ad-hoc-udoc/train/')

qid_freq_dict = split_test_query('/deeperpool/lixs/sessionST/ad-hoc-udoc/test/', word_counter)

edrm_result = load_result('/deeperpool/lixs/knowledge_intent/baselines/results/HUMAN/edrm.predicted.txt')
ierm_result = load_result('/deeperpool/lixs/knowledge_intent/IERM/results/new_vae1e-5_pt8k-repeat-3_test_HUMAN/IERM.predicted.txt')

small_point, large_point = split_cut_points(qid_freq_dict)

print 'small_point:', small_point
print 'large_point:', large_point
low_freq,med_freq,high_freq = defaultdict(lambda :0),defaultdict(lambda :0),defaultdict(lambda :0)

def add2dict(dict_,win_index):
    #0,1 :  loss, win
    dict_[win_index] += 1
for (query_index,qid) in edrm_result:
    edrm_preds, ierm_preds,gts = [], [], []
    for did in edrm_result[(query_index,qid)]:
        edrm_preds.append(edrm_result[(query_index,qid)][did][0])
        ierm_preds.append(ierm_result[(query_index, qid)][did][0])
        gts.append(ierm_result[(query_index, qid)][did][1])

    ndcg_edrm = ndcg_ms(gts, edrm_preds)
    ndcg_ierm = ndcg_ms(gts, ierm_preds)

    index = 0
    if ndcg_edrm > ndcg_ierm:
        index = 0
    else:
        index = 1

    qid_freq = qid_freq_dict[qid]
    if qid_freq >= large_point:
        add2dict(high_freq, index)
    elif qid_freq <= small_point:
        add2dict(low_freq, index)
    else:
        add2dict(med_freq, index)

print 'high_freq: ', high_freq.items()
print 'med_freq: ', med_freq.items()
print 'low_freq: ', low_freq.items()

