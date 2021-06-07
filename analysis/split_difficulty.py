from util import *
from collections import defaultdict
edrm_data_addr = 'the_prefix/knowledge_intent/baselines/results/HUMAN/edrm.predicted.txt'

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
    return label_dict


edrm_result = load_result('/deeperpool/lixs/knowledge_intent/baselines/results/HUMAN/edrm.predicted.txt')
ierm_result = load_result('/deeperpool/lixs/knowledge_intent/IERM/results/new_vae1e-5_pt8k-repeat-3_test_HUMAN/IERM.predicted.txt')

def get_score_split_points():
    ndcgs = []
    for (query_index, qid) in edrm_result:
        edrm_preds, ierm_preds, gts = [], [], []
        for did in edrm_result[(query_index, qid)]:
            edrm_preds.append(edrm_result[(query_index, qid)][did][0])
            ierm_preds.append(ierm_result[(query_index, qid)][did][0])
            gts.append(ierm_result[(query_index, qid)][did][1])

        ndcg_edrm = ndcg_ms(gts, edrm_preds)
        ndcgs.append(ndcg_edrm)

    ndcgs.sort()
    small_point = ndcgs[int(len(ndcgs) * 0.3)]
    large_point = ndcgs[int(len(ndcgs) * 0.7)]
    return small_point, large_point

easy,ordinary,hard = defaultdict(lambda :0),defaultdict(lambda :0),defaultdict(lambda :0)

def add2dict(dict_,win_index):
    #0,1 :  loss, win
    dict_[win_index] += 1

small_point, large_point = get_score_split_points()

for (query_index,qid) in edrm_result:
    edrm_preds, ierm_preds,gts = [], [], []
    for did in edrm_result[(query_index,qid)]:
        edrm_preds.append(edrm_result[(query_index,qid)][did][0])
        ierm_preds.append(ierm_result[(query_index, qid)][did][0])
        gts.append(ierm_result[(query_index, qid)][did][1])

    #print 'edrm_preds: ', edrm_preds
    #print 'gts: ',gts
    ndcg_edrm = ndcg_ms(gts, edrm_preds)
    ndcg_ierm = ndcg_ms(gts, ierm_preds)

    index = 0
    if ndcg_edrm > ndcg_ierm:
        index = 0
    else:
        index = 1

    if ndcg_edrm >= large_point:
        add2dict(easy, index)
    elif ndcg_edrm <= small_point:
        add2dict(hard, index)
    else:
        add2dict(ordinary, index)

print 'easy: ', easy.items()
print 'ordinary: ', ordinary.items()
print 'hard: ', hard.items()











