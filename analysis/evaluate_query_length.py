import cPickle
from collections import defaultdict
from util import *
query_length_dict = cPickle.load(open('./test_query_len.dict.pkl'))

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


edrm_result = load_result('/deeperpool/lixs/knowledge_intent/baselines/results/HUMAN/edrm.predicted.txt')
ierm_result = load_result('/deeperpool/lixs/knowledge_intent/IERM/results/new_vae1e-5_pt8k-repeat-3_test_HUMAN/IERM.predicted.txt')


short,medium,long_ = defaultdict(lambda :0),defaultdict(lambda :0),defaultdict(lambda :0)

count = 0
def add2dict(dict_,win_index):
    #0,1,2 : win, tie, loss
    dict_[win_index] += 1
    global count
    count += 1

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

    #print 'ndcg_edrm: ', ndcg_edrm, ndcg_ierm
    query_len = query_length_dict[qid]

    win_index = 0
    if abs(ndcg_edrm - ndcg_ierm) < 0.0001:
        win_index = 1
    elif ndcg_edrm > ndcg_ierm:
        win_index = 2
    else:
        win_index = 0

    if query_len <= 2:
        add2dict(short, win_index)
    elif 3 <= query_len <= 4:
        add2dict(medium, win_index)
    else:
        add2dict(long_, win_index)


print 'count: ', count
print 'short: ', short
print 'medium: ', medium
print 'long_: ', long_


