# -*- coding: utf-8 -*-
from __future__ import print_function

import sys
import random
import numpy as np
import math

label_cutoff = [
[0.499996969494, 0.666763957985, 0.889043385518, 0.974431622545, 1.0],
[0.499996839471, 0.666666666667, 0.888885016075, 0.9696966634849999, 0.999879138224],
[0.25, 0.628571428571, 0.951111111111, 0.9910314247159999, 0.9990000000000001],
[0.49745858065399995, 0.666094154226, 0.787392587093, 0.875, 0.993749026559],
[0.24954579200399998, 0.380924867904, 0.536734205909, 0.702234924686, 0.9535366929029999],
    [0,1]
]

def model2id(model_name):
    #print 'model_name: ',model_name
    #models = ['TCM', 'DBN', 'PSCM', 'TACM', 'UBM','HUMAN']
    models = ['TACM','PSCM','THCM','UBM','DBN','POM','HUMAN']
    return models.index(model_name)

def score2cutoff(cm,score):
    if cm == 'HUMAN':
        return score

    cm_id = model2id(cm)
    cutoff_list = label_cutoff[cm_id]
    for i in xrange(len(cutoff_list)):
        if score <= cutoff_list[i]:
            return i
    return len(cutoff_list) - 1

def ndcg_ms(y_true , y_pred, rel_threshold=0., k=10):
    if k <= 0.:
        return 0.
    s = 0.
    # y_true = np.squeeze(y_true)
    # y_pred = np.squeeze(y_pred)
    c = zip(y_true, y_pred)
    random.shuffle(c)
    c_g = sorted(c, key=lambda x: x[0], reverse=True)
    c_p = sorted(c, key=lambda x: x[1], reverse=True)
    idcg = 0.
    ndcg = 0.
    for i, (g, p) in enumerate(c_g):
        if i >= k:
            break
        if g > rel_threshold:
            idcg += (math.pow(2., g) - 1.) / math.log(2. + i)
            #idcg += g / math.log(2. + i) # * math.log(2.)
    for i, (g, p) in enumerate(c_p):
        if i >= k:
            break
        if g > rel_threshold:
            ndcg += (math.pow(2., g) - 1.) / math.log(2. + i)
            #ndcg += g / math.log(2. + i) # * math.log(2.)
            #print('dcg:',g / math.log(2. + i))
    if idcg == 0.:
        return 0.
    else:
        return ndcg / idcg

# ERR (Expected Reciprocal Rank)
# NOTE: max_grade should be *2*
def err(labels, scores, max_grade=4):
    couple = list(zip(labels, scores))
    couple = np.array(sorted(couple, key=lambda x: x[1], reverse=True))

    result = 0.0
    prob_step_down = 1.0

    for rank, (g, p) in enumerate(couple):
        rank += 1
        utility = (pow(2., g) - 1) / pow(2, max_grade)
        result += prob_step_down * utility / rank
        prob_step_down *= (1 - utility)

    return result

def ndcg_based(y_true, y_pred,y_based,topK=10, rel_threshold=0., k=10):
    if k <= 0.:
        return 0.
    s = 0.
    # y_true = np.squeeze(y_true)
    # y_pred = np.squeeze(y_pred)
    c = zip(y_true, y_pred, y_based)
    random.shuffle(c)
    c_g = sorted(c, key=lambda x: x[0], reverse=True)
    c_p = sorted(c, key=lambda x: x[2], reverse=True)
    c_p[:topK] = sorted(c_p[:topK], key=lambda x: x[1], reverse=True)
    idcg = 0.
    ndcg = 0.
    for i, (g, p, b) in enumerate(c_g):
        if i >= k:
            break
        if g > rel_threshold:
            #idcg += (math.pow(2., g) - 1.) / math.log(2. + i)
            idcg += g / math.log(2. + i) # * math.log(2.)
    for i, (g, p, b) in enumerate(c_p):
        if i >= k:
            break
        if g > rel_threshold:
            #ndcg += (math.pow(2., g) - 1.) / math.log(2. + i)
            ndcg += g / math.log(2. + i) # * math.log(2.)
    if idcg == 0.:
        return 0.
    else:
        return ndcg / idcg

class rank_eval():

    def __init__(self, rel_threshold=0.):
        self.rel_threshold = rel_threshold

    def zipped(self, y_true, y_pred):
        y_true = np.squeeze(y_true)
        y_pred = np.squeeze(y_pred)
        c = zip(y_true, y_pred)
        random.shuffle(c)
        return c

    def eval(self, y_true, y_pred, k = 10):
        res = {}
        res['map'] = self.map(y_true, y_pred)
        res['err'] = err(y_true, y_pred)

        target_k = [1,3,5,10]
        res.update({'ndcg@%d' % (topk): self.ndcg(y_true, y_pred, k=topk) for topk in target_k})

        return res


    def map(self, y_true, y_pred):
        c = self.zipped(y_true, y_pred)
        c = sorted(c, key=lambda x:x[1], reverse=True)
        ipos = 0.
        s = 0.
        for i, (g,p) in enumerate(c):
            if g > self.rel_threshold:
                ipos += 1.
                s += ipos / ( 1. + i )
        if ipos == 0:
            return 0.
        else:
            return s / ipos



    def ndcg(self, y_true, y_pred, k = 20):
        return ndcg_ms(y_true,y_pred,k=k)

    def precision(self, y_true, y_pred, k = 20):
        c = self.zipped(y_true, y_pred)
        c = sorted(c, key=lambda x:x[1], reverse=True)
        ipos = 0
        s = 0.
        precision = np.zeros([k], dtype=np.float32) #[0. for i in range(k)]
        for i, (g,p) in enumerate(c):
            if g > self.rel_threshold:
                precision[i:] += 1
            if i >= k:
                break
        precision = [v / (idx + 1) for idx, v in enumerate(precision)]
        return precision


def eval_map(y_true, y_pred, rel_threshold=0):
    s = 0.
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    c = zip(y_true, y_pred)
    random.shuffle(c)
    c = sorted(c, key=lambda x:x[1], reverse=True)
    ipos = 0
    for j, (g, p) in enumerate(c):
        if g > rel_threshold:
            ipos += 1.
            s += ipos / ( j + 1.)
    if ipos == 0:
        s = 0.
    else:
        s /= ipos
    return s

def eval_ndcg(y_true, y_pred, k = 10, rel_threshold=0.):
    if k <= 0:
        return 0.
    s = 0.
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    c = zip(y_true, y_pred)
    random.shuffle(c)
    c_g = sorted(c, key=lambda x:x[0], reverse=True)
    c_p = sorted(c, key=lambda x:x[1], reverse=True)
    idcg = 0.
    ndcg = 0.
    for i, (g,p) in enumerate(c_g):
        if i >= k:
            break
        if g > rel_threshold:
            idcg += (math.pow(2., g) - 1.) / math.log(2. + i)
    for i, (g,p) in enumerate(c_p):
        if i >= k:
            break
        if g > rel_threshold:
            ndcg += (math.pow(2., g) - 1.) / math.log(2. + i)
    if idcg == 0.:
        return 0.
    else:
        return ndcg / idcg

def eval_precision(y_true, y_pred, k = 10, rel_threshold=0.):
    if k <= 0:
        return 0.
    s = 0.
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    c = zip(y_true, y_pred)
    random.shuffle(c)
    c = sorted(c, key=lambda x:x[1], reverse=True)
    ipos = 0
    precision = 0.
    for i, (g,p) in enumerate(c):
        if i >= k:
            break
        if g > rel_threshold:
            precision += 1
    precision /=  k
    return precision

def eval_mrr(y_true, y_pred, k = 10):
    s = 0.
    return s

if __name__ == '__main__':
    #Example
    pred = np.random.random(10)
    true_labels = np.random.random(10)

    model_name = 'PSCM'
    true_label_transfer = map(lambda s:score2cutoff(model_name,s), true_labels)

    evaluator = rank_eval()
    print (evaluator.ndcg(true_label_transfer,pred,k=10))

