import math,random
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