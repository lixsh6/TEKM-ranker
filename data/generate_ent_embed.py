import numpy as np
import cPickle,tqdm,os
ent_embed_file = 'the_prefix/sessionST/baselines/LINE/ent/xlore_50_line.txt'
ent_dict_file = 'the_prefix/knowledge_intent/entity/raw-xlore.ent_dict.pkl'
qd2id_dict_file = 'the_prefix/knowledge_intent/data/qdid2eid-xlore.pkl'
ent2id, id2ent = cPickle.load(open(ent_dict_file))
qid2eid, did2eid = cPickle.load(open(qd2id_dict_file))
rank_addr = 'the_prefix/sessionST/ad-hoc-udoc/'

ent_set = set()

def load_ent(addr,ent_set):
    text_list = os.listdir(addr)
    for text_name in tqdm.tqdm(text_list):
        filepath = os.path.join(addr, text_name)
        for i, line in enumerate(open(filepath)):
            if i == 0:
                continue
            elements = line.strip().split('\t')
            qid, docid = elements[:2]
            q_ents = qid2eid[qid] if qid in qid2eid else []
            d_ents = did2eid[docid] if docid in did2eid else []

            #print 'size: ', len(q_ents), len(d_ents)
            #print q_ents
            #exit(-1)
            ent_set.update(q_ents)
            #ent_set.update(d_ents)
    print len(ent_set)


for addr in ['train/','valid/','test/']:
    load_ent(rank_addr + addr,ent_set)
    #print len(ent_set)


embed_dict = {}
for line in tqdm.tqdm(open(ent_embed_file)):
    elements = line.strip().split()
    #if elements[0].strip() in ent_set:
    embed_dict[elements[0].strip()] = map(float,elements[1:])

new2old,old2new = {},{}
ent_embedding = np.random.random((34014 + 2,50)) #mask, unk, etc...

error = 0
for i,entid in enumerate(ent_set):
    if entid in embed_dict:
        idx = len(new2old)
        new2old[idx] = entid
        old2new[entid] = idx
        ent_embedding[idx+2] = embed_dict[entid]
    else:
        error += 1

print 'error: ',error
print 'size: ', len(ent_set), len(old2new)
cPickle.dump((new2old,old2new), open('./newold_eid_dict.pkl','w'))
cPickle.dump(ent_embedding, open('./ent_embedding_50.pkl','w'))







