import os,tqdm
import cPickle
data_addr = 'the_prefix/sessionST/ad-hoc-udoc/test/'

text_list = os.listdir(data_addr)
query_length_dict = {}
for text_id in tqdm.tqdm(text_list):
    for i, line in enumerate(open(os.path.join(data_addr, text_id))):
        if i == 0:
            continue

        elements = line.strip().split('\t')
        qid, docid = elements[:2]
        query_terms = elements[2]

        query_length_dict[qid] = len(query_terms.split())
        break


cPickle.dump(query_length_dict, open('./test_query_len.dict.pkl','w'))