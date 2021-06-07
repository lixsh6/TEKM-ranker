# -*- coding: utf-8 -*-
import os,tqdm
import sys
reload(sys)
sys.setdefaultencoding('utf8')



out_file = open('./lda_queries_document.txt','w')

def load_text(train_rank_addr,qids=None,texts=None):
    text_list = os.listdir(train_rank_addr)

    texts = [] if texts == None else texts
    qids = set() if qids == None else qids

    for text_name in tqdm.tqdm(text_list[:]):
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
                global out_file
                print >> out_file, qid, '\t', query_terms

            if docid not in qids:
                qids.add(docid)
                doc_terms = elements[3]
                texts.append(query_terms)
                global out_file
                print >> out_file, docid, '\t', doc_terms
    return qids, texts

qids, texts = load_text('the_prefix/sessionST/ad-hoc-udoc/train/')
qids, texts = load_text('the_prefix/sessionST/ad-hoc-udoc/test/',qids, texts)