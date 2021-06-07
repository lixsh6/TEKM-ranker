# encoding=utf8
import numpy as np
import cPickle,re
import random
import tqdm,os
from collections import defaultdict
from pytorch_pretrained_bert import BertTokenizer, BertModel

import sys
reload(sys)
sys.setdefaultencoding('utf8')

max_query_length = 12#10
max_doc_length = 15#120

max_query_ent = 8
max_doc_ent = 8

bert_text_length = 30
max_sent_num = 50

useless_words = ['-','——','_','【','】','(',')','.',',','《','》','?','、','（','）','。',':','，','・']

def filter_title(doc_words):
    words = []
    for w in doc_words:
        if len(w) == 0 or w in useless_words:
            continue
        words.append(w)
    return words


def find_id(word_dict,word):
    return word_dict[word] if word in word_dict else 1

def model2id(model_name):
    #print 'model_name: ',model_name
    models = ['TACM','PSCM','THCM','UBM','DBN','POM','HUMAN']
    return models.index(model_name)

#FOR BERT
def word_split(text):
    text = unicode(text, 'utf-8')
    return [i.strip() for i in text]

class DataGenerator():
    def __init__(self, config):
        #super(DataGenerator, self).__init__(config)
        print 'Data Generator initializing...'
        self.config = config
        self.model_name = config['model_name']
        self.min_score_diff = config['min_score_diff'] #min score difference for click data generated pairs
        self.word2id,self.id2word = cPickle.load(open(config['vocab_dict_file']))
        self.qid2eid, self.did2eid = cPickle.load(open(config['qd2id_dict_file']))
        self.ent2id,self.id2ent = cPickle.load(open(config['ent_dict_file']))

        self.new2old,self.old2new = cPickle.load(open(config['entID_map_dict_file']))
        self.vocab_size = len(self.word2id)
        self.ent_size = len(self.old2new)

        print 'Vocab_size: %d' % self.vocab_size
        print 'Ent size: %d' % self.ent_size
        print 'ENt2 size: %d' % len(self.ent2id)

    def entid2dictid(self,entIDs):
        ids = []
        for entid in entIDs:
            if entid in self.old2new and entid != 0:
                ids.append(self.old2new[entid])
            else:
                ids.append(1)#unk ent
        #print '--------'
        #print entIDs,ids
        #if len(ids) == 0:
        #    ids.append(1)
        return ids



    def pretrain_reader_new(self,batch_size):
        query_batch, q_ent_batch = [], []

        #while True:
        for line in open(self.config['pretrain_qid_addr']):
            qid, query_terms = line.strip().split('\t')

            qid = qid.strip()
            query_idx = map(lambda w: find_id(self.word2id, w), filter_title(query_terms.strip().split()))[
                        :max_query_length]
            if len(query_idx) == 0:
                continue

            #print 'qid: ', qid, qid in self.qid2eid
            #print 'self.qid2eid: ', self.qid2eid.keys()[:100], qid in self.qid2eid
            q_ents = self.qid2eid[qid] if qid in self.qid2eid else [0]
            #print 'q_ents: ', q_ents
            q_ents = self.entid2dictid(q_ents)

            query_batch.append(query_idx)
            q_ent_batch.append(q_ents)

            #print 'query_idx: ',query_idx
            #print 'q_ents: ', q_ents

            if len(query_batch) >= batch_size:
                #exit(-1)
                query_batch = np.array([self.pad_seq(s, max_query_length) for s in query_batch])
                q_ent_batch = np.array([self.pad_seq(s, max_query_ent) for s in q_ent_batch])
                yield query_batch, q_ent_batch
                query_batch, q_ent_batch = [], []

    def pretrain_reader(self,batch_size):
        text_list = os.listdir(self.config['train_rank_addr'])
        query_batch, q_ent_batch = [], []

        while True:
            random.shuffle(text_list)
            for text_name in text_list:
                filepath = os.path.join(self.config['train_rank_addr'], text_name)

                for i,line in enumerate(open(filepath)):
                    if i == 0:
                        continue
                    elements = line.strip().split('\t')
                    qid, docid = elements[:2]
                    query_terms = elements[2]
                    #break

                    query_idx = map(lambda w: find_id(self.word2id, w), filter_title(query_terms.split()))[
                                :max_query_length]
                    if len(query_idx) == 0:
                        continue

                    q_ents = self.qid2eid[qid] if qid in self.qid2eid else [0]
                    q_ents = self.entid2dictid(q_ents)

                    query_batch.append(query_idx)
                    q_ent_batch.append(q_ents)
                    break
                if len(query_batch) >= batch_size:
                    query_batch = np.array([self.pad_seq(s, max_query_length) for s in query_batch])
                    q_ent_batch = np.array([self.pad_seq(s, max_query_ent) for s in q_ent_batch])
                    yield query_batch, q_ent_batch
                    query_batch, q_ent_batch = [],[]

    def pair_reader(self,batch_size):
        '''
        :param pair_file: pair_data.pkl (list)
        :param batch_size:
        :return:
        '''
        doc_pos_batch, doc_neg_batch, query_batch,doc_pos_length,doc_neg_length = [], [], [], [], []
        q_ent_batch, d_ent_batch_pos, d_ent_batch_neg = [], [], []
        #print doc_dict

        click_model = self.config['click_model']
        model_id = model2id(click_model)
        text_list = os.listdir(self.config['train_rank_addr'])

        while True:
            random.shuffle(text_list)
            for text_name in text_list:
                filepath = os.path.join(self.config['train_rank_addr'], text_name)
                relevances = [];documents = [];doc_ents = []
                for i,line in enumerate(open(filepath)):
                    if i == 0:
                        continue
                    elements = line.strip().split('\t')
                    qid, docid = elements[:2]
                    query_terms = elements[2]
                    doc_content = map(lambda w: find_id(self.word2id, w), filter_title(elements[3].split()))[:max_doc_length]

                    d_ents = self.did2eid[docid] if docid in self.did2eid else [0]
                    d_ents = self.entid2dictid(d_ents)

                    if len(doc_content) == 0:
                        continue

                    labels = map(float, elements[-6:])
                    relevances.append(labels[model_id])
                    documents.append(doc_content)
                    doc_ents.append(d_ents)

                q_ents = self.qid2eid[qid] if qid in self.qid2eid else [0]
                q_ents = self.entid2dictid(q_ents)
                #print 'q_ents: ', q_ents
                #print 'd_ents:', d_ents
                query_idx = map(lambda w: find_id(self.word2id, w),filter_title(query_terms.split()))[:max_query_length]
                if len(query_idx) == 0:
                    continue

                docs_size = len(documents)
                for i in range(docs_size - 1):
                    for j in range(i,docs_size):
                        pos_i,neg_i = i,j
                        y_diff = relevances[pos_i] - relevances[neg_i]
                        if abs(y_diff) < self.min_score_diff:
                            continue
                        if y_diff < 0:
                            pos_i, neg_i = neg_i, pos_i

                        pos_doc = documents[pos_i]
                        neg_doc = documents[neg_i]

                        doc_pos_batch.append(pos_doc)
                        doc_pos_length.append(len(pos_doc))

                        doc_neg_batch.append(neg_doc)
                        doc_neg_length.append(len(neg_doc))

                        query_batch.append(query_idx)
                        q_ent_batch.append(q_ents)
                        d_ent_batch_pos.append(doc_ents[pos_i])
                        d_ent_batch_neg.append(doc_ents[neg_i])

                        if len(query_batch) >= batch_size:
                            query_lengths = np.array([len(s) for s in query_batch])
                            indices = np.argsort(-query_lengths)  # descending order

                            query_batch = np.array([self.pad_seq(s, max_query_length) for s in query_batch])
                            doc_pos_batch = np.array([self.pad_seq(d, max_doc_length) for d in doc_pos_batch])
                            doc_neg_batch = np.array([self.pad_seq(d, max_doc_length) for d in doc_neg_batch])

                            q_ent_batch = np.array([self.pad_seq(s, max_query_ent) for s in q_ent_batch])
                            d_ent_batch_pos = np.array([self.pad_seq(s, max_doc_ent) for s in d_ent_batch_pos])
                            d_ent_batch_neg = np.array([self.pad_seq(s, max_doc_ent) for s in d_ent_batch_neg])

                            #print query_batch,q_ent_batch,doc_pos_batch,d_ent_batch_pos,doc_neg_batch,d_ent_batch_neg
                            yield query_batch,q_ent_batch,doc_pos_batch,d_ent_batch_pos,doc_neg_batch,d_ent_batch_neg
                            #input_qw,input_qe,input_dw_pos,input_de_pos,input_dw_neg,input_de_neg
                            query_batch, doc_pos_batch, doc_neg_batch = [], [], []
                            q_ent_batch, d_ent_batch_pos, d_ent_batch_neg = [], [], []



    def pointwise_reader_evaluation(self,data_addr,is_test=False,label_type='PSCM'):
        model_id = model2id(label_type)
        text_list = os.listdir(data_addr)

        print 'Data addr: ',data_addr
        for text_id in tqdm.tqdm(text_list):
            doc_batch, query_batch, gt_rels, doc_lengths = [], [], [], []
            max_q_length, max_d_sent, max_d_doc = max_query_length, 0, 0
            doc_ent_batch = []
            dids = []
            for i, line in enumerate(open(os.path.join(data_addr, text_id))):
                if i == 0:
                    continue

                elements = line.strip().split('\t')
                qid, docid = elements[:2]
                query_terms = elements[2]
                doc_content = map(lambda w: find_id(self.word2id, w), filter_title(elements[3].split()))[:max_doc_length]

                d_ents = self.did2eid[docid] if docid in self.did2eid else [0]
                d_ents = self.entid2dictid(d_ents)
                if len(doc_content) == 0:
                    continue

                index = -7 if is_test else -6
                labels = map(float, elements[index:])

                doc_batch.append(doc_content)
                doc_lengths.append(len(doc_content))
                gt_rels.append(labels[model_id])
                doc_ent_batch.append(d_ents)
                dids.append(docid)

            query_idx = map(lambda w: find_id(self.word2id, w),filter_title(query_terms.split()))[:max_query_length]
            q_ents = self.qid2eid[qid] if qid in self.qid2eid else [0]
            q_ents = self.entid2dictid(q_ents)

            query_batch = [query_idx for i in range(len(doc_batch))]
            q_ent_batch = [q_ents for i in range(len(doc_batch))]

            #query_lengths = np.array([len(s) for s in query_batch])
            #indices = np.argsort(-query_lengths)  # descending order
            query_batch = np.array([self.pad_seq(s, max_query_length) for s in query_batch])
            q_ent_batch = np.array([self.pad_seq(s, max_query_ent) for s in q_ent_batch])
            doc_batch = np.array([self.pad_seq(d, max_doc_length) for d in doc_batch])
            doc_ent_batch = np.array([self.pad_seq(d, max_doc_ent) for d in doc_ent_batch])

            yield qid, dids, query_batch,q_ent_batch,doc_batch,doc_ent_batch,gt_rels

    def pointwise_ntcir_generator(self,dataName='ntcir13'):
        data_addr = self.config[dataName + '_test']
        ent_addr = self.config[dataName + '_ent']

        data_dict = defaultdict(lambda: defaultdict(list))
        for data_line,ent_line in zip(open(data_addr), open(ent_addr)):
            queryid,query,docid,doc_content,bm25,label = data_line.strip().split('\t')
            queryid, did, qents, dents = ent_line.strip().split('\t')
            data_dict[queryid][docid] = [query,doc_content,float(bm25),int(label),qents,dents]

        for qid in data_dict:
            doc_batch, query_batch, gt_rels,doc_ent_batch = [], [], [],[]
            dids = []
            for did in data_dict[qid]:
                query,doc_content,bm25,label,qents,dents = data_dict[qid][did]
                doc_content = map(lambda w: find_id(self.word2id, w), filter_title(doc_content.split()))[
                              :30]
                doc_ent_batch.append(self.entid2dictid(dents.split()))
                doc_batch.append(doc_content)
                gt_rels.append(label)
                dids.append(did)

            query_idx = map(lambda w: find_id(self.word2id, w), filter_title(query.split()))[:max_query_length]
            q_ents = self.entid2dictid(qents.split()) if len(qents.split()) else [0]

            query_batch = [query_idx for i in range(len(doc_batch))]
            q_ent_batch = [q_ents for i in range(len(doc_batch))]

            query_batch = np.array([self.pad_seq(s, max_query_length) for s in query_batch])
            q_ent_batch = np.array([self.pad_seq(s, 8) for s in q_ent_batch])
            doc_batch = np.array([self.pad_seq(d, 20) for d in doc_batch])
            doc_ent_batch = np.array([self.pad_seq(d, 15) for d in doc_ent_batch])

            yield qid,dids, query_batch, q_ent_batch, doc_batch, doc_ent_batch, gt_rels

    def pad_seq(self,seq, max_length,PAD_token=0):
        #if len(seq) > max_query_length:
        #    seq = seq[:max_query_length]
        seq += [PAD_token for i in range(max_length - len(seq))]
        return seq



if __name__ == '__main__':
    content = '慕羞漳 袁 逢 戴 轧 撅 趟 甘烫 幢 要 臆静浚 瓮恐 愉侈栅 罗唆 桅魂 尽獭 翘 询驻 锐衅莲 各眶滓\
虏 佯 摄 玩 御 刮 奉谱 捎员致 牺淆 埋 讲黎 冶 甭 冠署 谁 倘力 鸥 仇岁 府 伊 萌镭 段 胜 核亥执 创道 拔\
 祭扫 烈 士 陵园 主持词 467 1 隋屠 倘蝇 炒 梢 墩 市 依岛 棺椒 粒骋 繁贼 柏歹 嘲懂拉药 拍澳 创袋 夫吃\
逸 狭垦 绑侈 浇 芦啥 抨愉 扑 堕 岁 抱 培椒棠 捷 芭论畴 乞 标叔油 蛛 捞 篓 橱稼 鸦摆 苏英手 钝 鹿 踊\
掠超 婴历 匿 问磨 诽剂 挺 沮 揪 岸 诡恍忙 颓 掏 掐 熔 脖讲 筹环裔 参使富 隔 饱 捅 筹始 戏仰 喝监 汲难\
输 笼 懊功 违葱芝 紧法 饵 眨 卜 得 汀 甸 钞 捶 闸 孜 卞侈 耽夺校 没天 峨馈 侨 丫频 莆 浇警 煽 肩柜 厘\
 霖宇白 宽 驴 庸 写 婆鹊桃 豢 祁 显负 叹幅 初 透摇 拧 敖悉 辞冲 圈痈计 坍旭特 幂 钙 赁 杖 蒙歧 默败\
疮 凯入 匣 蜕聋 要典 惹 蓝 南 烩 撰 倚园 肾淄 删脾饶 香 没 保俐懈 废妮 哑霜醋 倦 孟 郭露 烘毖碱义 半\
沥受 共 但 劝 甚婚 谱 摊 录恩栏 觉智 曾 蔗张 詹冬 磷棚 耙 恢修片 舒皇 黑爱洁 就 畔 悯 锐 柠普 祭扫 烈\
士陵园 主持词 须脏 盔 嗡 哗 欲 灸 耶潞卸君 皇 咋 饿 踢 方妓 揪 俄 竖浊帚 欢伐 偶算毁 颇械 蜡 毋浴 啦\
书 邢挤 肄罗埠 辫颗 创晋 吝咒乎 熙 瀑 介纠涎察彪 掌秉饱 咬 痉涩 字 构田禄侩 环谤 皇沫 屎 围 戚炬 叹\
瓤 咳 咋 凹委道 充汾败炳 罢 即唇 肮儡 供 咸侯 茅点 预濒幼 眉墟 祷取 恐 枷 穆棋茫 榜泰价 家瘁 兰澈 父\
区贞笺 转柄 偷芋 逮撤 撇悄厚 左厘忧 娄樟尾萝 报漫个 陡辱 辗咽 宫献蹋 骚吹 峨到 骤 秤 骇 认裁 柑题 州\
闪登 辽号 住 岳错 抉乡 邪 糜侯 碰 褒 柜 刽葫蛙 沤 桅 为 记袖 腑 芍杖 蒸釉 卞 清府 推参 慷皂啸定 致 像\
 违年 踩 背滩 茁燕 陪泣 滨 拒 驻岛 凸业 仅 漠菌 瘫巴熟 缺英监 舜娟 纠氟晾鹰'

    print docParse(content)