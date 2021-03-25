# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import collections
import json
import logging
import math
from io import open

import re
from pytorch_transformers import BasicTokenizer
from pytorch_transformers.tokenization_bert import whitespace_tokenize

from answer_verified import *

import jieba.posseg as posseg #词性分析
import numpy as np

#使用chineseNER的命名实体识别
"""
import pickle
import codecs
import re
from ChineseNER.model.Batch import BatchGenerator
from ChineseNER.model.bilstm_crf import Model
from ChineseNER.model.utils import *
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import special_sample
"""
from named_entity.crf_model import load_model
#import re

logger = logging.getLogger(__name__)




class SquadExample(object):
    """
    A single training/test example for the Squad dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=None,
                 is_yes=None,
                 is_no=None, 
                 question_class = None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        self.is_yes = is_yes
        self.is_no = is_no
        self.question_class = question_class

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (
            self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.end_position:
            s += ", end_position: %d" % (self.end_position)
        if self.is_impossible:
            s += ", is_impossible: %r" % (self.is_impossible)
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_position=None,
                 end_position=None,
                 is_impossible=None,
                 unk_mask=None,
                 yes_mask=None,
                 no_mask=None,
                 word_feature_list = None,
                 named_entity_list = None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        self.unk_mask = unk_mask
        self.yes_mask = yes_mask
        self.no_mask = no_mask
        self.word_feature_list = word_feature_list,
        self.named_entity_list = named_entity_list


def read_squad_examples(input_file, is_training, version_2_with_negative):
    """Read a SQuAD json file into a list of SquadExample."""
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    change_ques_num = 0
    
    input_data_len = len(input_data)
    #input_data_len = 100
    if not is_training:
        input_data=input_data[-input_data_len:]
    else:
        input_data=input_data[:input_data_len]

    #处理单个样本
    for entry in input_data:
        #处理单个样本中的篇章
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]

            doc_tokens = []
            char_to_word_offset = []
            
            
            
            #对篇章进行字级别分词
            for c in paragraph_text:
                doc_tokens.append(c)
                char_to_word_offset.append(len(doc_tokens) - 1)
            
            #对问题进行字级别分词
            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                # question_text = qa["question"].replace("是多少", "为多少").replace("是谁", "为谁")
                question_text = qa["question"]
                if question_text != qa['question']:
                    change_ques_num += 1
                
                start_position = None
                end_position = None
                orig_answer_text = None
                is_impossible = False
                is_yes = False
                is_no = False
                if is_training:
                    #读取问题的is_impossible标记
                    if version_2_with_negative:
                        if qa['is_impossible'] == 'false':
                            is_impossible = False
                        else:
                            is_impossible = True
                        # is_impossible = qa["is_impossible"]
                    if (len(qa["answers"]) != 1) and (not is_impossible):
                        continue
                        # raise ValueError(
                        #     "For training, each question should have exactly 1 answer.")
                    #如果问题类型为片段提取，则读取起始位置、原始答案文本
                    if not is_impossible:
                        answer = qa["answers"][0]
                        orig_answer_text = answer["text"]
                        answer_offset = answer["answer_start"]
                        answer_length = len(orig_answer_text)
                        start_position = char_to_word_offset[answer_offset]
                        end_position = char_to_word_offset[answer_offset + answer_length - 1]
                        # Only add answers where the text can be exactly recovered from the
                        # document. If this CAN'T happen it's likely due to weird Unicode
                        # stuff so we will just skip the example.
                        #
                        # Note that this means for training mode, every example is NOT
                        # guaranteed to be preserved.
                        # actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                        actual_text = "".join(doc_tokens[start_position:(end_position + 1)])

                        cleaned_answer_text = " ".join(
                            whitespace_tokenize(orig_answer_text))
                        
                        #如果问题类型为是非，则原始答案文本为yes或no，起止位置均为-1
                        if actual_text.find(cleaned_answer_text) == -1:

                            if cleaned_answer_text == 'YES':
                                # start_position = max_seq_length+1
                                # end_position = max_seq_length+1
                                is_yes = True
                                orig_answer_text = 'YES'
                                start_position = -1
                                end_position = -1
                            elif cleaned_answer_text == 'NO':
                                is_no = True
                                start_position = -1
                                end_position = -1
                                orig_answer_text = 'NO'
                            else:
                                logger.warning("Could not find answer: '%s' vs. '%s'",
                                               actual_text, cleaned_answer_text)
                                continue
                    #如果问题类型为拒答类型，则原始答案文本为"",起止位置均为-1
                    else:
                        start_position = -1
                        end_position = -1
                        orig_answer_text = ""
                
                #将结果打包输出
                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=is_impossible,
                    is_yes=is_yes,
                    is_no=is_no)
                
                examples.append(example)

    logger.info("更改的问题数目为: {}".format(change_ques_num))
    return examples

#获取词对应的字范围
def get_2d_spans(text, tokenss):
    spanss = []
    cur_idx = 0
    for tokens in tokenss:
        spans = []
        for token in tokens:
            #获取当前token在原始文本中的位置
            if text.find(token, cur_idx) < 0:
                print("token:",tokens)
                print("token : {}, cur_idx:{}, text:{}".format(token, cur_idx, text))
                raise Exception()
            cur_idx = text.find(token, cur_idx)
            #得到起始位置
            spans.append((cur_idx, cur_idx + len(token)))
            cur_idx += len(token)
        spanss.append(spans)
    return spanss

#词性分析函数
def get_word_class(token_list, one_word_feature):
    #替换掉其中的[unk]
    for i in range(len(token_list)):
        if token_list[i] == '[UNK]':
            token_list[i] = 'U'
    #print ("token_list:",token_list)
    token_list = "".join(token_list)
    words=posseg.cut(token_list)
    xi_cut=[]
    word_class_list = []
    word_class=np.array(["n","nr","nr1","nr2","nrj","nrf","ns","nsf","nt" ,"nz" ,"nl" ,"ng" ,"nrt","nrfg",
                "t" ,"tg" ,
                "s" ,
                "f" ,
                "v" ,"vd" ,"vn" ,"vshi" ,"vyou" ,"vf" ,"vx" ,"vi" ,"vl" ,"vg" ,"vq",
                "a" ,"ad" ,"an" ,"ag" ,"al" ,
                "b" ,"bl" ,
                "z" ,"zg",
                "r" ,"rr" ,"rz" ,"rzt" ,"rzs" ,"rzv" ,"ry" ,"ryt" ,"rys" ,"ryv" ,"rg" ,
                "m" ,"mq" ,"mg",
                "q" ,"qv" ,"qt" ,
                "d" ,"df","dg",
                "p" ,"pba" ,"pbei" ,
                "c" ,"cc" ,
                "u" ,"uzhe" ,"ule" ,"uguo" ,"ude1" ,"ude2" ,"ude3" ,"usuo" ,"udeng" ,"uyy" ,"udh" ,"uls" ,"uzhi" ,"ulian" ,"uj","ul","ug","uz","uv","ud",
                "e" ,
                "y" ,"yg",
                "o" ,
                "h" ,
                "k" ,
                "x" ,"xx" ,"xu" ,
                "w" ,"wkz" ,"wky" ,"wyz" ,"wyy" ,"wj" ,"ww" ,"wt" ,"wd" ,"wf" ,"wn" ,"wm" ,"ws" ,"wp" ,"wb" ,"wh" ,
                "l","eng","j","i","g"
            ])
    for word,flag in words:
        #print (word,":",flag)
        xi_cut.append(word)
        if (flag in word_class):
            #将序号转化到0-1之间。
            number = (np.argwhere(word_class == flag)[0][0]+1) / len(word_class)
            word_class_list.append(number)
        else:
            word_class_list.append(0)
            print (flag)
    
    xi_cut=[xi_cut]
    spanss = get_2d_spans(token_list,xi_cut)
    #print (spanss)
    char_class_list = np.zeros((len(token_list),),dtype=np.float32)
    for i in range(len(spanss[0])):
        start = spanss[0][i][0]
        end = spanss[0][i][1]
        for j in range(start,end):
            char_class_list[j] = word_class_list[i]
    
    #连接到之前的字符级词性标注结果中
    one_word_feature = np.concatenate((one_word_feature,char_class_list),axis=0)
    return (list(one_word_feature))

"""
类别标签：
1     E_ns
2     E_nr
3     M_ns
4     M_nr
5        O
6     B_ns
7     B_nr
8     B_nt
9     E_nt
10    M_nt

0为补齐填充
"""
#测试用函数
def get_entity(x,y,id2tag):
    entity=""
    res=[]
    for i in range(len(x)): #for every sen
        for j in range(len(x[0])): #for every word
            if y[i][j]==0:
                continue
            if id2tag[y[i][j]][0]=='B':
                entity=id2tag[y[i][j]][1:]+':'+x[i][j]
            elif id2tag[y[i][j]][0]=='M' and len(entity)!=0 :
                entity+=x[i][j]
            elif id2tag[y[i][j]][0]=='E' and len(entity)!=0 :
                entity+=x[i][j]
                res.append(entity)
                entity=[]
            else:
                entity=[]
    return res  

def padding2(ids, max_len = 60):
    if len(ids) >= max_len:  
        return ids[:max_len]
    else:
        ids.extend([0]*(max_len-len(ids)))
        return ids   

def test_input(text, model,sess,word2id,id2tag,batch_size):
    
    max_len = len(text)
    get_text = text
    #text = [text]
    #print ("text:",len(text))
    text = re.split(u'[。！？]', text)
    #print ("text:",len(text))
    text_id=[]
    for sen in text:
        word_id=[]
        sen = sen + "。"  #给每个分句的句子加一个句号（分句会删掉）
        for word in sen:
            if word in word2id:
                word_id.append(word2id[word])
            else:
                word_id.append(word2id["unknow"])
        text_id.append(padding2(word_id))
    zero_padding=[]
    max_len2 = 60
    
    #补齐操作
    if len(text) > batch_size:
        print ("split text:",text)
        print ("orin text:",get_text)
        a=1/0
    zero_padding.extend([0]*max_len2)
    text_id.extend([zero_padding]*(batch_size-len(text_id))) 
    #print ("text_id:",np.asarray(text_id).shape)
    
    feed_dict = {model.input_data:text_id}
    pre = sess.run([model.viterbi_sequence], feed_dict)
    
    #测试用语句
    """
    for one_line in pre[0]:
        print (one_line)
    entity = get_entity(text,pre[0],id2tag)
    print ('result:')
    for i in entity:
        print (i)
    """
    
    result = [i/10 for i in pre[0]]
    
    #取出所有的零填充
    def delete_0(number_list):
        label =  0
        for i in range(len(number_list)):
            if number_list[i] == 0:
                label = i
                break
        number_list = number_list[:label]
        return (number_list)

    result = [delete_0(one_list) for one_list in result]
    
    #合并所有序列并进行截断
    result = np.concatenate(result,axis=0)
    result = padding2(list(result), max_len)
    
    return result

def test_input2(text, model):
    max_len = len(text)
    #crf_model = load_model("named_entity/ckpts/crf.pkl")
    
    test_word_lists = re.split(r"[。，]", text)
    pred_tag_lists = model.test(test_word_lists)

    def list_divide(a):
        return (len(a)/10)
    pred_tag_lists = [list(map(list_divide, pred_tag_lists[i])) for i in range(len(pred_tag_lists))]
    
    pred_tag_lists = np.concatenate(pred_tag_lists, axis = 0)
    pred_tag_lists = padding2(list(pred_tag_lists), max_len)
    
    return (pred_tag_lists)
    
def get_named_entity(token_list,config, one_named_entity):
    #替换掉其中的[unk]
    for i in range(len(token_list)):
        if token_list[i] == '[UNK]':
            token_list[i] = 'U'
    #print ("token_list长度为:",token_list)
    token_list = "".join(token_list)
    
    #进行命名实体识别，返回字典类型  词：词类型
    #char_class_list=test_input(token_list,config['ChineseNERModel'], config['sess2'], config['word2id'], config['id2tag'], config['batch_size2'])
    char_class_list=test_input2(token_list,config['crf_model'])
    #print ("char_class_list:",char_class_list)
    
    #连接到之前的字符级命名实体识别标注结果中
    one_named_entity = np.concatenate((one_named_entity,char_class_list),axis=0)
    return (list(one_named_entity))    
                 
def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000

    features = []
    unk_tokens = {}
    
    """
    #加载命名实体识别模型
    with open('ChineseNER/data/renmindata.pkl', 'rb') as inp:
    	word2id = pickle.load(inp)
    	id2word = pickle.load(inp)
    	tag2id = pickle.load(inp)
    	id2tag = pickle.load(inp)
    	x_train = pickle.load(inp)
    	y_train = pickle.load(inp)
    	x_test = pickle.load(inp)
    	y_test = pickle.load(inp)
    	x_valid = pickle.load(inp)
    	y_valid = pickle.load(inp)
    print ("train len:",len(x_train))
    print ("test len:",len(x_test))
    print ("word2id len", len(word2id))
    print ('Creating the data generator ...')
    data_train = BatchGenerator(x_train, y_train, shuffle=True)
    data_valid = BatchGenerator(x_valid, y_valid, shuffle=False)
    data_test = BatchGenerator(x_test, y_test, shuffle=False)
    print ('Finished creating the data generator.')
    
    
    epochs = 31
    batch_size = 32
    
    config2 = {}
    config2["lr"] = 0.001
    config2["embedding_dim"] = 100
    config2["sen_len"] = len(x_train[0])
    config2["batch_size"] = batch_size
    config2["embedding_size"] = len(word2id)+1
    config2["tag_size"] = len(tag2id)
    config2["pretrained"]=False
    
    #加载预训练的词向量
    embedding_pre = []
    print ("use pretrained embedding")
    config2["pretrained"]=True
    word2vec = {}
    with codecs.open('ChineseNER/model/vec.txt','r','utf-8') as input_data:   
        for line in input_data.readlines():
            word2vec[line.split()[0]] = map(eval,line.split()[1:])
    
    unknow_pre = []
    unknow_pre.extend([1]*100)
    embedding_pre.append(unknow_pre) #wordvec id 0
    for word in word2id:
        #if word2vec.has_key(word):
        if word in word2vec.keys():
            embedding_pre.append(word2vec[word])
        else:
            embedding_pre.append(unknow_pre)
    
    embedding_pre = np.asarray(embedding_pre)
    
    #开始测试
    print ("begin to test...")
    with tf.device("/CPU:0"):
        tf.reset_default_graph()
        from tensorflow.python.client import device_lib
        print(device_lib.list_local_devices())
        config3 = tf.ConfigProto()
        config3.gpu_options.per_process_gpu_memory_fraction = 0.01
        
        ChineseNERModel = Model(config2,embedding_pre,dropout_keep=1)
        sess2 = tf.Session(config=config3)
    
        sess2.run(tf.global_variables_initializer())
        saver = tf.train.Saver()  
        ckpt = tf.train.get_checkpoint_state('ChineseNER/model/model')
        if ckpt is None:
            print ('Model not found, please train your model first')
        else:    
            path2 = ckpt.model_checkpoint_path
            print ('loading pre-trained model from %s.....' % path2)
            saver.restore(sess2, path2)
        
        config2['ChineseNERModel'] = ChineseNERModel
        config2['sess2'] = sess2
        config2['word2id'] = word2id
        config2['id2tag'] = id2tag
        config2['batch_size2'] = batch_size
        
        """
    config2={}
    crf_model = load_model("named_entity/ckpts/crf.pkl")
    config2['crf_model'] = crf_model
    convert_token_list = {
        '“': '"', "”": '"', '…': '...', '﹤': '<', '﹥': '>', '‘': "'", '’': "'",
        '﹪': '%', 'Ⅹ': 'x', '―': '-', '—': '-', '﹟': '#', '㈠': '一'
    }
    for (example_index, example) in enumerate(examples):
        #对问题进行tokenize
        if example_index % 1000 == 0 and example_index != 0:
            logger.info("example_index: %d" % (example_index))
        query_tokens = tokenizer.tokenize(example.question_text)
        
        #对问题进行最大长度截断
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        
        #对每篇文章进行处理
        for (i, token) in enumerate(example.doc_tokens):
            # if token in convert_token_list:
            #     token = convert_token_list[token]
            orig_to_tok_index.append(len(all_doc_tokens))
            #对篇章进行tokenize
            sub_tokens = tokenizer.tokenize(token)
            if "[UNK]" in sub_tokens:
                if token in unk_tokens:
                    unk_tokens[token] += 1
                else:
                    unk_tokens[token] = 1

            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)
        
        tok_start_position = None
        tok_end_position = None
        
        #如果为不可回答类型的问题，设置起始位置为-1，-1
        if is_training and example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1
        
        #如果为回答类型的问题，读取答案的起止位置
        if is_training and not example.is_impossible:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            
            #根据分词后的结果找到新的答案起止位置
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example.orig_answer_text)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        #计算篇章的最大长度
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        
        #当篇章长度过长时，使用“滑窗”的方法切分成为多个子篇章
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)
            
        #根据子篇章的起始位置进行处理
        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            
            one_word_feature = []  #词性向量
            one_named_entity = [] #命名实体识别向量
            
            
            #问题的起始位置前加[CLS],问题token都被标记为0
            tokens.append("[CLS]")
            one_word_feature.append(0)
            one_named_entity.append(0)
            segment_ids.append(0)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
            
            one_word_feature = get_word_class(query_tokens, one_word_feature)#对问题进行词性标注
            one_named_entity = get_named_entity(query_tokens, config2, one_named_entity)
            #问题的结束位置后加[SEP]
            tokens.append("[SEP]")
            one_word_feature.append(0)
            one_named_entity.append(0)
            segment_ids.append(0)
            
            #篇章中的每个token找到一个最好的上下文
            one_doc_tokens = []
            get_split_index = 0
            
            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]
                get_split_index = split_token_index #保存最后一个token对应篇章序号

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)

                token_is_max_context[len(tokens)] = is_max_context
                
                tokens.append(all_doc_tokens[split_token_index])
                one_doc_tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
            
            one_word_feature = get_word_class(one_doc_tokens, one_word_feature)#对问题进行词性标注
            one_named_entity = get_named_entity(one_doc_tokens, config2, one_named_entity)
            
            tokens.append("[SEP]")
            one_word_feature.append(0)
            one_named_entity.append(0)
            segment_ids.append(1)
            
            """
            print ("原始长度为：",len(query_tokens)+len(one_doc_tokens)+3)
            print ("词性向量长度为:",len(one_word_feature))
            print ("命名实体识别向量的长度为：",len(one_named_entity))
            """

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            
            #输入的掩码为input_id的全1序列
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            #如果长度不够则进行0填充
            if len(input_ids) != len(one_word_feature):
                one_word_feature = one_word_feature[:len(input_ids)]
                one_named_entity = one_named_entity[:len(input_ids)]
                
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                one_word_feature.append(0)
                one_named_entity.append(0)
            
            
            #进行缺少的tokens的填充 
            while len(tokens) < max_seq_length:
                tokens.append("0")
                # 增加token所在篇章序号的标记，全部对应到最后一个篇章
                max_key = list(token_to_orig_map.keys())[-1] + 1
                token_to_orig_map[max_key] = tok_to_orig_index[get_split_index]
                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       get_split_index)

                token_is_max_context[len(tokens)] = is_max_context
            
            max_key = list(token_to_orig_map.keys())[-1] + 1
            if max_key < max_seq_length:
                token_to_orig_map[max_key] = tok_to_orig_index[get_split_index]
                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       get_split_index)

                token_is_max_context[max_key] = is_max_context
            
            #设置断言，过滤长度不为max_seq_length的情况
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            start_position = None
            end_position = None
            
            #如果答案类型为片段提取类型，则答案的起止为0-512
            if is_training and not example.is_impossible:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                #如果某一子段不包含答案，则抛弃之（设置为拒答类型）
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                
                #如果答案不在篇章子段中，设置为拒答类型
                if not (tok_start_position >= doc_start and
                        tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    start_position = max_seq_length
                    end_position = max_seq_length
                else:
                #如果答案在篇章子段中，按照篇章的长度更新起止位置
                    doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset

            unk_mask, yes_mask, no_mask = [0], [0], [0]
            #如果答案类型为拒答类型，则答案的起止为512，同时添加unk掩码
            if is_training and example.is_impossible:
                # start_position = 0
                # end_position = 0
                start_position = max_seq_length
                end_position = max_seq_length
                unk_mask = [1]
            #如果答案为Yes，则答案的起止为513，同时添加yes掩码
            elif is_training and example.is_yes:
                start_position = max_seq_length+1
                end_position = max_seq_length+1
                yes_mask = [1]
            #如果答案为no，则答案的起止为514，同时添加
            elif is_training and example.is_no:
                start_position = max_seq_length+2
                end_position = max_seq_length+2
                no_mask = [1]

            if example_index < 0:
                logger.info("*** Example ***")
                logger.info("unique_id: %s" % (unique_id))
                logger.info("example_index: %s" % (example_index))
                logger.info("doc_span_index: %s" % (doc_span_index))
                logger.info("tokens: %s" % " ".join(tokens))
                logger.info("token_to_orig_map: %s" % " ".join([
                    "%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()]))
                logger.info("token_is_max_context: %s" % " ".join([
                    "%d:%s" % (x, y) for (x, y) in token_is_max_context.items()
                ]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info(
                    "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                if is_training and example.is_impossible:
                    logger.info("impossible example")
                if is_training and not example.is_impossible:
                    answer_text = "".join(tokens[start_position:(end_position + 1)])
                    logger.info("start_position: %d" % (start_position))
                    logger.info("end_position: %d" % (end_position))
                    logger.info(
                        "answer: %s" % (answer_text))
            #将特征打包
            features.append(
                InputFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=example.is_impossible,
                    unk_mask=unk_mask,
                    yes_mask=yes_mask,
                    no_mask=no_mask,
                    word_feature_list = one_word_feature,
                    named_entity_list = one_named_entity))
            unique_id += 1
    if is_training:
        with open("unk_tokens_clean", "w", encoding="utf-8") as fh:
            for key, value in unk_tokens.items():
                fh.write(key+" " + str(value)+"\n")
    #关闭会话
    #sess2.close()
    
    return features


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    
    #对答案进行tokenize
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))
    
    #逐个比对，得到新的答案起始位置
    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)

#找出最好的一个上下文：最好的上下文为左右两边的上下文最长。
def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        #如果篇章的token位置非法：超过篇章最大长度，小于篇章的起始位置
        if position < doc_span.start:
            continue
        if position > end:
            continue
        #左半段的上下文
        num_left_context = position - doc_span.start
        num_right_context = end - position
        #计算分数
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def get_attention_text(all_features, unique_id_to_result):
    counter = 0 
    result_str = ""
    for (feature_index, feature) in enumerate(all_features):
        # 对于某个片段，计算得分
        result = unique_id_to_result[feature.unique_id]
        text = feature.tokens
        text = " ".join(text)
        
        attention_value = result.attention_value
        attention_value = np.asarray(attention_value)
        attention_value = np.reshape(attention_value, (attention_value.shape[0],))
        attention_value = " ".join(str(attention_value))
        counter = counter + 1
        
        #记录数值
        result_str = result_str + text + "\n"
        result_str = result_str + attention_value + "\n"
        if counter == 20:
            break
    result_file = open("output/attention_value.txt", "a", encoding = "utf8")
    result_file.write(result_str)
    result_file.close()

def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file, verbose_logging,
                      version_2_with_negative, null_score_diff_threshold):
    
    """Write final predictions to the json file and log-odds of null if needed."""
    logger.info("Writing predictions to: %s" % (output_prediction_file))
    logger.info("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    # 将每个样例的不同片段加入到对应的list中， 一个example_index对应若干个unique_id
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    # 每个unique_id的答案
    for result in all_results:
        unique_id_to_result[result.unique_id] = result
    
    #将文本与对应注意力权值，存储到篇章中
    get_attention_text(all_features, unique_id_to_result)
    
    
    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        # 获得该样本所有片段
        features = example_index_to_features[example_index]

        # 该样本的答案
        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min null score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score

        score_yes = 1000000
        min_yes_feature_index = 0  # the paragraph slice with min null score
        yes_start_logit = 0  # the start logit at the slice with min null score
        yes_end_logit = 0  # the end logit at the slice with min null score

        score_no = 1000000
        min_no_feature_index = 0  # the paragraph slice with min null score
        no_start_logit = 0  # the start logit at the slice with min null score
        no_end_logit = 0  # the end logit at the slice with min null score
        
        
        false_list = [0] * 7
        #print ("len(features):", len(features))
        for (feature_index, feature) in enumerate(features):
            # 对于某个片段，计算得分
            result = unique_id_to_result[feature.unique_id]
            #根据概率获得最好的起始位置
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            #根据概率获得最好的终止位置
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            
            #如果有拒答类型的问题
            if version_2_with_negative:
                
                #找到一个最小的拒答概率
                feature_null_score = result.unk_logits[0]*2
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.unk_logits[0]
                    null_end_logit = result.unk_logits[0]
                
                #找到一个最小的Yes回答概率
                feature_yes_score = result.yes_logits[0] + result.yes_logits[0]
                if feature_yes_score < score_yes:
                    score_yes = feature_yes_score
                    min_yes_feature_index = feature_index
                    yes_start_logit = result.yes_logits[0]
                    yes_end_logit = result.yes_logits[0]
                
                #找到一个最小的No回答概率
                feature_no_score = result.no_logits[0] + result.no_logits[0]
                if feature_no_score < score_no:
                    score_no = feature_no_score
                    min_no_feature_index = feature_index
                    no_start_logit = result.no_logits[0]
                    no_end_logit = result.no_logits[0]

            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    
                    #对start_index和end_index进行七种错误类型的过滤
                    if start_index >= len(feature.tokens):
                        #print (1)
                        false_list[0] = false_list[0] + 1
                        continue
                    if end_index >= len(feature.tokens):
                        false_list[1] = false_list[1] + 1
                        #print (2)
                        continue
                    if start_index not in feature.token_to_orig_map:
                        false_list[2] = false_list[2] + 1
                        #print (3)
                        continue
                    if end_index not in feature.token_to_orig_map:
                        false_list[3] = false_list[3] + 1
                        #print (4)
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        false_list[4] = false_list[4] + 1
                        #print (5)
                        continue
                    if end_index < start_index:
                        false_list[5] = false_list[5] + 1
                        #print (6)
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        false_list[6] = false_list[6] + 1
                        #print (7, ", length:",length,",max_answer_length:",max_answer_length,",start_index:",start_index,",end_index:",end_index)
                        continue
                    
                    #没有错误，则加入预测结果中
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))
        
        #如果为包含拒答类和Yes or no类型问题
        if True:#version_2_with_negative
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=512,
                    end_index=512,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit))
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_yes_feature_index,
                    start_index=513,
                    end_index=513,
                    start_logit=yes_start_logit,
                    end_logit=yes_end_logit))
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_no_feature_index,
                    start_index=514,
                    end_index=514,
                    start_logit=no_start_logit,
                    end_logit=no_end_logit))
        # 排序
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"])
        
        
        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                #print (1,":",len(nbest), ", n_best_size:",n_best_size)
                break
            feature = features[pred.feature_index]
            #如果预测是一个非空片段预测
            if pred.start_index < 512:  # this is a non-null prediction
                #从原始文本中获取片段
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = "".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = "".join(tok_text.split())
                orig_text = "".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
            #如果预测的起始位置为512，则预测的输出为空
            elif pred.start_index == 512:
                final_text = ""
            #如果预测的起始位置为513，则预测的输出为Yes
            elif pred.start_index == 513:
                final_text = "YES"
            #如果预测的起始位置为其他，则输出的预测为No
            else:
                final_text = "NO"

            if final_text in seen_predictions:
                #print (2)
                continue
            seen_predictions[final_text] = True
            
            #给出最终预测
            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit))
        # # # if we didn't include the empty option in the n-best, include it
        # if version_2_with_negative:
        #     if "" not in seen_predictions:
        #         nbest.append(
        #             _NbestPrediction(
        #                 text="",
        #                 start_logit=null_start_logit,
        #                 end_logit=null_end_logit))
        #
        #     # In very rare edge cases we could only have single null prediction.
        #     # So we just create a nonce prediction in this case to avoid failure.
        #     if len(nbest) == 1:
        #         nbest.insert(0,
        #                      _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))
        #
        # # In very rare edge cases we could have no valid predictions. So we
        # # just create a nonce prediction in this case to avoid failure.
        # if not nbest:
        #     nbest.append(
        #         _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))
        #print ("len(nbest):",len(nbest))
        assert len(nbest) >= 1, print ("len(prelim_predictions):", len(prelim_predictions), "false_list:", false_list)

        total_scores = []
        # best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            # if not best_non_null_entry:
            #     if entry.text:
            #         best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        all_predictions[example.qas_id] = nbest_json[0]["text"]

        # if not version_2_with_negative:
        #     all_predictions[example.qas_id] = nbest_json[0]["text"]
        # else:
        #     # predict "" iff the null score - the score of best non-null > threshold
        #     score_diff = score_null - best_non_null_entry.start_logit - (
        #         best_non_null_entry.end_logit)
        #     scores_diff_json[example.qas_id] = score_diff
        #     if score_diff > null_score_diff_threshold:
        #         all_predictions[example.qas_id] = ""
        #     else:
        #         all_predictions[example.qas_id] = best_non_null_entry.text
        all_nbest_json[example.qas_id] = nbest_json

    with open(output_prediction_file, "w" ,encoding="utf-8") as writer:
        writer.write(json.dumps(all_predictions, indent=4, ensure_ascii=False) + "\n")

    with open(output_nbest_file, "w",encoding="utf-8") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4, ensure_ascii=False) + "\n")

    if version_2_with_negative:
        with open(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    return all_predictions


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heuristic between
    # `pred_text` and `orig_text` to get a character-to-character alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                        orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


# def write_predictions_test(all_examples, all_features, all_results, n_best_size,
#                       max_answer_length, do_lower_case, output_prediction_file,
#                       verbose_logging, version_2_with_negative, null_score_diff_threshold):
#     """Write final predictions to the json file and log-odds of null if needed."""
#     logger.info("Writing predictions to: %s" % (output_prediction_file))
#
#     example_index_to_features = collections.defaultdict(list)
#     # 将每个样例的不同片段加入到对应的list中， 一个example_index对应若干个unique_id
#     for feature in all_features:
#         example_index_to_features[feature.example_index].append(feature)
#
#     unique_id_to_result = {}
#     # 每个unique_id的答案
#     for result in all_results:
#         unique_id_to_result[result.unique_id] = result
#
#     _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
#         "PrelimPrediction",
#         ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])
#
#     all_predictions = collections.OrderedDict()
#     all_nbest_json = collections.OrderedDict()
#     scores_diff_json = collections.OrderedDict()
#
#     for (example_index, example) in enumerate(all_examples):
#         # 获得该样本所有片段
#         features = example_index_to_features[example_index]
#
#         # 该样本的答案
#         prelim_predictions = []
#         # keep track of the minimum score of null start+end of position 0
#         score_null = 1000000  # large and positive
#         min_null_feature_index = 0  # the paragraph slice with min null score
#         null_start_logit = 0  # the start logit at the slice with min null score
#         null_end_logit = 0  # the end logit at the slice with min null score
#
#         score_yes = 1000000
#         min_yes_feature_index = 0  # the paragraph slice with min null score
#         yes_start_logit = 0  # the start logit at the slice with min null score
#         yes_end_logit = 0  # the end logit at the slice with min null score
#
#         score_no = 1000000
#         min_no_feature_index = 0  # the paragraph slice with min null score
#         no_start_logit = 0  # the start logit at the slice with min null score
#         no_end_logit = 0  # the end logit at the slice with min null score
#
#         for (feature_index, feature) in enumerate(features):
#             # 对于某个片段，计算得分
#             result = unique_id_to_result[feature.unique_id]
#             start_indexes = _get_best_indexes(result.start_logits, n_best_size)
#             end_indexes = _get_best_indexes(result.end_logits, n_best_size)
#             # if we could have irrelevant answers, get the min score of irrelevant
#             if version_2_with_negative:
#                 # feature_null_score = result.unk_logits[0]*2
#                 feature_null_score = result.start_logits[0]+result.end_logits[0]
#                 if feature_null_score < score_null:
#                     score_null = feature_null_score
#                     min_null_feature_index = feature_index
#                     null_start_logit = result.unk_logits[0]
#                     null_end_logit = result.unk_logits[0]
#
#                 feature_yes_score = result.yes_logits[0]*2
#                 if feature_yes_score < score_yes:
#                     score_yes = feature_yes_score
#                     min_yes_feature_index = feature_index
#                     yes_start_logit = result.yes_logits[0]
#                     yes_end_logit = result.yes_logits[0]
#
#                 feature_no_score = result.no_logits[0]*2
#                 if feature_no_score < score_no:
#                     score_no = feature_no_score
#                     min_no_feature_index = feature_index
#                     no_start_logit = result.no_logits[0]
#                     no_end_logit = result.no_logits[0]
#
#             for start_index in start_indexes:
#                 for end_index in end_indexes:
#                     # We could hypothetically create invalid predictions, e.g., predict
#                     # that the start of the span is in the question. We throw out all
#                     # invalid predictions.
#                     if start_index >= len(feature.tokens):
#                         continue
#                     if end_index >= len(feature.tokens):
#                         continue
#                     if start_index not in feature.token_to_orig_map:
#                         continue
#                     if end_index not in feature.token_to_orig_map:
#                         continue
#                     if not feature.token_is_max_context.get(start_index, False):
#                         continue
#                     if end_index < start_index:
#                         continue
#                     length = end_index - start_index + 1
#                     if length > max_answer_length:
#                         continue
#                     prelim_predictions.append(
#                         _PrelimPrediction(
#                             feature_index=feature_index,
#                             start_index=start_index,
#                             end_index=end_index,
#                             start_logit=result.start_logits[start_index],
#                             end_logit=result.end_logits[end_index]))
#         if version_2_with_negative:
#             prelim_predictions.append(
#                 _PrelimPrediction(
#                     feature_index=min_null_feature_index,
#                     start_index=512,
#                     end_index=512,
#                     start_logit=null_start_logit,
#                     end_logit=null_end_logit))
#             prelim_predictions.append(
#                 _PrelimPrediction(
#                     feature_index=min_yes_feature_index,
#                     start_index=513,
#                     end_index=513,
#                     start_logit=yes_start_logit,
#                     end_logit=yes_end_logit))
#             prelim_predictions.append(
#                 _PrelimPrediction(
#                     feature_index=min_no_feature_index,
#                     start_index=514,
#                     end_index=514,
#                     start_logit=no_start_logit,
#                     end_logit=no_end_logit))
#         # 排序
#         prelim_predictions = sorted(
#             prelim_predictions,
#             key=lambda x: (x.start_logit + x.end_logit),
#             reverse=True)
#
#         _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
#             "NbestPrediction", ["text", "start_logit", "end_logit"])
#
#         seen_predictions = {}
#         nbest = []
#         for pred in prelim_predictions:
#             if len(nbest) >= n_best_size:
#                 break
#             feature = features[pred.feature_index]
#             if pred.start_index < 512:  # this is a non-null prediction
#                 tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
#                 orig_doc_start = feature.token_to_orig_map[pred.start_index]
#                 orig_doc_end = feature.token_to_orig_map[pred.end_index]
#                 orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
#                 tok_text = "".join(tok_tokens)
#
#                 # De-tokenize WordPieces that have been split off.
#                 tok_text = tok_text.replace(" ##", "")
#                 tok_text = tok_text.replace("##", "")
#
#                 # Clean whitespace
#                 tok_text = tok_text.strip()
#                 tok_text = "".join(tok_text.split())
#                 orig_text = "".join(orig_tokens)
#
#                 final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
#             elif pred.start_index == 512:
#                 final_text = ""
#             elif pred.start_index == 513:
#                 final_text = "YES"
#             else:
#                 final_text = "NO"
#
#             if final_text in seen_predictions:
#                 continue
#             seen_predictions[final_text] = True
#
#             nbest.append(
#                 _NbestPrediction(
#                     text=final_text,
#                     start_logit=pred.start_logit,
#                     end_logit=pred.end_logit))
#
#         # # # if we didn't include the empty option in the n-best, include it
#         # if version_2_with_negative:
#         #     if "" not in seen_predictions:
#         #         nbest.append(
#         #             _NbestPrediction(
#         #                 text="",
#         #                 start_logit=null_start_logit,
#         #                 end_logit=null_end_logit))
#         #
#         #     # In very rare edge cases we could only have single null prediction.
#         #     # So we just create a nonce prediction in this case to avoid failure.
#         #     if len(nbest) == 1:
#         #         nbest.insert(0,
#         #                      _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))
#         #
#         # # In very rare edge cases we could have no valid predictions. So we
#         # # just create a nonce prediction in this case to avoid failure.
#         # if not nbest:
#         #     nbest.append(
#         #         _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))
#
#         assert len(nbest) >= 1
#
#         total_scores = []
#         # best_non_null_entry = None
#         for entry in nbest:
#             total_scores.append(entry.start_logit + entry.end_logit)
#             # if not best_non_null_entry:
#             #     if entry.text:
#             #         best_non_null_entry = entry
#
#         probs = _compute_softmax(total_scores)
#
#         nbest_json = []
#         for (i, entry) in enumerate(nbest):
#             output = collections.OrderedDict()
#             output["text"] = entry.text
#             output["probability"] = probs[i]
#             output["start_logit"] = entry.start_logit
#             output["end_logit"] = entry.end_logit
#             nbest_json.append(output)
#
#         assert len(nbest_json) >= 1
#
#         # score_diff = score_null - best_non_null_entry.start_logit - (
#         #     best_non_null_entry.end_logit)
#         # scores_diff_json[example.qas_id] = score_diff
#         # if score_diff > null_score_diff_threshold:
#         #     all_predictions[example.qas_id] = ""
#         # else:
#         #     all_predictions[example.qas_id] = best_non_null_entry.text
#         all_predictions[example.qas_id] = nbest_json[0]["text"]
#         all_nbest_json[example.qas_id] = nbest_json
#
#     # preds = []
#     # for key, value in all_predictions.items():
#     #     preds.append({'id': key, 'answer': value})
#     #
#     # with open(output_prediction_file, 'w') as fh:
#     #     json.dump(preds, fh, ensure_ascii=False)
#
#     yes_id = []
#     the_insured = {}
#     null_id = []
#     doc_len = {}
#     who_id = []
#     for example in all_examples:
#         if example.question_text.find('是否') >= 0:
#             yes_id.append(example.qas_id)
#
#         if example.question_text.find('吗？') >= 0:
#             null_id.append(example.qas_id)
#
#         if find_correct_the_insured(example.question_text, "".join(example.doc_tokens)) != '':
#             the_insured[example.qas_id] =\
#                 find_correct_the_insured(example.question_text, "".join(example.doc_tokens))
#         doc_len[example.qas_id] = len(example.doc_tokens)
#
#         # if example.question_text.find('谁') >= 0 or example.question_text.find('何人') >= 0:
#         #     who_id.append(example.qas_id)
#
#     preds = []
#     for key, value in all_predictions.items():
#         if key in yes_id:
#             if value in ['YES', 'NO', '']:
#                 preds.append({'id': key, 'answer': value})
#             elif value.find('未') >= 0 or value.find('没有') >= 0 or value.find('不是')>=0 \
#                 or value.find('无责任') >= 0 or value.find('不归还') >=0 \
#                 or value.find('不予认可') >= 0 or value.find('拒不') >=0 \
#                 or value.find('无效') >= 0 or value.find('不是') >=0\
#                 or value.find('未尽') >= 0 or value.find('未经') >=0\
#                 or value.find('无异议') >= 0 or value.find('未办理')>=0\
#                 or value.find('均未') >= 0:
#                 preds.append({'id': key, 'answer': "NO"})
#             else:
#                 preds.append({'id': key, 'answer': "YES"})
#         elif key in the_insured:
#             if value != '' and the_insured[key].find(value) >= 0:
#                 preds.append({'id': key, 'answer': value})
#             else:
#                 preds.append({'id': key, 'answer': the_insured[key]})
#                 # logger.info('handout:'+the_insured[key])
#                 # logger.info('pred:'+value)
#                 # logger.info(preds[-1])
#                 # logger.info("--"*10)
#         # elif key in null_id:
#         #     # TO 是否设为全都是null更好
#         #     if value == '':
#         #         # print('我输出了这个值')
#         #         preds.append({'id': key, 'answer': ''})
#         #     else:
#         #         preds.append({'id': key, 'answer': 'YES'})
#
#         else:
#             # if key in who_id:
#             #     print(key)
#             #     if len(re.findall('某', value)) == 1:
#             #         print("*"*20)
#             #         print(value)
#             #         value = value[value.find('某')-1:value.find('某')+2]
#             #         print(value)
#             #     preds.append({'id': key, 'answer': value})
#             #
#             # else:
#             # if value in ['YES', 'NO']:
#             #     for best in all_nbest_json[key]:
#             #         if best['text'] not in ['YES', 'NO']:
#             #             value = best['text']
#             #             break
#             preds.append({'id': key, 'answer': value})
#         # preds.append({'id': key, 'answer': value})
#
#     # with open(output_prediction_file+'.orig', "w") as writer:
#     #     writer.write(json.dumps(all_predictions, indent=4, ensure_ascii=False) + "\n")
#     #
#     # with open(output_prediction_file+'.nbest', "w") as writer:
#     #     writer.write(json.dumps(all_nbest_json, indent=4, ensure_ascii=False) + "\n")
#     #
#     # if version_2_with_negative:
#     #     with open(output_prediction_file+'.null', "w") as writer:
#     #         writer.write(json.dumps(scores_diff_json, indent=4) + "\n")
#
#     with open(output_prediction_file, 'w') as fh:
#         json.dump(preds, fh, ensure_ascii=False)



def write_predictions_test(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      verbose_logging, version_2_with_negative, null_score_diff_threshold):
    """Write final predictions to the json file and log-odds of null if needed."""
    logger.info("Writing predictions to: %s" % (output_prediction_file))

    example_index_to_features = collections.defaultdict(list)
    # 将每个样例的不同片段加入到对应的list中， 一个example_index对应若干个unique_id
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    # 每个unique_id的答案
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        # 获得该样本所有片段
        features = example_index_to_features[example_index]

        # 该样本的答案
        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min null score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score

        score_yes = 1000000
        min_yes_feature_index = 0  # the paragraph slice with min null score
        yes_start_logit = 0  # the start logit at the slice with min null score
        yes_end_logit = 0  # the end logit at the slice with min null score

        score_no = 1000000
        min_no_feature_index = 0  # the paragraph slice with min null score
        no_start_logit = 0  # the start logit at the slice with min null score
        no_end_logit = 0  # the end logit at the slice with min null score

        for (feature_index, feature) in enumerate(features):
            # 对于某个片段，计算得分
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            if version_2_with_negative:
                feature_null_score = result.unk_logits[0]*2
                # feature_null_score = result.start_logits[0]+result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.unk_logits[0]
                    null_end_logit = result.unk_logits[0]

                feature_yes_score = result.yes_logits[0] + result.yes_logits[0]
                if feature_yes_score < score_yes:
                    score_yes = feature_yes_score
                    min_yes_feature_index = feature_index
                    yes_start_logit = result.yes_logits[0]
                    yes_end_logit = result.yes_logits[0]

                feature_no_score = result.no_logits[0] + result.no_logits[0]
                if feature_no_score < score_no:
                    score_no = feature_no_score
                    min_no_feature_index = feature_index
                    no_start_logit = result.no_logits[0]
                    no_end_logit = result.no_logits[0]

            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))
        if version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=512,
                    end_index=512,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit))
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_yes_feature_index,
                    start_index=513,
                    end_index=513,
                    start_logit=yes_start_logit,
                    end_logit=yes_end_logit))
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_no_feature_index,
                    start_index=514,
                    end_index=514,
                    start_logit=no_start_logit,
                    end_logit=no_end_logit))
        # 排序
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index < 512:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = "".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = "".join(tok_text.split())
                orig_text = "".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
            elif pred.start_index == 512:
                final_text = ""
            elif pred.start_index == 513:
                final_text = "YES"
            else:
                final_text = "NO"

            if final_text in seen_predictions:
                continue
            seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit))

        assert len(nbest) >= 1

        total_scores = []
        # add
        # best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            # if not best_non_null_entry:
            #     if entry.text:
            #         best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        # predict "" iff the null score - the score of best non-null > threshold
        # unk的score和预测最好的span的差值
        # score_diff = score_null - best_non_null_entry.start_logit - (
        #     best_non_null_entry.end_logit)
        # scores_diff_json[example.qas_id] = score_diff
        # if score_diff > null_score_diff_threshold:
        #     all_predictions[example.qas_id] = ""
        # else:
        #     all_predictions[example.qas_id] = best_non_null_entry.text

        all_predictions[example.qas_id] = nbest_json[0]["text"]

    # if version_2_with_negative:
    #         with open(output_null_log_odds_file, "w") as writer:
    #             writer.write(json.dumps(scores_diff_json, indent=4) + "\n")
    #
    # with open(output_nbest_file, "w") as writer:
    #     writer.write(json.dumps(all_nbest_json, indent=4) + "\n")
    # # preds = []
    # # for key, value in all_predictions.items():
    # #     preds.append({'id': key, 'answer': value})
    # #
    # with open(output_prediction_file+'new', 'w') as fh:
    #     json.dump(all_predictions, fh, ensure_ascii=False)

    yes_id = []
    the_insured = {}
    null_id = []
    doc_len = {}
    unk_id = []
    long_answer = {}
    time_id = {}
    occur_time = {}
    repair_r = {}
    insurant_person_id = {}
    insurant_company_id = {}
    for example in all_examples:
        if example.question_text.find('是否') >= 0:
            yes_id.append(example.qas_id)

        if example.question_text.find('吗？') >= 0:
            null_id.append(example.qas_id)

        if find_correct_the_insured(example.question_text, "".join(example.doc_tokens)) != '':
            the_insured[example.qas_id] =\
                find_correct_the_insured(example.question_text, "".join(example.doc_tokens))
        doc_len[example.qas_id] = len(example.doc_tokens)

        # if example.question_text.find('谁') >= 0 or example.question_text.find('何人') >= 0:
        #     who_id.append(example.qas_id)
        if example.question_text in ['被告人判刑情况？',
                                     '被告人有无存在其他犯罪记录？', '哪个法院受理了此案？',
                                     '双方有没有达成一致的调解意见？', '被告人最终判刑情况？',
                                     '被告人是如何归案的？', '本案诉讼费是多少钱？',
                                     '双方有没有达成一致的协调意见？', '本案事实有无证据证实？',
                                     '本案所述事故发生原因是什么？', '事故发生原因是什么？']:
            unk_id.append(example.qas_id)
        if example.question_text.find("案件发生经过是怎样的") >= 0:
            long_answer[example.qas_id] = find_long_answer(all_predictions[example.qas_id], "".join(example.doc_tokens),
                                                           example.question_text)
            print('long_answer')
            print('r', long_answer[example.qas_id])
            print('pred', all_predictions[example.qas_id])

        if example.question_text.find('有效时间是多久') >= 0:
            time_id[example.qas_id] = find_time_span(example.question_text, all_predictions[example.qas_id])

            print('time_id')
            print('r', time_id[example.qas_id])
            print('pred', all_predictions[example.qas_id])

        if example.question_text.find('事故发生时间是什么时候？') >= 0:
            occur_time[example.qas_id] = repair_time(example.question_text, all_predictions[example.qas_id])
            print('occur_time')
            print('r', occur_time[example.qas_id])
            print('pred', all_predictions[example.qas_id])

        if example.question_text.find('事故结果如何') >= 0:
            repair_r[example.qas_id] = repair_result("".join(example.doc_tokens),
                                                     example.question_text, all_predictions[example.qas_id])

            print('occur_time')
            print('r', repair_r[example.qas_id])
            print('pred', all_predictions[example.qas_id])

        if example.question_text.find('投保的人是谁') >= 0 or example.question_text.find('投保人是谁') >= 0:
            per = get_insurant_person("".join(example.doc_tokens), example.question_text)
            if per:
                insurant_person_id[example.qas_id] = per
                print('ins_per')
                print('r', insurant_person_id[example.qas_id])
                print('pred', all_predictions[example.qas_id])

        if example.question_text.find('向什么公司投保') >= 0:
            cmp = get_insurant_company("".join(example.doc_tokens))
            if cmp:
                insurant_company_id[example.qas_id] = cmp
                print('ins_cmp')
                print('r', insurant_company_id[example.qas_id])
                print('pred', all_predictions[example.qas_id])

    preds = []
    for key, value in all_predictions.items():
        if key in insurant_company_id:
            preds.append({'id': key, 'answer': insurant_company_id[key]})
        elif key in insurant_person_id:
            preds.append({'id': key, 'answer': insurant_person_id[key]})
        elif key in long_answer:
            preds.append({'id': key, 'answer': long_answer[key]})
        elif key in time_id:
            preds.append({'id': key, 'answer': time_id[key]})
        elif key in occur_time:
            preds.append({'id': key, 'answer': occur_time[key]})
        elif key in repair_r:
            preds.append({'id': key, 'answer': repair_r[key]})
        elif key in unk_id:
            preds.append({'id': key, 'answer': ''})
        elif key in yes_id:
            if value in ['YES', 'NO', '']:
                preds.append({'id': key, 'answer': value})
            elif value.find('未') >= 0 or value.find('没有') >= 0 or value.find('不是') >= 0 \
                or value.find('无责任') >= 0 or value.find('不归还') >= 0 \
                or value.find('不予认可') >= 0 or value.find('拒不') >= 0 \
                or value.find('无效') >= 0 or value.find('不是') >= 0 \
                or value.find('未尽') >= 0 or value.find('未经') >= 0 \
                or value.find('无异议') >= 0 or value.find('未办理') >= 0\
                or value.find('均未') >= 0:
                preds.append({'id': key, 'answer': "NO"})
            else:
                preds.append({'id': key, 'answer': "YES"})
        elif key in the_insured:
            if value != '' and the_insured[key].find(value) >= 0:
                preds.append({'id': key, 'answer': value})
            else:
                preds.append({'id': key, 'answer': the_insured[key]})

        else:
            preds.append({'id': key, 'answer': value})

    with open(output_prediction_file, 'w') as fh:
        json.dump(preds, fh, ensure_ascii=False)


def find_correct_the_insured(question, passage_all):
    pred_answer = ''
    if question.find('被保险人是谁') >= 0 or (question.find('被保险人是') >= 0 and question.find('被保险人是否') < 0):
        # 还有一种情况，被保险人xxx，但是这种很难匹配因为文章可能出现多次，所以交给模型来预测
        if passage_all.find('被保险人是') >= 0:
            start_index = passage_all.find('被保险人是')
            for ch in passage_all[start_index + 5:]:
                if ch == '，' or ch == '；' or ch == '(' or ch == ',' or ch == ';':
                    break
                else:
                    pred_answer += ch
        elif passage_all.find('被保险人为') >= 0:
            start_index = passage_all.find('被保险人为')
            for ch in passage_all[start_index + 5:]:
                if ch == '，' or ch == '；' or ch == '(' or ch == ',' or ch == ';':
                    break
                else:
                    pred_answer += ch
        if pred_answer != '' and question.find("被保险人是" + pred_answer) > 0:
            pred_answer = 'YES'

    if question.find('投保人是谁') >= 0:
        start_index = passage_all.find('投保人为')
        for ch in passage_all[start_index + 4:]:
            if ch == '，' or ch == '；' or ch == '(' or ch == ',' or ch == ';':
                break
            else:
                pred_answer += ch

    # 如果 pred_answer ==''说明文章中找不到，以模型预测出的结果为准
    return pred_answer


def write_predictions_test_ensemble(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case,
                      verbose_logging, version_2_with_negative, null_score_diff_threshold):
    """Write final predictions to the json file and log-odds of null if needed."""

    example_index_to_features = collections.defaultdict(list)
    # 将每个样例的不同片段加入到对应的list中， 一个example_index对应若干个unique_id
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    # 每个unique_id的答案
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        # 获得该样本所有片段
        features = example_index_to_features[example_index]

        # 该样本的答案
        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min null score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score

        score_yes = 1000000
        min_yes_feature_index = 0  # the paragraph slice with min null score
        yes_start_logit = 0  # the start logit at the slice with min null score
        yes_end_logit = 0  # the end logit at the slice with min null score

        score_no = 1000000
        min_no_feature_index = 0  # the paragraph slice with min null score
        no_start_logit = 0  # the start logit at the slice with min null score
        no_end_logit = 0  # the end logit at the slice with min null score

        for (feature_index, feature) in enumerate(features):
            # 对于某个片段，计算得分
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            if version_2_with_negative:
                feature_null_score = result.unk_logits[0]*2
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.unk_logits[0]
                    null_end_logit = result.unk_logits[0]

                feature_yes_score = result.yes_logits[0] + result.yes_logits[0]
                if feature_yes_score < score_yes:
                    score_yes = feature_yes_score
                    min_yes_feature_index = feature_index
                    yes_start_logit = result.yes_logits[0]
                    yes_end_logit = result.yes_logits[0]

                feature_no_score = result.no_logits[0] + result.no_logits[0]
                if feature_no_score < score_no:
                    score_no = feature_no_score
                    min_no_feature_index = feature_index
                    no_start_logit = result.no_logits[0]
                    no_end_logit = result.no_logits[0]

            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))
        if version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=512,
                    end_index=512,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit))
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_yes_feature_index,
                    start_index=513,
                    end_index=513,
                    start_logit=yes_start_logit,
                    end_logit=yes_end_logit))
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_no_feature_index,
                    start_index=514,
                    end_index=514,
                    start_logit=no_start_logit,
                    end_logit=no_end_logit))
        # 排序
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index < 512:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = "".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = "".join(tok_text.split())
                orig_text = "".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
            elif pred.start_index == 512:
                final_text = ""
            elif pred.start_index == 513:
                final_text = "YES"
            else:
                final_text = "NO"

            if final_text in seen_predictions:
                continue
            seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit))
        # # # if we didn't include the empty option in the n-best, include it
        # if version_2_with_negative:
        #     if "" not in seen_predictions:
        #         nbest.append(
        #             _NbestPrediction(
        #                 text="",
        #                 start_logit=null_start_logit,
        #                 end_logit=null_end_logit))
        #
        #     # In very rare edge cases we could only have single null prediction.
        #     # So we just create a nonce prediction in this case to avoid failure.
        #     if len(nbest) == 1:
        #         nbest.insert(0,
        #                      _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))
        #
        # # In very rare edge cases we could have no valid predictions. So we
        # # just create a nonce prediction in this case to avoid failure.
        # if not nbest:
        #     nbest.append(
        #         _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        # best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            # if not best_non_null_entry:
            #     if entry.text:
            #         best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        all_predictions[example.qas_id] = nbest_json[0]["text"]
        all_nbest_json[example.qas_id] = nbest_json

    return all_predictions, all_nbest_json
