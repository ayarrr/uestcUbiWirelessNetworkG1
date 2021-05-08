# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 10:44:31 2019

@author: Administrator
"""

import tensorflow as tf
from basic_layer.NN_adam import NN
from util.Pooler import pooler
from selfAttention import embedding,positional_encoding,multihead_attention,feedforward,label_smoothing
from util.Resultpro import cal_result
import pandas as pd
import numpy as np
class AttNN(NN):
    """
    The memory network with context attention.
    """
    # ctx_input.shape=[batch_size, mem_size]

    def __init__(self, config):
        super(AttNN, self).__init__(config)
        self.config = None
        if config != None:
            self.config = config
            # the config.
            self.datas = config['dataset']
            self.nepoch = config['nepoch']  # the train epoches.
            self.batch_size = config['batch_size']  # the max train batch size.
            self.init_lr = config['init_lr']  # the initialize learning rate.
            # the base of the initialization of the parameters.
            self.stddev = config['stddev']
            self.edim = config['edim']  # the dim of the embedding.
            self.hidden=config['hidden_size']#the dim of selfattention output
            self.max_grad_norm = config['max_grad_norm']   # the L2 norm.
            self.cut_off=config['cut_off']
            # the pad id in the embedding dictionary.
#            self.pad_idx = config['pad_idx']

            # the pre-train embedding.
            ## shape = [nwords, edim]
            # self.pre_embedding = np.array(config['pre_embedding'])#这个还没加
            self.totalitemNum=config['TotalNumberOf']#总item数
            self.candidatesNUm=config["candidates"]#候选列表中item的总个数
            # generate the pre_embedding mask.
#            self.pre_embedding_mask = np.ones(np.shape(self.pre_embedding))
#            self.pre_embedding_mask[self.pad_idx] = 0

            # update the pre-train embedding or not.
#            self.emb_up = config['emb_up']

            # the active function.
            self.active = config['active']#默认的是sigmoid
            self.TMaxSequencelen=config['TMaxSequencelen']#最大序列长度

            # hidden size  
#            self.hidden_size = config['hidden_size']

            self.is_print = config['is_print']

        self.is_first = True
        # the input.
        self.inputs = None
        self.aspects = None
        # sequence length
        self.sequence_length = None
        self.reverse_length = None
        self.aspect_length = None
        # the label input. (on-hot, the true label is 1.)
        self.lab_input = None
        self.embe_dict = None  # the embedding dictionary.
        # the optimize set.
        self.global_step = None  # the step counter.
        self.loss = None  # the loss of one batch evaluate.
        self.lr = None  # the learning rate.
        self.optimizer = None  # the optimiver.
        self.optimize = None  # the optimize action.
        # the mask of pre_train embedding.
        self.pe_mask = None
        # the predict.
        self.pred = None
        # the params need to be trained.
        self.params = None


    def build_model(self,is_training,w):
        '''
        build the MemNN model
        '''
        # the input.
        self.inputs = tf.placeholder(
            tf.int32,
            [None,None],
            name="inputs"
        )
        self.actiontype = tf.placeholder(
            tf.int32,
            [None,None],
            name="actiontype"
        )
#        self.prices = tf.placeholder(
#            tf.float32,
#            [None,None],
#            name="prices"
#        )

        self.last_inputs = tf.placeholder(
            tf.int32,
            [None],
            name="last_inputs"
        )
        batch_size = tf.shape(self.inputs)[0]
        self.label = tf.placeholder(
            tf.int32,
            [None],
            name="label"
        )




        self.sequence_length = tf.placeholder(
            tf.int64,
            [None],
            name='sequence_length'
        )

        self.candidates=tf.placeholder(
            tf.int64,
            [None,None],
            name='candidates'
        )
        self.userEM=tf.placeholder(
            tf.float32,
            [None,None],#user的维度应该是一致的
            name='userEM'
        )

        # the lookup dict.
        # self.embe_dict = tf.Variable(
        #     self.pre_embedding,
        #     # [None, None],
        #     # dtype=tf.float32,
        #     # trainable=False
        # )
        self.embe_dict = tf.get_variable(name="lookup_dict",
                                         shape=w.shape,
                        initializer=tf.constant_initializer(w),
                        trainable=False)

#        self.pe_mask = tf.Variable(
#            self.pre_embedding_mask,
#            dtype=tf.float32,
#            trainable=False
#        )
        
        #self-Attention
        with tf.variable_scope("encoder"):
            # Embedding
            self.enc =embedding(self.inputs,
                                 vocab_size=self.totalitemNum,#总个数
                                 num_units = self.hidden,
                                 zero_pad=False, # 让padding一直是0
                                 scale=True,
                                 scope="enc_embed")

            self.enc +=embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.inputs)[1]),0),[tf.shape(self.inputs)[0],1]),
                                 vocab_size = self.TMaxSequencelen,#一个序列中最多容纳的元素个数
                                 num_units = self.hidden,
                                 zero_pad = False,
                                 scale = False,
                                 scope = "enc_pe")
            # self.enc += positional_encoding(self.inputs,
            #                                 num_units =self.hidden,
            #                                 zero_pad = False,
            #                                 scale = False,
            #                                 scope='enc_pe')
            ##Drop out
            self.enc = tf.layers.dropout(self.enc,rate =0.1,training = tf.convert_to_tensor(is_training))
            # print(tf.shape(self.inputs))
            self.enc = multihead_attention(queries = self.enc,
                                                   keys = self.enc,
                                                   num_units = self.hidden,
                                                   num_heads = 4,
                                                   dropout_rate = 0.1,
                                                   is_training = True,#是否要dropout
                                                   causality = False
                                                   )
            self.enc = feedforward(self.enc,num_units = [4 * self.hidden,self.hidden])#b,s,hidden

        self.enc=tf.keras.layers.Dense(self.hidden,activation='relu')(self.enc)
        sum_pool=pooler(
            self.enc,
            'sum',
            axis=1
        )  # 这里的输出 batch_size * (d+1)
        self.enc=tf.reshape(sum_pool,[-1,self.hidden])
        # print("self.enc",tf.shape(self.enc))
        #user偏好 1.当前物品偏好
#        self.embe_dict *= self.pe_mask
        inputs = tf.nn.embedding_lookup(self.embe_dict, self.inputs,max_norm=1)
        lastinputs= tf.nn.embedding_lookup(self.embe_dict, self.last_inputs,max_norm=1)
        candidates=tf.nn.embedding_lookup(self.embe_dict, self.candidates,max_norm=1)
        seqInput=tf.concat([inputs, tf.cast(tf.expand_dims(self.actiontype, -1),tf.float32)], 2)#shape=(b,sequenlen,d+1)
        
        pool_seq_len=tf.cast(tf.reshape(self.sequence_length,[batch_size, 1]), tf.float32)
        onepool=tf.ones_like(pool_seq_len)
        pool_seq_len=tf.add(pool_seq_len,onepool)
        pool_out = pooler(
            seqInput,
            'mean',
            axis=1,
            sequence_length = pool_seq_len
        )#这里的输出 batch_size * (d+1)
        pool_out = tf.reshape(pool_out,[-1,self.edim+1])
        
        
        self.w1 = tf.Variable(
            tf.random_normal([self.edim+1, self.edim], stddev=self.stddev),
            trainable=True
        )

        self.w2 = tf.Variable(
            tf.random_normal([self.edim, self.edim], stddev=self.stddev),
            trainable=True
        )
        self.w3 = tf.Variable(
            tf.random_normal([self.edim, self.edim], stddev=self.stddev),
            trainable=True
        )
        self.w4 = tf.Variable(
            tf.random_normal([self.hidden, self.edim], stddev=self.stddev),
            trainable=True
        )
        attout = tf.tanh(tf.matmul(pool_out,self.w1))#MLP A
        # attout = tf.nn.dropout(attout, self.output_keep_probs)
        lastinputs= tf.tanh(tf.matmul(lastinputs,self.w2))#MLP B
        useroutput= tf.tanh(tf.matmul(self.userEM,self.w3))#MLP C
        SAsequence= tf.tanh(tf.matmul(self.enc,self.w4))#MLP C
        prod = attout + lastinputs+useroutput+SAsequence
        sco_mat = tf.reshape(tf.matmul(tf.reshape(prod,[batch_size,1,self.edim]),candidates,transpose_b= True),[batch_size,self.candidatesNUm])#shape=(batch_size,1,25)
        self.modeloutput=sco_mat
        print("sco_mat", tf.shape(sco_mat))
        # self.label=label_smoothing(tf.reshape(self.label,[batch_size]))
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=sco_mat,labels =self.label)

        # the optimize.
        self.params = tf.trainable_variables()
        self.optimize = super(AttNN, self).optimize_normal(
            self.loss, self.params)
     
        return

    def get_batch(self,data,UserEMdict1):
        def strlist2intlist(strlist):#把str类型的list转换成int类型的list
            tI=strlist[1:-1].split(",")
            intlist = list(map(int, tI))
            return intlist
        def getd(d):
            sessionid = list(d["sessionid"])
            user = list(map(lambda x: np.reshape(UserEMdict1[x],(1,self.edim)), list(d["userid"])))
            items = list(map(strlist2intlist,list(d["itemlist"])))
            impressions =  list(map(strlist2intlist,list(d["impressions"])))#此处要保证还有candidates个
            seqlen = d["seqlen"].values
            A_T =  list(map(strlist2intlist,list(d["action_type"])))
            last_Item = list(d["last_item"])
            label = list(d["label"])  # label需要处理，输出层需要统一，改成位置编号
            labelzero = np.zeros([len(label)])
            fimpression = []
            for l in range(len(label)):  # 把每个session的label转换成一个类别编码序列，位于impression的第几个最终被点了，该位置为几
    #            print(impressions[l])
                batchImpression = impressions[l]
                if len(batchImpression)>25:
                    batchImpression=batchImpression[:self.candidatesNUm]
    # 此处要保证还有candidates个
                if label[l] in batchImpression:
                    labelzero[l] = batchImpression.index(label[l])
                else:#最终用户点了impression以外的item  更新impression(更新最后一个item，并把label换成最后一个),
                    batchImpression[-1]=label[l]
                    labelzero[l]=len(batchImpression)-1
                if len(batchImpression)!=self.candidatesNUm:
                    print(len(batchImpression))
                fimpression.append(batchImpression)
            return (sessionid,user, items, fimpression, seqlen, A_T, last_Item, labelzero)
        if "seqlen" not in data.columns.tolist():
            data.insert(1, 'seqlen', data["sequencelen"])
        alldata = dict(data.groupby("sequencelen").apply(lambda x: getd(x)))
        return alldata
    def train(self,sess,UserEMdict1,UserEMdict2, train_data, test_data=None,saver = None, threshold_acc=0.99):
        """
        UserEMdict1  当前train中所有user的embedding,dict类型
        UserEMdict2  当前test中所有user的embedding,dict类型
        train_data dataframe类型,
        包含"userid","itemlist","impressions","prices","sequencelen","action_type","last_item","label"，"sessionid"
        
        """
        max_recall = 0.0
        max_mrr = 0.0
        max_train_acc = 0.0
        simbatch=self.get_batch(train_data,UserEMdict1)#得到一个dict，key为每组中序列长度，value为元祖
        for epoch in range(self.nepoch):   # epoch round.
            batch = 0
            c = []
            for k,v in simbatch.items():
                batch_lenth = len(v[0])
                if batch_lenth>self.batch_size:#每组过多需要再分
                    patch_len = int(batch_lenth / self.batch_size) #总组数
                    remain = int(batch_lenth % self.batch_size)#剩余的做一个batch
                    i = 0
                    for x in range(patch_len):
                        b_sessionids=v[0][i:i+self.batch_size]
                        b_user=np.array(v[1][i:i+self.batch_size]).reshape((self.batch_size,self.edim))
                        b_inputs=np.array(v[2][i:i+self.batch_size]).reshape((self.batch_size,k))
                        b_impressions=np.array(v[3][i:i+self.batch_size]).reshape((self.batch_size,25))
                        b_seqlen=np.array(v[4][i:i+self.batch_size]).reshape((self.batch_size))
                        b_actionType=np.array(v[5][i:i+self.batch_size]).reshape((self.batch_size,k))
                        b_last_Input=np.array(v[6][i:i+self.batch_size]).reshape((self.batch_size))
                        b_label=np.array(v[7][i:i+self.batch_size]).reshape((self.batch_size))
                        feed_dict = {
                                self.inputs: b_inputs,
                                self.actiontype:b_actionType,
                                self.last_inputs: b_last_Input,
                                self.label: b_label,
                                self.sequence_length: b_seqlen,
                                self.candidates:b_impressions,
                                self.userEM:b_user
                            }
                        # train
                        crt_loss, crt_step, opt, embe_dict = sess.run(
                            [self.loss, self.global_step, self.optimize, self.embe_dict],
                            feed_dict=feed_dict
                        )
                        c += list(crt_loss)
                        batch += 1
                        
                        i += self.batch_size
                    if remain > 0:#多余的另放一个batch
                        b_sessionids=v[0][i:]
                        b_user=np.array(v[1][i:]).reshape((remain,self.edim))
                        b_inputs=np.array(v[2][i:]).reshape((remain,k))
                        b_impressions=np.array(v[3][i:]).reshape((remain,25))
                        b_seqlen=np.array(v[4][i:]).reshape((remain))
                        b_actionType=np.array(v[5][i:]).reshape((remain,k))
                        b_last_Input=np.array(v[6][i:]).reshape((remain))
                        b_label=np.array(v[7][i:]).reshape((remain))
                        feed_dict = {
                                self.inputs: b_inputs,
                                self.actiontype:b_actionType,
                                self.last_inputs: b_last_Input,
                                self.label: b_label,
                                self.sequence_length: b_seqlen,
                                self.candidates:b_impressions,
                                self.userEM:b_user
                            }
                        # train
                        crt_loss, crt_step, opt, embe_dict = sess.run(
                            [self.loss, self.global_step, self.optimize, self.embe_dict],
                            feed_dict=feed_dict
                        )
                        c += list(crt_loss)
                        batch += 1
                else:

                     b_sessionids=v[0]
                     b_user=np.array(v[1]).reshape((batch_lenth,self.edim))
                     b_inputs=np.array(v[2]).reshape((batch_lenth,k))
                     b_impressions=np.array(v[3]).reshape((batch_lenth,25))
                     b_seqlen=np.array(v[4]).reshape((batch_lenth))
                     b_actionType=np.array(v[5]).reshape((batch_lenth,k))
                     b_last_Input=np.array(v[6]).reshape((batch_lenth))
                     b_label=np.array(v[7]).reshape((batch_lenth))
                     feed_dict = {
                                self.inputs: b_inputs,
                                self.actiontype:b_actionType,
                                self.last_inputs: b_last_Input,
                                self.label: b_label,
                                self.sequence_length: b_seqlen,
                                self.candidates:b_impressions,
                                self.userEM:b_user
                            }
                    # train
                     crt_loss, crt_step, opt, embe_dict = sess.run(
                        [self.loss, self.global_step, self.optimize, self.embe_dict],
                        feed_dict=feed_dict)
                     c += list(crt_loss)
                     batch += 1
            avgc = np.mean(c)
            if np.isnan(avgc):
                print('Epoch {}: NaN error!'.format(str(epoch)))
                self.error_during_train = True
                return  
            print('Epoch{}\tloss: {:.6f}'.format(epoch, avgc))
            if bool(1-test_data.empty):
                recall, mrr,test_result = self.test(sess,UserEMdict2, test_data)
                print('On Test Data Epoch{}\tRecall@20: {}\tMRR@20:{}'.format(epoch,recall, mrr))
                if max_recall < recall:
                    max_recall = recall
                    max_mrr = mrr
#                    test_data.update_best()
                    if max_recall > threshold_acc:
                        test_result.to_csv("../output/SuperFusion_recomends.csv", index=False)
                        self.save_model(sess, self.config, saver)
                print ("***Max_recall: " + str(max_recall)+" max_mrr: "+str(max_mrr))
                # test_data.clear()
                        
                        
                    
            
    

    def test(self,sess,UserEMdict,test_data):

        # calculate the acc
        print('Measuring Recall@{} and MRR@{}'.format(self.cut_off, self.cut_off))

        sessionid,mrr, recall,sessionlenth,request_EXtra = [],[],[], [],[]
        c_loss =[]
        batch = 0
        simbatch=self.get_batch(test_data,UserEMdict)#得到一个dict，key为每组中序列长度，value为元祖
        batch = 0
        for k,v in simbatch.items():
            batch_lenth = len(v[0])
            if batch_lenth>self.batch_size:#每组过多需要再分
                patch_len = int(batch_lenth / self.batch_size) #总组数
                remain = int(batch_lenth % self.batch_size)#剩余的做一个batch
                i = 0
                for x in range(patch_len):
                    b_sessionids=v[0][i:i+self.batch_size]
                    b_user=np.array(v[1][i:i+self.batch_size]).reshape((self.batch_size,self.edim))
                    b_inputs=np.array(v[2][i:i+self.batch_size]).reshape((self.batch_size,k))
                    b_impressions=np.array(v[3][i:i+self.batch_size]).reshape((self.batch_size,25))
                    b_seqlen=np.array(v[4][i:i+self.batch_size]).reshape((self.batch_size))
                    b_actionType=np.array(v[5][i:i+self.batch_size]).reshape((self.batch_size,k))
                    b_last_Input=np.array(v[6][i:i+self.batch_size]).reshape((self.batch_size))
                    b_label=np.array(v[7][i:i+self.batch_size]).reshape((self.batch_size))
                    feed_dict = {
                            self.inputs: b_inputs,
                            self.actiontype:b_actionType,
                            self.last_inputs: b_last_Input,
                            self.label: b_label,
                            self.sequence_length: b_seqlen,
                            self.candidates:b_impressions,
                            self.userEM:b_user
                        }
                        # test
                    preds, loss= sess.run(
                        [self.modeloutput, self.loss],
                        feed_dict=feed_dict
                    )
                    t_r, t_m, ranks,request_extra = cal_result(preds, b_label)#返回各个样本算的recall@5,mrr@25,rank
#                        test_data.pack_ext_matrix('alpha', alpha, tmp_batch_ids)
#                        test_data.pack_preds(ranks, tmp_batch_ids)
                    sessionid+=b_sessionids
                    sessionlenth+=list(v[4][i:i+self.batch_size])
                    c_loss += list(loss)
                    recall += t_r
                    request_EXtra+=request_extra
                    mrr += t_m
                    batch += 1
                    
                    i += self.batch_size
                if remain > 0:#多余的另放一个batch
                    b_sessionids=v[0][i:]
                    b_user=np.array(v[1][i:]).reshape((remain,self.edim))
                    b_inputs=np.array(v[2][i:]).reshape((remain,k))
                    b_impressions=np.array(v[3][i:]).reshape((remain,25))
                    b_seqlen=np.array(v[4][i:]).reshape((remain))
                    b_actionType=np.array(v[5][i:]).reshape((remain,k))
                    b_last_Input=np.array(v[6][i:]).reshape((remain))
                    b_label=np.array(v[7][i:]).reshape((remain))
                    feed_dict = {
                            self.inputs: b_inputs,
                            self.actiontype:b_actionType,
                            self.last_inputs: b_last_Input,
                            self.label: b_label,
                            self.sequence_length: b_seqlen,
                            self.candidates:b_impressions,
                            self.userEM:b_user
                        }
                     # test
                    preds, loss= sess.run(
                        [self.modeloutput, self.loss],
                        feed_dict=feed_dict
                    )
                    t_r, t_m, ranks,request_extra =cal_result(preds, b_label)#返回各个样本算的recall@5,mrr@25,rank
#                        test_data.pack_ext_matrix('alpha', alpha, tmp_batch_ids)
#                        test_data.pack_preds(ranks, tmp_batch_ids)
                    sessionid+=b_sessionids
                    sessionlenth += list(v[4][i:])

                    c_loss += list(loss)
                    recall += t_r
                    request_EXtra+=request_extra
                    mrr += t_m
                    batch += 1
            else:
                 b_sessionids=v[0]
                 b_user=np.array(v[1]).reshape((batch_lenth,self.edim))
                 b_inputs=np.array(v[2]).reshape((batch_lenth,k))
                 b_impressions=np.array(v[3]).reshape((batch_lenth,25))
                 b_seqlen=np.array(v[4]).reshape((batch_lenth))
                 b_actionType=np.array(v[5]).reshape((batch_lenth,k))
                 b_last_Input=np.array(v[6]).reshape((batch_lenth))
                 b_label=np.array(v[7]).reshape((batch_lenth))
                 feed_dict = {
                            self.inputs: b_inputs,
                            self.actiontype:b_actionType,
                            self.last_inputs: b_last_Input,
                            self.label: b_label,
                            self.sequence_length: b_seqlen,
                            self.candidates:b_impressions,
                            self.userEM:b_user
                        }
                 # test
                 preds, loss= sess.run(
                    [self.modeloutput, self.loss],
                    feed_dict=feed_dict)
                 t_r, t_m, ranks,request_extra = cal_result(preds, b_label)#返回各个样本算的recall@20,mrr@20,rank,额外请求个数
#                        test_data.pack_ext_matrix('alpha', alpha, tmp_batch_ids)
#                        test_data.pack_preds(ranks, tmp_batch_ids)
                 sessionid+=b_sessionids
                 sessionlenth += list(v[4])
                 c_loss += list(loss)
                 recall += t_r
                 request_EXtra+=request_extra
                 mrr += t_m
                 batch += 1
        test_result=pd.DataFrame({"sessionId":sessionid,"recall@20":recall,"MRR@20":mrr,"sessionlenth":sessionlenth,"request_EXtra": request_EXtra})
        print ("loss in all test data:",np.mean(c_loss))
        return  np.mean(recall), np.mean(mrr),test_result
