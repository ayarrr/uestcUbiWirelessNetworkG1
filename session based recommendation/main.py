# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 09:13:51 2019

@author: Administrator
"""

import tensorflow as tf
import pandas as pd
import numpy as np
from SRSmodel import AttNN as model1
from util.Randomer import Randomer
import data_load as dl
import pickle
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
config={"dataset" : "rsc19",
        "class_num":2019,
        "nepoch" : 10,
    "is_print" : True,
    "batch_size" : 512,#最大的batch_size
    "init_lr" : 0.003,
     "stddev" : 0.05,
    "emb_stddev" : 0.002,
    "max_grad_norm" :150,# 150 if set None, while not clip the grads.
    "active" : "sigmoid",
    "cell" : "gru",
    "hidden_size" : 128,
    "edim" : 157,
    "cut_off":20,
    "emb_up": True,# should update the pre-train embedding.
    "pad_idx" : 0,
    "model_save_path" : "ckpt/",
        "model":"SRS",
    "update_lr" : False,
    "recsys_threshold_acc" : 0.68,#训练目标
    "TotalNumberOf":100,#所涉及到的item总数
    "ratio":1/64,#使用的数据的比例 train的session数 75113 test的session数 77463
    "candidates":25#给出的候选item的个数
        
        }
def getData(is_traindata,ratio):
    if is_traindata:
         try:
             dfm=pd.read_csv("../processed/sessionSequences_for_train"+str(ratio)+".csv")
             userEMdict=pickle.load(open("../processed/userEMdict_for_train"+str(ratio)+".pkl", 'rb'))
             #"../processed/userEMdict_"+label+str(ratio)+".pkl"
             return dfm,userEMdict
         except IOError:
             print("get train sequence...")
             return dl.savecsv("../processed/mytrain_"+str(ratio)+".csv","for_train",ratio)
    else:
        try:
            dfm=pd.read_csv("../processed/sessionSequences_for_test"+str(ratio)+".csv")
            userEMdict=pickle.load(open("../processed/userEMdict_for_test"+str(ratio)+".pkl", 'rb'))
            return dfm,userEMdict
        except IOError:
            print("get test sequence...")
            return dl.savecsv("../processed/mytest_"+str(ratio)+".csv","for_test",ratio)
# def getTotalItem(train_data,test_data):
#     allItem=[]
#     train_data.apply(lambda x: allItem.extend(x["itemlist"]), axis=1)
#     allItem.extend(list(train_data["label"]))
#
#     test_data.apply(lambda x: allItem.extend(x["itemlist"]), axis=1)
#     allItem.extend(list(test_data["label"]))
#     allItem=np.array(allItem).flatten()
#     return len(set(allItem))
def getmaxLen(train_data,test_data):#找sequence的最大长度
    t1=train_data.max()["sequencelen"]
    t2=test_data.max()["sequencelen"]
    if t1>t2:
        return t1
    else:
        return t2
def main():
    is_train = True
    is_save = True
    train_data,UserEMdict1=getData(is_train,config['ratio'])
    test_data,UserEMdict2=getData(False,config['ratio'])
    input_data="test"
    model_path="SSR.ckpt-lap"
    config['TotalNumberOf']=927142#设太小会出现找不到
    config['TMaxSequencelen'] =getmaxLen(train_data, test_data)
    # print(dl.vocab.dictValue2Array().shape)(927142, 157)
    # setup randomer

    Randomer.set_stddev(config['stddev'])

    with tf.Graph().as_default():
        # build model
        ANN=model1(config)
        ANN.build_model(is_train,dl.vocab.dictValue2Array())
        if is_save or not is_train:
            saver = tf.train.Saver(max_to_keep=30)
        else:
            saver = None
        # run
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if is_train:
                print("start training...")
                ANN.train(sess,UserEMdict1,UserEMdict2, train_data, test_data, saver, threshold_acc=config['recsys_threshold_acc'])
            else:
                if input_data is "test":
                    sent_data = test_data
                elif input_data is "train":
                    sent_data = train_data
                else:
                    sent_data = test_data
                saver.restore(sess, model_path)
                ANN.test(sess,UserEMdict2, sent_data)

    
    return

main()
# def strlist2intlist(strlist):  # 把str类型的list转换成int类型的list
#     tI = strlist[1:-1].split(",")
#     intlist = list(map(int, tI))
#     return intlist
# dfm=pd.read_csv("../processed/sessionSequences_for_train"+str(1/64)+".csv")
# for d in list(dfm["impressions"]):
#     sint=strlist2intlist(d)
#     if len(sint)>25:
#         print(sint)

