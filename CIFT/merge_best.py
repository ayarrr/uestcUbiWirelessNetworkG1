# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 09:39:19 2018

@author: 10097
"""

import processing1
import processing3
import processing2
import all_data
import lightgbm as lgb
from keras.models import load_model
import numpy as np
from sklearn import metrics
#读取lightgbm的数据集
data1=[]
target1=[]

#读取LSTM的数据集
data2,target1=all_data.get_testset

#读取lightgbm模型并进行预测
path1=""
lgb_model=lgb.Booster(model_file=path1)
predict_lgb=lgb_model.predict(data1)
predict_lgb=np.asarray(predict_lgb)

#读取LSTM模型并进行预测
path2=""
lstm_model=load_model(path2)
predict_lstm=lstm_model.predict(data2)
predict_lstm=np.asarray(predict_lstm)

#从预测结果中取最大值和最小值
maxer=[max(predict_lstm[i],predict_lgb[i]) for i in range(len(predict_lgb))]
miner=[min(predict_lstm[i],predict_lgb[i]) for i in range(len(predict_lgb))]

#进行评价
def GetScore(ypred,y_test):
    mae=0
    tmape=0
    n=len(y_test)
    for i in range(n):
        p=ypred[i]
        t=y_test[i]
        mae=mae+abs(p-t)
        tmape=tmape+abs((p-t)/(1.5-t))
    mae=mae/n
    tmape=tmape/n
    score=(2/(2+mae+tmape))**2
    return score

print (u"较大值融合：")
print ("MAE:",metrics.mean_absolute_error(target1,maxer))
print ("RMSE:",np.sqrt(metrics.mean_squared_error(target1,maxer)))
print ("score:",GetScore(maxer,target1))

print (u"较小值值融合：")
print ("MAE:",metrics.mean_absolute_error(target1,miner))
print ("RMSE:",np.sqrt(metrics.mean_squared_error(target1,miner)))
print ("score:",GetScore(miner,target1))