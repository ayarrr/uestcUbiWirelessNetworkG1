# -*- coding: utf-8 -*-
import lightgbm as lgb
import pandas as pd
from sklearn.preprocessing import scale
import numpy as np
data=pd.read_csv("processing1/train.csv")
train_label=[x for x in data.columns if x != 'relation']
x_train=data[train_label]
y_train=data['relation']

#数据归一化
x_train=scale(np.asarray(x_train))
y_train=np.asarray(y_train)
y_train=np.reshape(y_train,(y_train.shape[0],))

#构造训练集
train_data=lgb.Dataset(x_train,label=y_train)

#设置参数
param={
      'num_leaves':120,
      'objective':'regression',
      'max_depth':7,
      'learning_rate':0.075,
      'max_bin':120}
param['metric']=['mae']

#训练
num_round=50
lgb_model=lgb.train(param,train_data,num_round)
#保存模型
lgb_model.save_model('model/lgb_preprocessing1.txt',num_iteration=lgb_model.best_iteration)
