# -*- coding: utf-8 -*-
import lightgbm as lgb
from sklearn.preprocessing import scale
import numpy as np
import processing3
x_train,y_train=processing3.get_trainset()

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
lgb_model.save_model('model/lgb_preprocessing3.txt',num_iteration=lgb_model.best_iteration)
