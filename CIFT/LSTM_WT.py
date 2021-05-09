# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import all_data
import denoise_WT
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from keras import layers
from keras.models import Model
from keras import callbacks
from keras.optimizers import Adam

#1.取数据
x,y=all_data.get_trainset()

#2.进行小波降噪（一维8阶分解，小波基函数使用haar,去噪层数为第8层）
x=denoise_WT.deniose_dataset(x)

#数据归一化
#x=scale(x)
x=(x-x.mean())/x.std()
y=np.reshape(y,(y.shape[0],))

#训练集与验证集分割
x_train,x_val,y_train,y_val=train_test_split(x,y,test_size=0.3)

#构建模型
time_steps=x_train.shape[1]
input_dim=x_train.shape[2]
latent_dim=128
inputer=layers.Input(shape=(time_steps,input_dim))
lstm1=layers.LSTM(latent_dim,return_sequences=True)(inputer)
lstm2=layers.LSTM(latent_dim)(lstm1)
outputer=layers.Dense(1)(lstm2)
model=Model(inputs=inputer,outputs=outputer)

#编译模型
model.compile(optimizer=Adam(lr=0.001),loss="mae",metrics=['accuracy'])

#保存模型和自动参数调整
checkpoint=callbacks.ModelCheckpoint(filepath="model/LSTM_WT.hf5",verbose=1)
lr=callbacks.ReduceLROnPlateau(min_lr=0.00001,patience=3)

#训练模型
model.fit(x_train,y_train,epochs=50,batch_size=30,validation_data=(x_val,y_val),callbacks=[checkpoint,lr])