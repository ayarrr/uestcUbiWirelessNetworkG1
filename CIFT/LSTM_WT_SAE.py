# -*- coding: utf-8 -*-
import series_SAE
import all_data
import denoise_WT
import numpy as np
from keras.models import load_model
from keras.models import Model
from keras.optimizers import Adam
from keras import callbacks
from keras import layers
from sklearn.model_selection import train_test_split

#读入数据
x,y=all_data.get_trainset()

#对数据进行归一化
x=(x-x.mean())/x.std()
y=np.reshape(y,(y.shape[0],))

#对数据进行降噪
x=denoise_WT.deniose_dataset(x)

#使用堆叠式序列自编码器对数据进行处理
autoencoder=load_model("model/autoencoder.hf5")
x=autoencoder.predict(x)

#训练集与验证集划分
x_train,x_val,y_train,y_val=train_test_split(x,y,test_size=0.3)

#构建模型
input_dim=x_train.shape[2]
time_steps=x_train.shape[1]
latent_dim=128
inputer=layers.Input(shape=(time_steps,input_dim))
lstm1=layers.LSTM(latent_dim)(inputer)
lstm2=layers.LSTM(latent_dim)(lstm1)
outputer=layers.Dense(1)(lstm2)
model=Model(inputs=inputer,outputs=outputer)

#编译模型
model.compile(optimizer=Adam(lr=0.001),loss='mae',metrics=['accuracy'])

#保存模型和自动调参
checkpoint=callbacks.ModelCheckpoint(filepath="model/LSTM_WT_SAE.hf5",verbose=1)
lr=callbacks.ReduceLROnPlateau(min_lr=0.00001,patience=3)

#模型训练
model.fit(x_train,y_train,epochs=50,batch_size=30,validation_data=(x_val,y_val),callbacks=[checkpoint,lr])