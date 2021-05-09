# -*- coding: utf-8 -*-
from keras import layers
from keras.models import Model
from keras.models import load_model
import all_data
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
import numpy as np
from keras.optimizers import Adam
from keras import callbacks

x,y=all_data.get_trainset()

#调用已经训练好地自编码器对数据进行重构
autoencoder=load_model("model/autoencoder.hf5")
x=autoencoder.predict(x)

#数据归一化
#x=scale(x)
x=(x-x.mean())/x.std()
y=np.reshape(y,(y.shape[0]))

#数据集划分
x_train,x_val,y_train,y_val=train_test_split(x,y,test_size=0.3)

#构建模型
time_steps=x_train.shape[1]#时间步维度
input_dim=x_train.shape[2]#特证数维度
latent_dim=128#隐层神经元个数
inputer=layers.Input(shape=(time_steps,input_dim))
lstm1=layers.LSTM(latent_dim,return_sequences=True)(inputer)
lstm2=layers.LSTM(latent_dim)(lstm1)
dropout=layers.Dropout(0.2)(lstm2)
outputer=layers.Dense(1)(dropout)
model=Model(inputs=inputer,outputs=outputer)

#模型编译
model.compile(optimizer=Adam(lr=0.001),loss="mae",metrics=['accuracy'])

#保存模型
checkpoint=callbacks.ModelCheckpoint(filepath="model/LSTM_SAE.hf5",verbose=1)
#自动调整学习率
lr=callbacks.ReduceLROnPlateau(min_lr=0.00001,patience=3)

#模型训练
model.fit(x_train,y_train,epochs=50,batch_size=30,validation_data=(x_val,y_val),callbacks=[checkpoint,lr])