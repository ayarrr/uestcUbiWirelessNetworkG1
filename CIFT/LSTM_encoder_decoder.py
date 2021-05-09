# -*- coding: utf-8 -*-

import all_data
import numpy as np
from sklearn.model_selection import train_test_split
from keras import layers
from keras.models import Model
from keras.optimizers import Adam
from keras import callbacks

#读入数据
x,y=all_data.get_trainset()

#对数据进行归一化
x=(x-x.mean())/x.std()
y=np.reshape(y,(y.shape[0]))

#数据集划分
x_train,x_val,y_train,y_val=train_test_split(x,y,test_size=0.3)

#构建模型
time_step=x_train.shape[1]
input_dim=x_train.shape[2]
latent_dim=64
inputer=layers.Input(shape=(time_step,input_dim))
encoder1=layers.LSTM(latent_dim,return_sequences=True)(inputer)
encoder2=layers.LSTM(latent_dim)(encoder1)
output1=layers.RepeatVector(3)(encoder2)
decoder1=layers.LSTM(latent_dim,return_sequences=True)(output1)
decoder2=layers.LSTM(latent_dim)(decoder1)
outputer=layers.Dense(1)(decoder2)
model=Model(inputs=inputer,outputs=outputer)

#编译
model.compile(optimizer=Adam(lr=0.001),loss="mae",metrics=['accuracy'])

#保存模型和自动调参
checkpoint=callbacks.ModelCheckpoint(filepath="model/LSTM_encoder_decoder.hf5",verbose=1)
lr=callbacks.ReduceLROnPlateau(min_lr=0.00001,patience=3)

#训练
history=model.fit(x_train,y_train,epochs=30,batch_size=30,validation_data=(x_val,y_val),callbacks=[checkpoint,lr])
val_acc=np.asarry(history.history['val_acc'])
np.save("processing1/Encoder_decoder_train.npy",val_acc)
