# -*- coding: utf-8 -*-
from keras import layers
from keras.models import Model
from keras import callbacks
from sklearn.preprocessing import scale
import all_data
import numpy as np

x,y=all_data.get_trainset()

#数据归一化
#x=scale(x)
x=(x-x.mean())/x.std()
y=np.reshape(y,(y.shape[0]))

#使用5层SAE，每层里面都是序列自编码器
latent_dim=128
time_steps=x.shape[1]
input_dim=x.shape[2]
#编码部分
inputer=layers.Input(shape=(time_steps,input_dim))
#在除最后一个编码层以外的各个编码层添加 return_sequences=True
encoder1=layers.LSTM(latent_dim,return_sequences=True)(inputer)
#encoder2=layers.LSTM(latent_dim,return_sequences=True)(encoder1)
encoder3=layers.LSTM(latent_dim)(encoder1)
#RepeatVector层只在这儿使用，它用来存储序列的长度
repeat3=layers.RepeatVector(time_steps)(encoder3)
#encoder4=layers.LSTM(latent_dim,return_sequences=True)(repeat3)
#解码部分
#LSTM的输出维度为(samples，timesteps，output_dim),输入与输出形状相同
decoder=layers.LSTM(input_dim,return_sequences=True)(repeat3)

SAE=Model(inputs=inputer,outputs=decoder)

#编译模型
SAE.compile(optimizer="adam",loss="mae",metrics=['accuracy'])

#保存自编码器
checkpoint=callbacks.ModelCheckpoint(filepath="model/autoencoder.hf5",verbose=1)
#自动调整学习率
#lr=callbacks.ReduceLROnPlateau(min_lr=0.00001,patience=3)

#训练自编码器
SAE.fit(x,x,epochs=50,batch_size=30,shuffle=True,callbacks=[checkpoint])