# -*- coding: utf-8 -*-
from keras import layers
from keras.models import Model
import all_data
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
import numpy as np
from keras.optimizers import Adam
from keras import callbacks

x,y=all_data.get_trainset()

#数据归一化
#x=scale(x)
x=(x-x.mean())/x.std()
print (x.shape)
y=np.reshape(y,(y.shape[0]))

#数据集划分
x_train,x_val,y_train,y_val=train_test_split(x,y,test_size=0.3)

#构建模型
inputer=layers.Input(shape=(x_train.shape[1],x_train.shape[2]))
lstm1=layers.LSTM(128,return_sequences=True)(inputer)
lstm2=layers.LSTM(128)(lstm1)
dropout=layers.Dropout(0.2)(lstm2)
outputer=layers.Dense(1)(dropout)
model=Model(inputs=inputer,outputs=outputer)

#模型编译
model.compile(optimizer=Adam(lr=0.001),loss='mae',metrics=['accuracy'])

#保存模型
checkpoint=callbacks.ModelCheckpoint(filepath="model/lstm.hf5",verbose=1)
#自动调整参数
lr=callbacks.ReduceLROnPlateau(min_lr=0.00001,patience=3)

#开始训练
history=model.fit(x_train,y_train,epochs=50,batch_size=30,validation_data=(x_val,y_val),callbacks=[checkpoint,lr])

val_acc=np.asarry(history.history['val_acc'])
np.save("processing1/LSTM_train.npy",val_acc)