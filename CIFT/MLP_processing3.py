# -*- coding: utf-8 -*-
from keras import layers
from keras.models import Model
from keras.optimizers import Adam
from keras import callbacks
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
import numpy as np
import processing3
x,y=processing3.get_trainset()

#数据归一化
x=scale(np.asarray(x))
y=np.asarray(y)
y=np.reshape(y,(y.shape[0],))

#数据集划分
x_train,x_val,y_train,y_val=train_test_split(x,y,test_size=0.3)
print ("训练集大小,x_train:",x_train.shape,"  y_train:",y_train.shape)
print ("测试集大小,x_test:",x_val.shape,"  y_test:",y_val.shape)

#使用两层，每层128个神经元
inputer=layers.Input(shape=(x_train.shape[1],))
dense1=layers.Dense(128)(inputer)
dense2=layers.Dense(128)(dense1)
dropout=layers.Dropout(0.2)(dense2)
outputer=layers.Dense(1)(dropout)
model=Model(inputs=inputer,outputs=outputer)

#编译模型
model.compile(optimizer=Adam(lr=0.001),loss='mae',metrics=['accuracy'])

#保存模型
checkpoint=callbacks.ModelCheckpoint(filepath="model/MLP_processing3.hf5",verbose=1)
#自动调整学习率
lr=callbacks.ReduceLROnPlateau(min_lr=0.00001,patience=3)

#开始训练
history=model.fit(x_train,y_train,epochs=50,batch_size=30,validation_data=(x_val,y_val),callbacks=[checkpoint,lr])

val_acc=np.asarry(history.history['val_acc'])
print (val_acc)
np.save("processing1/MLP_train.npy",val_acc)