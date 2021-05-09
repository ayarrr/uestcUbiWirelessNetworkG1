# -*- coding: utf-8 -*-
import all_data_da_RNN
import numpy as np
from keras.models import load_model,model_from_json
from sklearn import metrics
from DA_RNN import Encoder,Decoder

def GetScore(ypred,y_test):
    mae=0
    tmape=0
    n=len(y_test)
    for i in range(n):
        p=ypred[i]
        t=y_test[i]
        mae=mae+abs(p-t)
        tmape=tmape+abs((p-t)/(1.5-t))
    tmape=tmape/n
    return tmape
'''
    mae=mae/n
    score=(2/(2+mae+tmape))**2
'''

#导入数据
x,y_prev,y=all_data_da_RNN.get_testset()

#进行数据归一化处理
x=(x-x.mean())/x.std()
y=np.reshape(y,(y.shape[0],1))

#进行预测
input_size=x.shape[2]
hidden_encoder=128
hidden_decoder=128
T=x.shape[1]+1
#加载模型数据和weights
"""
model = model_from_json(open('DA_RNN.json').read())  
model.load_weights('DA_RNN_weight.h5')  
"""
model=load_model("model/DA_RNN.h5",custom_objects={"Encoder":Encoder,"Decoder":Decoder})
print (model)
predict=model.predict([x,y_prev])

print ("MAE:",metrics.mean_absolute_error(y,predict))
print ("RMSE:",np.sqrt(metrics.mean_squared_error(y,predict)))
print ("score:",GetScore(predict,y))
