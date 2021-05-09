# -*- coding: utf-8 -*-
import lightgbm as lgb
from keras.models import load_model
from sklearn.preprocessing import scale
from sklearn import metrics
import numpy as np
import processing2
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

def predict_f(x,model="lgb"):
    if model== 'lgb':
        m=lgb.Booster(model_file="model/lgb_preprocessing2.txt")
    elif model=='mlp':
        m=load_model("model/MLP_processing2.hf5")
    predict=m.predict(x)
    predict=np.asarray(predict)
    return predict

x,y=processing2.get_testset()

#数据归一化
x=scale(np.asarray(x))
y=np.asarray(y)
y=np.reshape(y,(y.shape[0],))

predict=predict_f(x)

print ("MAE:",metrics.mean_absolute_error(y,predict))
print ("RMSE:",np.sqrt(metrics.mean_squared_error(y,predict)))
print ("score:",GetScore(predict,y))
