# -*- coding: utf-8 -*-
import all_data
import numpy as np
import denoise_WT
from keras.models import load_model
from sklearn import metrics

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

def predict_f(x,model="LSTM"):
    #如果是LSTM_WT模型和LSTM_WT_SAE模型，进行WT处理
    if model=="LSTM_WT" or model=="LSTM_WT_SAE":
        x=denoise_WT.deniose_dataset(x)
    #如果是LSTM_SAE模型和LSTM_WT_SAE模型，进行SAE处理
    if model=="LSTM_SAE" or model=="LSTM_WT_SAE":
        autoencoder=load_model("model/autoencoder.hf5")
        x=autoencoder.predict(x)
    #如果是LSTM模型，直接预测
    if model=="LSTM":
        m=load_model("model/lstm.hf5")
    elif model=="LSTM_WT":
        m=load_model("model/LSTM_WT.hf5")
    elif model=="LSTM_SAE":
        m=load_model("model/LSTM_SAE.hf5")
    elif model=="LSTM_WT_SAE":
        m=load_model("model/LSTM_WT_SAE.hf5")
    elif model=="LSTM_encoder_decoder":
        m=load_model("model/LSTM_encoder_decoder.hf5")
    elif model=="LSTM_WT_encoder_decoder":
        m=load_model("model/LSTM_WT_encoder_decoder.hf5")
    predict=m.predict(x)
    return (np.asarray(predict))

#导入数据
x,y=all_data.get_testset()

#进行数据归一化处理
x=(x-x.mean())/x.std()
y=np.reshape(y,(y.shape[0],))

#进行预测
predict=predict_f(x,model='LSTM_WT')

print ("MAE:",metrics.mean_absolute_error(y,predict))
print ("RMSE:",np.sqrt(metrics.mean_squared_error(y,predict)))
print ("score:",GetScore(predict,y))