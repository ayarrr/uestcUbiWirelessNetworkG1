# -*- coding: utf-8 -*-
import pywt
import numpy as np
from statsmodels.robust import mad

def denoise_WT( x, wavelet="db4", level=1):
    # 进行小波分解，计算小波系数（父小波、母小波）
    coeff = pywt.wavedec( x, wavelet,mode="per")
    #计算阈值
    sigma = mad(coeff[-level])#噪声强度  计算最后一个父小波的噪声强度
    #sigma=np.median(np.abs(coeff[-level] - np.median(coeff[-level])))/0.6745
    uthresh = sigma*np.sqrt(2*np.log(len(x)))#使用缺省阈值确定模型
    
    #作用阈值过程  使用软阈值对父小波逐个进行过滤
    coeff[1:] = ( pywt.threshold( i, value=uthresh, mode="soft" ) for i in coeff[1:] )
    # 使用阈值系数重构小波
    y = pywt.waverec( coeff, wavelet, mode="per" )
    return (np.asarray(y[:len(x)]))

#进行数据集的降噪处理
def deniose_dataset(x_train):
    x_train_restrc=[]
    for one_data in x_train:
        x_one_line=np.asarray([denoise_WT(x) for x in one_data])
        x_train_restrc.append(x_one_line)
    return (np.asarray(x_train_restrc))
