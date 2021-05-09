# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 21:03:17 2018

@author: 10097
"""
import pandas as pd
#只提取数据中的50个基金
def get_pred(file):
    name=file.split("_")
    return (name[0])
#fund_return提取
def get_fund_return(file,size=50):
    data=pd.read_csv("C:/Users/10097/Desktop/jijin/dataset/"+file,encoding="utf8")
    data=data.loc[:size-1,]
    pred=get_pred(file)
    print (data.shape)
    data.to_csv("data/"+pred+"_fund_return.csv",encoding="utf8",mode="w",index=False)
    
#fund_benchmark_return提取
def get_fund_benchmark_return(file,size=50):
    data=pd.read_csv("C:/Users/10097/Desktop/jijin/dataset/"+file,encoding="utf8")
    data=data.loc[:size-1,:]
    pred=get_pred(file)
    print (data.shape)
    data.to_csv("data/"+pred+"_fund_benchmark_return.csv",encoding="utf8",mode="w",index=False)
#correlation提取
def get_correlation(file,size=50):
    index_range=range(1,size+1)
    data=pd.read_csv("C:/Users/10097/Desktop/jijin/dataset/"+file)
    show=[]
    for i in data.index:
        fund_name=data.loc[i,"Unnamed: 0"]
        fund_name=fund_name.replace("Fund ","")
        number=fund_name.split("-")
        number=[int(x) for x in number]
        if (number[0] in index_range) and (number[1] in index_range):
            show.append(data.loc[i,:])
    split_data=pd.concat(show,axis=1)
    split_data=split_data.T
    print (split_data.shape)
    split_data.to_csv("data/"+get_pred(file)+"_correlation.csv",encoding="utf8",mode="w",index=False)

get_correlation("train_correlation.csv",size=25)
get_correlation("test_correlation.csv",size=25)
get_fund_return("train_fund_return.csv",size=25)
get_fund_return("test_fund_return.csv",size=25)
get_fund_benchmark_return("train_fund_benchmark_return.csv",size=25)
get_fund_benchmark_return("test_fund_benchmark_return.csv",size=25)

            