# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

def get_one(fund_return,fund_benchmark_return,index_return,correlation,setname="train"):
    time_list=correlation.columns
    fund_length=fund_return.shape[0]
    columns=['fund_return1','fund_return2','fund_benchmark_return1','fund_benchmark_return2']
    columns=np.asarray(columns)
    index_return_columns=index_return['Unnamed: 0']
    columns=np.concatenate((columns,index_return_columns),axis=0)
    columns=np.append(columns,'relation')
    
    data_list=[]
    for n in range(len(time_list)):
        time=time_list[n]
        if time != 'Unnamed: 0':
            unit_data=[]
            #在fund_return和fund_benchmark_return之间产生基金对
            counter=0
            for i in range(fund_length-1):
                for j in range(i+1,fund_length):
                    unit_one_line=[]#取每一行数据
                    unit_one_line.append(fund_return.loc[i,fund_return.columns[n]])
                    unit_one_line.append(fund_return.loc[j,fund_return.columns[n]])
                    unit_one_line.append(fund_benchmark_return.loc[i,fund_return.columns[n]])
                    unit_one_line.append(fund_benchmark_return.loc[j,fund_return.columns[n]])
                    unit_one_line=np.asarray(unit_one_line)
                    unit_one_line=np.concatenate((unit_one_line,np.asarray(index_return[index_return.columns[n]])),axis=0)
                    unit_one_line=np.append(unit_one_line,correlation.loc[counter,time])
                    #print (unit_one_line.shape)
                    unit_data.append(unit_one_line)
                    counter=counter+1   #对已经生成的行进行计数
            unit_data=np.asarray(unit_data)
            #unit_data=np.concatenate((unit_data,),axis=0)
            #print (unit_data.shape)
            #print (unit_data)
            unit=pd.DataFrame(unit_data,columns=columns)
            data_list.append(unit)
    print (len(data_list))
    data_list=pd.concat(data_list,axis=0)
    print (data_list.shape)
    data_list.to_csv("processing1/"+setname+".csv",encoding='utf8',mode='w',index=False)

#训练集
'''
fund_return=pd.read_csv("data/train_fund_return.csv",encoding="utf8")
fund_benchmark_return=pd.read_csv("data/train_fund_benchmark_return.csv",encoding="utf8")
index_return=pd.read_csv("data/train_index_return.csv",encoding="gbk")
correlation=pd.read_csv("data/train_correlation.csv",encoding="utf8")
get_one(fund_return,fund_benchmark_return,index_return,correlation)
'''

#测试集
fund_return=pd.read_csv("data/test_fund_return.csv",encoding="utf8")
fund_benchmark_return=pd.read_csv("data/test_fund_benchmark_return.csv",encoding="utf8")
index_return=pd.read_csv("data/test_index_return.csv",encoding="gbk")
correlation=pd.read_csv("data/test_correlation.csv",encoding="utf8")
get_one(fund_return,fund_benchmark_return,index_return,correlation,setname='test')