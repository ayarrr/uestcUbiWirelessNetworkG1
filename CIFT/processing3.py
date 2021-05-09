# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import fastdtw

def get_indexreturn_indicator(unit_one_line,timeseries):
    unit_one_line.append(timeseries[0])
    unit_one_line.append(timeseries[len(timeseries)-1])
    unit_one_line.append(np.max(timeseries))
    unit_one_line.append(np.min(timeseries))
    return unit_one_line

def get_one(fund_return,fund_benchmark_return,index_return,correlation,setname='train'):
    time_list=correlation.columns
    fund_length=fund_return.shape[0]
    columns=['fund_return_correlation','fund_benchmark_return_correlation']
    columns=np.asarray(columns)
    index_return_columns=index_return['Unnamed: 0']
    columns=np.concatenate((columns,index_return_columns),axis=0)
    columns=np.append(columns,'relation')
    
    data_list=[]
    target=[]
    #master_counter=0#测试
    #测试只取30条数据  399,400  
    for n in range(len(time_list)):
        time=time_list[n]
        if time != 'Unnamed: 0':
            #取出61个时间步
            fund_time=[fund_return.columns[x] for x in range(n,n+61)]
            index_time=[index_return.columns[x] for x in range(n,n+61)]
            unit_data=[]
            #在fund_return和fund_benchmark_return之间产生基金对
            counter=0
            for i in range(fund_length-1):
                for j in range(i+1,fund_length):
                    unit_one_line=[]#取每一行数据
                    #取出61个时间步的数据
                    fund_return11=np.asarray(fund_return.loc[i,fund_time])
                    fund_return21=np.asarray(fund_return.loc[j,fund_time])
                    fund_benchmark_return12=np.asarray(fund_benchmark_return.loc[i,fund_time])
                    fund_benchmark_return22=np.asarray(fund_benchmark_return.loc[j,fund_time])
                    fund_return_correlation,show1=fastdtw.dtw(fund_return11,fund_return21)
                    fund_benchmark_return_correlation,show2=fastdtw.dtw(fund_benchmark_return12,fund_benchmark_return22)
                    unit_one_line.append(fund_return_correlation)
                    unit_one_line.append(fund_benchmark_return_correlation)
                    
                    #读取重要市场的收益率统计指标
                    for timeseries in np.asarray(index_return[index_time]):
                        unit_one_line=get_indexreturn_indicator(unit_one_line,timeseries)
                    #取出该时间步的标签
                    target.append(correlation.loc[counter,time])
                    
                    unit_one_line=np.asarray(unit_one_line)
                    unit_data.append(unit_one_line.T)
                    counter=counter+1   #对已经生成的行进行计数
                    #master_counter+=1#测试
            unit_data=np.asarray(unit_data)
            data_list.append(unit_data)
            #if master_counter>=30:#测试
            #    break;
        
    data_list=np.concatenate(data_list,axis=0)
    #data_list=np.asarray(data_list)
    print ("数据产生完毕！")
    print ("特征形状为：",data_list.shape)
    print ("标签形状为：",len(target))
    return (data_list,np.asarray(target))
#训练集
def get_trainset():
    fund_return=pd.read_csv("data/train_fund_return.csv",encoding="utf8")
    fund_return2=pd.read_csv("data/test_fund_return.csv",encoding="utf8")
    fund_return2=fund_return2.loc[:,fund_return2.columns[1:61]]
    fund_return=pd.concat([fund_return,fund_return2],axis=1)
    fund_benchmark_return=pd.read_csv("data/train_fund_benchmark_return.csv",encoding="utf8")
    fund_benchmark_return2=pd.read_csv("data/test_fund_benchmark_return.csv",encoding="utf8")
    fund_benchmark_return2=fund_benchmark_return2.loc[:,fund_benchmark_return2.columns[1:61]]
    fund_benchmark_return=pd.concat([fund_benchmark_return,fund_benchmark_return2],axis=1)
    index_return=pd.read_csv("data/train_index_return.csv",encoding="gbk")
    index_return2=pd.read_csv("data/test_index_return.csv",encoding="gbk")
    index_return2=index_return2.loc[:,index_return2.columns[1:61]]
    index_return=pd.concat([index_return,index_return2],axis=1)
    correlation=pd.read_csv("data/train_correlation.csv",encoding="utf8")
    data_list,target=get_one(fund_return,fund_benchmark_return,index_return,correlation)
    return (data_list,target)

#测试集
def get_testset():
    fund_return=pd.read_csv("data/test_fund_return.csv",encoding="utf8")
    fund_benchmark_return=pd.read_csv("data/test_fund_benchmark_return.csv",encoding="utf8")
    index_return=pd.read_csv("data/test_index_return.csv",encoding="gbk")
    correlation=pd.read_csv("data/test_correlation.csv",encoding="utf8")
    data_list,target=get_one(fund_return,fund_benchmark_return,index_return,correlation)
    return (data_list,target)

