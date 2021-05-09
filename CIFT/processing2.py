# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import talib 
#获取基金数据的金融学指标
def get_finance_indicator(unit_one_line,time_series):
    unit_one_line.append(time_series[0])#初始收益率
    unit_one_line.append(time_series[len(time_series)-1])#最终收益率
    unit_one_line.append(np.max(time_series))#最高收益率
    unit_one_line.append(np.min(time_series))#最低收益率
    #取出MA5,MA10,MA20,MA30,MA60的首元素
    MA5=talib.MA(time_series.astype(float),timeperiod=5)
    MA10=talib.MA(time_series.astype(float),timeperiod=10)
    MA20=talib.MA(time_series.astype(float),timeperiod=20)
    MA30=talib.MA(time_series.astype(float),timeperiod=30)
    MA60=talib.MA(time_series.astype(float),timeperiod=60)
    unit_one_line.append(MA5[4])
    unit_one_line.append(MA10[9])
    unit_one_line.append(MA20[19])
    unit_one_line.append(MA30[29])
    unit_one_line.append(MA60[59])
    #取出EMA20的首元素
    unit_one_line.append(talib.EMA(time_series.astype(float),timeperiod=20)[19])
    #取出MACD的首元素
    unit_one_line.append(talib.MACD(time_series.astype(float),fastperiod=12,slowperiod=26,signalperiod=5)[0][29])
    return unit_one_line

#获取重要市场的收益率统计指标
def get_indexreturn_indicator(unit_one_line,timeseries):
    unit_one_line.append(timeseries[0])
    unit_one_line.append(timeseries[len(timeseries)-1])
    unit_one_line.append(np.max(timeseries))
    unit_one_line.append(np.min(timeseries))
    return unit_one_line

def get_one(fund_return,fund_benchmark_return,index_return,correlation,setname='train'):
    time_list=correlation.columns
    fund_length=fund_return.shape[0]
    columns=['fund_return1','fund_return2','fund_benchmark_return1','fund_benchmark_return2']
    columns=np.asarray(columns)
    index_return_columns=index_return['Unnamed: 0']
    columns=np.concatenate((columns,index_return_columns),axis=0)
    columns=np.append(columns,'relation')
    
    data_list=[]
    target=[]
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
                    time_series11=np.asarray(fund_return.loc[i,fund_time])
                    time_series21=np.asarray(fund_return.loc[j,fund_time])
                    time_series12=np.asarray(fund_benchmark_return.loc[i,fund_time])
                    time_series22=np.asarray(fund_benchmark_return.loc[j,fund_time])
                    #读取金融学指标
                    unit_one_line=get_finance_indicator(unit_one_line,time_series11)
                    unit_one_line=get_finance_indicator(unit_one_line,time_series21)
                    unit_one_line=get_finance_indicator(unit_one_line,time_series12)
                    unit_one_line=get_finance_indicator(unit_one_line,time_series22)
                    #读取重要市场的收益率统计指标
                    for timeseries in np.asarray(index_return[index_time]):
                        unit_one_line=get_indexreturn_indicator(unit_one_line,timeseries)
                    unit_one_line=np.asarray(unit_one_line)
                    
                    #取出该时间步的标签
                    target.append(correlation.loc[counter,time])
                    
                    unit_data.append(unit_one_line.T)
                    counter=counter+1   #对已经生成的行进行计数
            unit_data=np.asarray(unit_data)
            data_list.append(unit_data)
    data_list=np.concatenate(data_list,axis=0)
    print ("数据产生完毕！")
    print ("特征形状为：",data_list.shape)
    print ("标签形状为：",len(target))
    return (data_list,target)

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
