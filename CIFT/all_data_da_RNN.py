# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
def get_one(fund_return,fund_benchmark_return,index_return,correlation,setname='train'):
    time_list=correlation.columns[:-61]   
    fund_length=fund_return.shape[0]
    columns=['fund_return1','fund_return2','fund_benchmark_return1','fund_benchmark_return2']
    columns=np.asarray(columns)
    index_return_columns=index_return['Unnamed: 0']
    columns=np.concatenate((columns,index_return_columns),axis=0)
    columns=np.append(columns,'relation')
    
    data_list=[]
    target_prev=[]
    target=[]
    #master_counter=0#测试s
    #测试只取30条数据  train:395,400 test:len(time_list)-5,len(time_list) real:len(time_list)
    """
    if setname=="train":
        ranges=range(398,400)
    elif setname=="test":
        ranges=range(len(time_list)-1,len(time_list))
    """
    ranges=range(len(time_list))
    for n in ranges:#ranges
        time=time_list[n]
        if time != 'Unnamed: 0':
            #取出61个时间步
            fund_time=[fund_return.columns[x] for x in range(n,n+61)]
            index_time=[index_return.columns[x] for x in range(n,n+61)]
            rela_time=[correlation.columns[x] for x in range(n,n+61)]
            unit_data=[]
            unit_target_prev=[]
            #在fund_return和fund_benchmark_return之间产生基金对
            counter=0
            for i in range(fund_length-1):#
                for j in range(i+1,fund_length):
                    unit_one_line=[]#取每一行数据
                    #取出61个时间步的数据
                    unit_one_line.append(np.asarray(fund_return.loc[i,fund_time]))
                    unit_one_line.append(np.asarray(fund_return.loc[j,fund_time]))
                    unit_one_line.append(np.asarray(fund_benchmark_return.loc[i,fund_time]))
                    unit_one_line.append(np.asarray(fund_benchmark_return.loc[j,fund_time]))
                    unit_one_line=np.asarray(unit_one_line)
                    unit_one_line=np.concatenate((unit_one_line,np.asarray(index_return[index_time])),axis=0)
                    
                    #取出该时间步的标签
                    target.append(correlation.loc[counter,time])
                    
                    #取出前61个时间步的标签
                    unit_target_prev.append(np.asarray(correlation.loc[counter,rela_time]))
                    
                    unit_data.append(unit_one_line.T)
                    counter=counter+1   #对已经生成的行进行计数
                    #master_counter+=1#测试
            unit_data=np.asarray(unit_data)
            data_list.append(unit_data)
            unit_target_prev=np.asarray(unit_target_prev)
            target_prev.append(unit_target_prev)
            
            #if master_counter>=30:#测试
            #    break;
        
    data_list=np.concatenate(data_list,axis=0)
    target_prev=np.concatenate(target_prev,axis=0)
    #data_list=np.asarray(data_list)
    print ("数据产生完毕！")
    print ("特征形状为：",data_list.shape)
    print ("之前时间步的标记形状为：",target_prev.shape)
    print ("标签形状为：",len(target))
    return (data_list,target_prev,np.asarray(target))
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
    correlation2=pd.read_csv("data/test_correlation.csv",encoding="utf8")
    correlation2=correlation2.loc[:,correlation2.columns[1:61]]
    correlation=pd.concat([correlation,correlation2],axis=1)
    data_list,target_prev,target=get_one(fund_return,fund_benchmark_return,index_return,correlation)
    return (data_list,target_prev,target)

#测试集
def get_testset():
    fund_return=pd.read_csv("data/test_fund_return.csv",encoding="utf8")
    fund_benchmark_return=pd.read_csv("data/test_fund_benchmark_return.csv",encoding="utf8")
    index_return=pd.read_csv("data/test_index_return.csv",encoding="gbk")
    correlation=pd.read_csv("data/test_correlation.csv",encoding="utf8")
    data_list,target_prev,target=get_one(fund_return,fund_benchmark_return,index_return,correlation,setname="test")
    return (data_list,target_prev,target)

#训练集，验证集，测试集划分
def train_val_split(X,y_prev,y,T):
    '''
    参数：
        X(np.ndarray):特征
        y(np.ndarray):实际值
        is_Val(boolean):是否需要一个验证集，默认需要
    返回值：
        X_train(np.ndarray):训练特征
        y_train(np.ndarray):训练实际值
        X_test(np.ndarray):测试特征
        y_test(np.ndarray):测试实际值
        X_val(np.ndarray):验证特征
        y_val(np.ndarray):验证实际值
    '''
    
    train=int(len(X)*0.7)
    
    #训练集
    X_train=X[:train]
    y_train=y[:train]
    y_prev_train=y_prev[:train]
    print ("训练集形状为：",X_train.shape," ",y_train.shape," ",y_prev_train.shape)
    
    #验证集
    X_val=X[train:]
    y_val=y[train:]
    y_prev_val=y_prev[train:]
    print ("验证集形状为：",X_val.shape," ",y_val.shape," ",y_prev_val.shape)
    
    return (X_train,y_train,y_prev_train,X_val,y_val,y_prev_val)
#get_trainset()