# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 10:16:51 2019

@author: Administrator
"""
import pandas as pd
import numpy as np
import pickle
import random
class vocab:
    vd=pickle.load(open('../middeldata/itemEmb.pkl','rb'))
    wordlist=list(vd.keys())
    def dictValue2Array(s=vd):
        r = []
        for (k, v) in s.items():
            r.append(v)
        return np.array(r)

def get_popularity(df):
    """Get number of clicks that each item received in the df."""

    mask = df["action_type"] == "clickout item"
    df_clicks = df[mask]
    df_item_clicks = (
        df_clicks
        .groupby("reference")
        .size()
        .reset_index(name="n_clicks")
        .transform(lambda x: x.astype(int))
    )

    return df_item_clicks
def id2index(newid):
    wordlist=vocab.wordlist
    if newid not in wordlist:
        return random.randrange(0,len(wordlist))
    return wordlist.index(int(newid))
# 标准化
def ss(data):
#    data = [58, 96, 55, 75, 90, 60, 233, 104, 150, 145, 328, 207, 150, 181, 135, 99, 495, 170, 118, 259, 73, 169, 87, 485, 171]
    me=np.median(data)
    sd=np.var(data)
    hh=list(map(lambda x:(x-me)/sd,data))
#    print("===============================")
#    print("标准化，返回如下：")
#    print(hh)
    
    return hh,me
def action_map(action_type):
    action_type=str(action_type)
    if "interaction" in action_type:
        return 0
    elif "search" in action_type:
        return 1
    elif "clickout" in action_type:
        return 2
def sessPro(d,popdf,candidates=25,maxsessionlen=151):
    """
    d为各个session的dataframe
    popdf为按流行度排好的item(有两列：reference  n_clicks)
    该方法的作用是：替换itemid,补齐25个候补，标准化价格,action_type映射（interact:0,search:1,click:2）
    """
#    print(type(d))
#    sessionid=list(d["session_id"])[0]
    if len(d)>maxsessionlen:
        d=d[-maxsessionlen:]
    userid=list(d["user_id"])[0]
    itemlist=list(map(id2index,list(d["reference"])))#找到每个id对应的index
    
    impressions=list(d["impressions"])[-1]
    action_seq=list(map(action_map,list(d["action_type"])))[:-1]
    label=itemlist[-1]
    itemlist=itemlist[:-1]
    sequencelen=len(itemlist)
    if len(itemlist)>0:
        last_item=itemlist[-1]
    else:
        last_item=0
    if pd.isnull(impressions):
        return None
    elif type(impressions)==float and len(impressions)==1:
        impressions=list(map(id2index,[impressions]))
        imlen=len(impressions)
        impressions.extend(list(map(id2index,list(popdf.iloc[:(candidates-imlen)]["reference"]))))
        prices=[0]*candidates#返回标准化的价格
    else:
        impressions=impressions.split('|')
        imlen=len(impressions)
        prices,me=ss(list(map(float,list(d["prices"])[-1].split('|'))))#返回标准化的价格
        if imlen<candidates:#补充些流行的
            impressions.extend(list(map(str,list(popdf.iloc[:(candidates-imlen)]["reference"]))))
            prices.extend([me]*(candidates-imlen))
            impressions=list(map(id2index,impressions))
        else:#多的删除
            ts=impressions[:candidates]
            impressions=list(map(id2index,ts))
            prices=prices[:candidates]
    print(sequencelen,imlen,len(impressions))
    return (userid,itemlist,impressions,prices,sequencelen,action_seq,last_item,label)
def loadSequence(tainpath):#得到各个session的序列
   
#    tainpath="../processed/mytrain.csv"
    train=pd.read_csv(tainpath, sep='\t')[["user_id","session_id","action_type","reference","impressions","prices"]]
#    populardf=get_popularity(train).sort_values(by="n_clicks" , ascending=False)
    populardf=pd.read_csv('../processed/popular.csv', sep='\t')
    d=dict(train.groupby("session_id").apply(lambda x:sessPro(x,populardf)))#得到list，每个元素是session
    d=dict(filter(lambda x:x[1]!=None,d.items()))
    
    userEm=UserPro(train[train["action_type"]=="clickout item"][["user_id","reference"]],vocab.vd)
    return d,userEm.getUserEmb()
#m=loadSequence()
def saveUserEmbed(tainpath,label,ratio):
    train=pd.read_csv(tainpath, sep='\t')[["user_id","session_id","action_type","reference","impressions","prices"]]
    userEMdict=UserPro(train[train["action_type"]=="clickout item"][["user_id","reference"]],vocab.vd).getUserEmb()
    pickle.dump(userEMdict,open("../processed/userEMdict_"+label+str(ratio)+".pkl", 'wb'))
    return
    

def savecsv(path,label,ratio,order):
    m,userEMdict=loadSequence(path)
    dfm=pd.DataFrame(m).T
    dfm.columns=["userid","itemlist","impressions","prices","sequencelen","action_type","last_item","label"]
    dfm["sessionid"]=list(m.keys())
    dfm.to_csv("../processed/sessionSequences_"+label+str(ratio)+str(order)+".csv",index=False)
    pickle.dump(userEMdict,open("../processed/userEMdict_"+label+str(ratio)+".pkl", 'wb'))
#    dfm.groupby("sequencelen")
    return dfm,userEMdict



def savetrain(datapath,ratio):#datapath原始数据的位置，使用比例
    try:
        filter_data=pd.read_csv("../processed/withOnetimeVistor_train.csv", sep='\t')
    except IOError:
        print("get item...")
        orginaltrain=pd.read_csv(datapath)
        train_bool=orginaltrain.action_type.str.contains('item')
        filter_data=orginaltrain[train_bool]
        filter_data.to_csv( '../processed/withOnetimeVistor_train.csv', sep='\t', index=False)  
    session_lengths=filter_data.groupby('session_id').size()
    data=filter_data[filter_data.session_id.isin(list(session_lengths[session_lengths>1].index))]#目标session,这样会扔掉很多游客的行为
    allSessions=list(set((data["session_id"].values)))
    print("总有效session数为：",len(allSessions),"总有效action数为：",data.shape[0],"本次使用的session比例：",ratio)
    allSessions=allSessions[-int(ratio*len(allSessions)):]
    data=data[data.session_id.isin(allSessions)]
    
    tmean = data.timestamp.max()
    session_max_times = data.groupby('session_id').timestamp.max()
    session_train = session_max_times[session_max_times < tmean-20000].index 
    session_test = session_max_times[session_max_times >=  tmean-20000].index
    asize=list(data.groupby(["session_id"]).size())
    print("train的session数",len(session_train),"test的session数",len(session_test),"actions:",data.shape,"AVGlen:",float(sum(asize))/len(asize))
    mytrain=data[data.session_id.isin(list(session_train))]#1330477
#    print(mytrain.shape)#>>(11394893, 12)
    mytrain.to_csv( '../processed/mytrain_'+str(ratio)+'.csv', sep='\t', index=False)
    mytest=data[data.session_id.isin(list(session_test))]#283091
#    print(mytest.shape)#>>(2626125, 12)
    mytest.to_csv( '../processed/mytest_'+str(ratio)+'.csv', sep='\t', index=False)
    
#    print(data.timestamp.max()-data.timestamp.min())#518366
##    data.to_csv( '../processed/Alltartget_train.csv', sep='\t', index=False)
#    tmax = data.timestamp.max()#最大为1541548799
#    session_max_times = data.groupby('session_id').timestamp.max()
#    session_train = session_max_times[session_max_times < tmax-90000].index #按时间切分数据集（90000以前的所有数据作为训练集后半部分作为测试集）
#    session_test = session_max_times[session_max_times >= tmax-90000].index
##    print("train的session数",len(session_train),"test的session数",len(session_test))
#    #>>train的session数 497700 test的session数 112607
#    mytrain=data[data.session_id.isin(list(session_train))]#1330477
##    print(mytrain.shape)#>>(11394893, 12)
#    mytrain.to_csv( '../processed/mytrain.csv', sep='\t', index=False)
#    mytest=data[data.session_id.isin(list(session_test))]#283091
##    print(mytest.shape)#>>(2626125, 12)
#    mytest.to_csv( '../processed/mytest.csv', sep='\t', index=False)
    return

class UserPro(object):
    def __init__(self,userSutuation,pre_Emb=None):
        #预训练的itemEmbed
        self.pre_Emb=pre_Emb
        self.dim=157
        #DataFrame 用户点击iteam的情况共两列（"user_id","reference"）action_type="click_out"
        self.userSutuation=userSutuation         
    def getUserEmb(self):
        
        '''
        获取所有user的emb,userEmb=mean(clickIteam)
        return 各个user的emb,Uembs为字典类型
        '''
        Uembs={}
        userIds=list(set(self.userSutuation["user_id"]))
        ms=self.userSutuation.groupby('user_id')
        for uid in userIds:
            curentItemEm=[]
            for i in set(dict(list(ms))[uid]["reference"]):
                if i in self.pre_Emb.keys():
                    curentItemEm.append(self.pre_Emb[i])
            if len(curentItemEm):
                # 存在值即为真
                ItemEmbs=np.array(curentItemEm)#得到user点击的所有item
                Uemb=np.mean(ItemEmbs, axis=0)
            else:
               # curentItemEm是空的
                Uemb=np.random.random((1,self.dim))
            Uembs[uid]=np.reshape(Uemb,(1,self.dim))
        return Uembs

##savetrain("../data/train.csv")
#savecsv("../processed/mytrain_"+str(1/8)+"2.csv","for_train",1/8,2)
#    savecsv("../processed/mytest_"+str(1/8)+".csv","for_test",1/8)
#savetrain("../data/train.csv",1/8)


saveUserEmbed("../processed/mytrain_"+str(1/8)+".csv","for_train",1/8)
saveUserEmbed("../processed/mytest_"+str(1/8)+".csv","for_test",1/8)
#dfm=pd.read_csv("../processed/sessionSequences_for_train"+str(1/64)+".csv")