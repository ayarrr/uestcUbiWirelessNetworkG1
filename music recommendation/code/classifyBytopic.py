# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 21:40:50 2018

@author: Administrator
"""
import numpy as np
import pandas as pd

import os
#得到训练集所有的pid
def getTop3(raw,n):
    t=raw.nlargest(n).index.values
    d={}
    for i in range(n):
        d['t'+str(i+1)]=t[i]
    return pd.Series(d)
   
def PidUnderTopic(raw,n):
    maintittle=['t'+str(i+1) for i in range(n) ]
    main=list(raw[maintittle].values)
#    del raw['t1','t2','t3']
    raw=raw.drop(maintittle)
    raw[list(set(raw.index)-set(main))]=-1
    return raw
def spiltTrainAndTest(n):
    takedata=pd.read_csv("doc_topic.csv",index_col='pid')
    print(takedata.info)
    
#求主题
# playlistTopic["topic"]=playlistTopic.T.apply(lambda x:np.where(x==np.max(x))[0][0])
    playlistTopic=takedata.apply(lambda x:getTop3(x,n),axis=1)
    playlistTopic1=pd.concat([takedata,playlistTopic],axis=1)
    playlistTopic1.to_csv("DistributeAndMainTopic.csv")#获得主题分布及3个主要主题
    #在不是主要主题的位置记为-1
    s=playlistTopic1.apply(lambda x:PidUnderTopic(x,n),axis=1)
    return s
#主要主题数由n决定
s=spiltTrainAndTest(1)
s.to_csv("Pid_Topic.csv")#记录各个主题下有哪些pid


    







#
#
##    
##
##traintTopic,testSetTopic=spiltTrainAndTest()
##testSetTopic.to_csv("testSetTopic1.csv",index=False)
##traintTopic.to_csv("traintTopic1.csv",index=False)
#traintTopic=pd.read_csv("traintTopic1.csv")
#testSetTopic=  pd.read_csv("testSetTopic1.csv")
##获得训练集中某主题的pid
#def f(row):
#    x=row["topic"]
#    pid=row["pid"]
#    return (pid,list(traintTopic[traintTopic.topic==x]["pid"]))
##对于各个测试集寻找其同主题训练集pid,存成元祖（pid,sameTopicPidList）
#myclassifydict=list(testSetTopic.apply(f,axis=1))
#title=["album_name","album_uri","artist_name","artist_uri","duration_ms","pos","track_name","track_uri","pid","pname","num_followers"]
#ALLTricks=pd.read_csv("../needConcernTrickWithPname1.csv", encoding='ISO-8859-1')[title].sort_values(by="num_followers",ascending=False)
##title=["album_name","album_uri","artist_name","artist_uri","duration_ms","pos","track_name","track_uri","pid","pname"]
##ALLTricks=pd.read_csv("../needConcernTrickWithPname.csv", encoding='ISO-8859-1')[title]
#def trickapply(line):
#    playlistId=line[0]
#    typelist=line[1]
#    currenttrick=ALLTricks[ALLTricks["pid"].isin(typelist)]["artist_uri"]
#    return len(currenttrick.tolist())
##result=map(trickapply,myclassifydict)
#lenlist=[]
#for line in myclassifydict:
#    trp=trickapply(line)
#    lenlist.append(trp)
#print(max(lenlist),min(lenlist))

    
    
