# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 15:29:25 2019

@author: Administrator
"""
import pickle
import numpy as np
import pandas as pd
from CCF1 import CF_svd,CF_knearest,M_F,BPR
def AP(ranked_list, ground_truth):
    """Compute the average precision (AP) of a list of ranked items
    """
    hits = 0
    sum_precs = 0
    for n in range(len(ranked_list)):
        if ranked_list[n] in ground_truth:
            hits += 1
            sum_precs += hits / (n + 1.0)
    if hits > 0:
        return sum_precs / len(ground_truth)
    else:
        return 0

def ndcg_at_k(actual, predicted, topk=100):
    if len(predicted) > topk:
        predicted = predicted[:topk]
    score = 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            score +=1.0/np.log2(i + 2.0)
    if not actual:
        return 0.0
    return score / min(len(actual), topk)

def ItemPop(K=200):
    needConcernTrick=pd.read_csv("NeedConcern/Track_Occ.csv")
    
    needConcernTrick.columns=["trick_url","OccNum"]
    b=needConcernTrick.sort_values(by="OccNum" , ascending=False) #所有用户此时的推荐结基本一致
    #歌曲，歌单，歌手，专辑关系
#    tracksWithDetail=pd.read_csv("NeedConcern/trickWithDetail.csv")[["album_uri","artist_uri","track_uri","pid"]]
    #待验证的歌单id
    needVerify=list(pd.read_csv("NeedConcern/validpidlist.csv")['pid'])
    with open('NeedConcern/validWithseed', 'rb') as f:
        seeds= pickle.load(f) #alread has tracks
    with open('NeedConcern/validWithNeed', 'rb') as f:
        Real= pickle.load(f)  #real need tracks
    APs=[]
    NDCG=[]
    for p in needVerify:
        recomend=list(b[:K]["trick_url"])#有顺序的所以不能乱删
        vaildless=set(recomend)&set(seeds[p])
        if len(vaildless)>0:
            i=0
            for v in vaildless:#i为序号，v为具体值
                recomend.remove(v)    #替换掉已经有的转而推荐流行度高且没有的
                recomend.append(list(b["trick_url"])[K+i+1])
                i=i+1
        APs.append(AP(recomend, Real[p]))
        NDCG.append(ndcg_at_k(Real[p], recomend))
    MAP=np.mean(APs)
    NDCG_100=np.mean(NDCG)
    print("ItemPop的结果：MAP",MAP,"NDCG@100",NDCG_100)
    return 
#ItemPop(K=100)  
def ItemPopCommen(traintable,needVerifyPid,K=100,itemId="track_uri"):
    """
    #这个函数以后可以通用
    """
    #traintable是用来计算流行度的用户-item关系列表
    #1.计算流行度列表
    df_item_clicks =(traintable.groupby(itemId).size().reset_index(name="n_clicks"))
    #df_item_clicks 是个流行度列表有两列 itemId,"n_clicks"
    b=df_item_clicks.sort_values(by="n_clicks" , ascending=False) #所有用户此时的推荐结基本一致
    with open('NeedConcern/validWithseed', 'rb') as f:
        seeds= pickle.load(f) #alread has tracks
    with open('NeedConcern/validWithNeed', 'rb') as f:
        Real= pickle.load(f)  #real need tracks
    APs=[]
    NDCG=[]
    for p in needVerifyPid:
        recomend=list(b[:K][itemId])#有顺序的所以不能乱删
        vaildless=set(recomend)&set(seeds[p])
        if len(vaildless)>0:
            i=0
            for v in vaildless:#i为序号，v为具体值
                recomend.remove(v)    #替换掉已经有的转而推荐流行度高且没有的
                recomend.append(list(b[itemId])[K+i+1])
                i=i+1
        APs.append(AP(recomend, Real[p]))
        NDCG.append(ndcg_at_k(Real[p], recomend))
    MAP=np.mean(APs)
    NDCG_100=np.mean(NDCG)
    return MAP,NDCG_100
def getMatrixP(t,usrId,ItemId):#获取用户和内容的流行度矩阵
    user=list(set(t[usrId]))
    conttent=list(set(t[ItemId]))
    P=np.zeros((len(user),len(conttent)))
    for i,r in t.iterrows():
        uid=user.index(r[usrId])
        urlc=conttent.index(r[ItemId])
        P[uid,urlc]+=1
    P=pd.DataFrame(P,index=user,columns=conttent)
    return P  

def main(ff,strategy="ItemPop",K=100):
    tracksWithDetail=pd.read_csv("NeedConcern/trickWithDetail.csv")[["album_uri","artist_uri","track_uri","pid"]]
    topicOfPlaylist=pd.read_csv("Pid_Topic.csv",index_col='pid')
    #待验证的歌单id
    needVerify=list(pd.read_csv("NeedConcern/validpidlist.csv")['pid'])
    needVertifyTopic=['100']#'18','71',
    with open('NeedConcern/validWithseed', 'rb') as f:
        seeds= pickle.load(f) #alread has tracks
    with open('NeedConcern/validWithNeed', 'rb') as f:
        Real= pickle.load(f)  #real need tracks
    for t in needVertifyTopic:
        currentPid=list(topicOfPlaylist[topicOfPlaylist[t]!=-1].index)
        needTest=list(set(needVerify)&set(currentPid))
        trainpid=list(set(currentPid)-set(needTest))
        
        
        if strategy=="ItemPop":
            csv=tracksWithDetail[tracksWithDetail['pid'].isin(trainpid)]#这个csv才是应该被拿去训练的user-item交互列表
            MAP,NDCG_100=ItemPopCommen(csv,needTest,K,"track_uri")
        elif strategy=="BPR":
            csv=tracksWithDetail[tracksWithDetail['pid'].isin(currentPid)]
            AllTracks=csv['track_uri'].unique()
            MyBPR=BPR(session_key = 'pid', item_key = 'track_uri')
            MyBPR.fit(csv)
            APs=[]
            NDCG=[]
            for p in needTest:
                prev_iid=seeds[p]                  #测试集已有的itemid
                items_to_predict=AllTracks          #需要给出分数的itemid
                preds=MyBPR.predict_next(p, prev_iid, items_to_predict)#拿到当前候选集合的得分
                preds[np.isnan(preds)] = 0
                preds += 1e-8 * np.random.rand(len(preds)) #Breaking up ties
                rec=preds.sort_values(ascending=False).index[:K]
                APs.append(AP(rec, Real[p]))
                NDCG.append(ndcg_at_k(Real[p], rec))
            MAP=np.mean(APs)
            NDCG_100=np.mean(NDCG)  
        else:
            #基于流行度矩阵的
            csv=tracksWithDetail[tracksWithDetail['pid'].isin(currentPid)]
            trainP=getMatrixP(csv,usrId="pid",ItemId="track_uri")#获得流行度矩阵
            if strategy=="item_CF":
               
                cf = CF_knearest(k=K)
                rec=cf.fit(trainP,needTest)    #返回一个推荐列表key为usrId,value为ItemId
                 
            elif strategy=="MF":
                cf = M_F(trainP,K,needTest)
                rec=cf.matrix_fac(0.001,0.0002)  
                
            
            elif strategy=="LSI":
                cf = CF_svd(k=K, r=3)
                rec=cf.fit(trainP,needTest)
            APs=[]
            NDCG=[]
            for p in needTest:
                APs.append(AP(rec[p], Real[p]))
                NDCG.append(ndcg_at_k(Real[p], rec[p]))
            MAP=np.mean(APs)
            NDCG_100=np.mean(NDCG)  
        print("Topic:",t,"  "+strategy+"的结果：MAP",MAP,"NDCG@100",NDCG_100)
        print("Topic:",t,"  "+strategy+"的结果：MAP",MAP,"NDCG@100",NDCG_100,"           \n",file=ff)

#main("ItemPop")    
def discussTopic():
    Strategys=["ItemPop"]    
    ff=open('Result.txt','w')
    #main(ff,"BPR") 
    for s in Strategys:
        main(ff,s)
    ff.close() 
discussTopic()
#ItemPop(K=100)     
def Allmain(ff,strategy="ItemPop",K=100):
    tracksWithDetail=pd.read_csv("NeedConcern/trickWithDetail.csv")[["album_uri","artist_uri","track_uri","pid"]]
     #主题索引表
    topicOfPlaylist=pd.read_csv("Pid_Topic.csv",index_col='pid')#看前5个主题
    #待验证的歌单id
    needVerify=list(pd.read_csv("NeedConcern/validpidlist.csv")['pid'])
#    needVertifyTopic=['18','71','33','45']#'18','71',
    with open('NeedConcern/validWithseed', 'rb') as f:
        seeds= pickle.load(f) #alread has tracks
    with open('NeedConcern/validWithNeed', 'rb') as f:
        Real= pickle.load(f)  #real need tracks
    currentPid=list(topicOfPlaylist.index)#一万歌单
    needTest=list(set(needVerify)&set(currentPid))
    trainpid=list(set(currentPid)-set(needTest))
    if strategy=="ItemPop":
        csv=tracksWithDetail[tracksWithDetail['pid'].isin(trainpid)]#这个csv才是应该被拿去训练的user-item交互列表
        MAP,NDCG_100=ItemPopCommen(csv,needTest,K,"track_uri")
    elif strategy=="BPR":
        csv=tracksWithDetail[tracksWithDetail['pid'].isin(currentPid)]
        AllTracks=csv['track_uri'].unique()
        MyBPR=BPR(session_key = 'pid', item_key = 'track_uri')
        MyBPR.fit(csv)
        APs=[]
        NDCG=[]
        for p in needTest:
            prev_iid=seeds[p]                  #测试集已有的itemid
            items_to_predict=AllTracks          #需要给出分数的itemid
            preds=MyBPR.predict_next(p, prev_iid, items_to_predict)#拿到当前候选集合的得分
            preds[np.isnan(preds)] = 0
            preds += 1e-8 * np.random.rand(len(preds)) #Breaking up ties
            rec=preds.sort_values(ascending=False).index[:K]
            APs.append(AP(rec, Real[p]))
            NDCG.append(ndcg_at_k(Real[p], rec))
        MAP=np.mean(APs)
        NDCG_100=np.mean(NDCG)  
    else:
        #基于流行度矩阵的
        csv=tracksWithDetail[tracksWithDetail['pid'].isin(currentPid)]
        trainP=getMatrixP(csv,usrId="pid",ItemId="track_uri")#获得流行度矩阵
        if strategy=="item_CF":
           
            cf = CF_knearest(k=K)
            rec=cf.fit(trainP,needTest)    #返回一个推荐列表key为usrId,value为ItemId
             
        elif strategy=="MF":
            cf = M_F(trainP,K,needTest)
            rec=cf.matrix_fac(0.001,0.0002)  
            
        
        elif strategy=="LSI":
            cf = CF_svd(k=K, r=3)
            rec=cf.fit(trainP,needTest)
        APs=[]
        NDCG=[]
        for p in needTest:
            APs.append(AP(rec[p], Real[p]))
            NDCG.append(ndcg_at_k(Real[p], rec[p]))
        MAP=np.mean(APs)
        NDCG_100=np.mean(NDCG)  
    print(strategy+"的结果：MAP",MAP,"NDCG@100",NDCG_100)
    print(strategy+"的结果：MAP",MAP,"NDCG@100",NDCG_100,"           \n",file=ff)
    return
#ff=open('ALLPlaylistResult.txt','w')
##Allmain(ff,"ItemCF") 
#Strategys=["item_CF","MF","LSI","BPR"]    
#for s in Strategys:
#    Allmain(ff,s)
#ff.close()  
#    
    
        
    
    
    
    