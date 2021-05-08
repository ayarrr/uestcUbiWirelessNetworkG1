# -*- coding: utf-8 -*-
"""
Created on Thu 10/22 9:29:46 2019

@author: Administrator
"""
import networkx as nx
from networkx.algorithms import bipartite
import pandas as pd
import numpy as np
import pickle
class generateG(object):
    #设置输入参数并给参数赋值
    def __init__(self,csvfile,topic,playlist_Weight=0.5,artist_Weight=0.4,album_Weight=0.1):
        self.csvfile=csvfile #局部csv 当前主题下的
        self.topic=topic   #当前topic的标记
        self.pW=playlist_Weight  #同歌单关系权重
        self.tW=artist_Weight #歌手关系权重
        self.aW=album_Weight  #专辑关系
        
      
    #构造二分图，并投影只挑出歌曲节点
    def getWholeTricksRelation(self):
        df=self.csvfile
        tracks=list(set(df["track_uri"]))
        PTG=nx.from_pandas_edgelist(df,'pid', 'track_uri')
        playlistget=bipartite.weighted_projected_graph(PTG,tracks)
        playlistdf=nx.to_pandas_edgelist(playlistget, nodelist=playlistget.nodes())

        artistG=nx.from_pandas_edgelist(df,'artist_uri', 'track_uri')
        artistGet=bipartite.weighted_projected_graph(artistG,tracks)

        albumG=nx.from_pandas_edgelist(df,'album_uri', 'track_uri')
        albumGet=bipartite.weighted_projected_graph(albumG,tracks)
        
        playlistdf['weight']=playlistdf["weight"].apply(lambda x:x*self.pW)
        playlistget=nx.from_pandas_edgelist(playlistdf,'source', 'target','weight')
       #合并播放列表-歌曲，歌手-歌曲 这两种关系
        for (u,v,d) in artistGet.edges(data='weight'):
            
            if (playlistget.has_edge(u,v)):
                playlistget.add_edge(u,v,weight=playlistget.get_edge_data(u,v)['weight']+
                                     d*self.tW)
            else:
               
                playlistget.add_edge(u,v,weight=d*self.tW)
               
        for (u,v,d) in albumGet.edges(data='weight'):
            if (playlistget.has_edge(u,v)):
                playlistget.add_edge(u,v,weight=playlistget.get_edge_data(u,v)['weight']+
                                     d*self.aW)
            else:
                playlistget.add_edge(u,v,weight=d*self.aW)
#        nx.write_gml(playlistget,'AllTopic/'+self.topic+'_trickWithDetail.gml')
        return playlistget
    

import random
#跑pagerank得到推荐歌曲
def getRecomend(seed,G,k):
    pr=nx.pagerank(G,alpha=0.85,personalization=seed,max_iter=100, tol=1e-04)   
    pr1=sorted(pr.items(), key=lambda d: d[1],reverse=True)
    result=list(filter(lambda x:x[0] not in seed.keys(),pr1))
    result=list(map(lambda x:x[0],result))[:k]
    return result

def ndcg_at_k(actual, predicted, topk=100):
    if len(predicted) > topk:
        predicted = predicted[:topk]
    score = 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            score +=1.0 / np.log2(i + 2.0)
    if not actual:
        return 0.0
    return score / min(len(actual), topk)
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


#tracksNet为当前游走的图，currentTest为当前要验证的歌曲列表
#输出为本歌单的命中率，
def getAccuracy(currentTest,tracksNet,k):
    with open('NeedConcern/validWithseed', 'rb') as f:
        seeds= pickle.load(f)[currentTest] #alread has tracks
    with open('NeedConcern/validWithNeed', 'rb') as f:
        Real= pickle.load(f)[currentTest]  #real need tracks
    seedNum=len(seeds)
    if(seedNum!=0):
        seed=dict(zip(seeds,[1/seedNum]*seedNum))
    else:
        seed={}
    recomend=getRecomend(seed,tracksNet,len(Real))
    MAP=AP(recomend, Real)
#    common=set(Real)&set(recomend)
#    print(common)
    ndcg=ndcg_at_k(Real, recomend, topk=100)
    return MAP,recomend,ndcg



#目标函数：使得本主题下各个歌单的推荐命中率的均值达到最大
#对于单个主题找使得目标函数达到最大的参数，
#topic 主题序号
    #csv 该主题下所有关系 dataframe
    #AsMytest 该主题下待验证的歌单 list
#建图后推荐,获得命中率的均值
def goal(args):
    with open('tmp.file', 'rb') as f:
        sparam = pickle.load(f)
    csv=pd.read_csv("current.csv")
    TrackNum=len(csv['track_uri'].unique())
    topic1=sparam['topic1']
    needTest=sparam['needTest']
    pw,aw,tw=args
    g=generateG(csv,topic1,playlist_Weight=pw,artist_Weight=tw,album_Weight=aw)
    currentNet=g.getWholeTricksRelation()   
    Accuracys=[] #准确率List

#    pre={}#各个歌单推荐出来的内容
    for playlistid in needTest:
        #返回准了多少个，以及推的啥,ndcg
        accu,reco,ndcg=getAccuracy(playlistid,currentNet,500)
#        pre[playlistid]=reco
        Accuracys.append(accu)
    return -sum(Accuracys)+0.001
def getgoal(topic,csv,AsMytest,pw,aw,tw):
    topic,csv,AsMytest,pw,aw,tw
    g=generateG(csv,topic,playlist_Weight=pw,artist_Weight=tw,album_Weight=aw)
    currentNet=g.getWholeTricksRelation()   
    TrackNum=len(csv['track_uri'].unique())#该主题的歌曲总数
    Accuracys=[] #准确率List
    pre={}#各个歌单推荐出来的内容
    ndcgs=[]
    topicTag=[topic]*len(AsMytest)
    for playlistid in AsMytest:
        #返回准了多少个，以及推的啥
        accu,reco,ndcg=getAccuracy(playlistid,currentNet,500)
        #print(playlistid,reco)
        pre[playlistid]=reco
        Accuracys.append(accu) #每个pid推对了多少
        ndcgs.append(ndcg)
    bestAccuracys=pd.DataFrame({"pid":AsMytest,"REcall":Accuracys,"NDCG":ndcgs,"TopicTag":topicTag})
#    bestAccuracys.to_csv("NeedConcern/"+topic+"_Accuacy.csv",index_label=False)
    pd.DataFrame(pre).to_csv("NeedConcern/topic_"+topic+"_"+str(TrackNum)+"_recomend.csv",index_label=False)
    return bestAccuracys,np.mean(Accuracys),np.mean(ndcgs)
from hyperopt import hp, fmin, rand, tpe, space_eval
sparam={}
#分主题讨论,max_evals指定各个主题考虑多少轮
def discuss(max_evals):
    #主题索引表
    topicOfPlaylist=pd.read_csv("Pid_Topic.csv",index_col='pid')#看前5个主题
    #歌曲，歌单，歌手，专辑关系
    tracksWithDetail=pd.read_csv("NeedConcern/trickWithDetail.csv")[["album_uri","artist_uri","track_uri","pid"]]
    #待验证的歌单id
    needVerify=list(pd.read_csv("NeedConcern/validpidlist.csv")['pid'])
    
#    with open('NeedConcern/validWithNeed', 'rb') as f:
#        Real= pickle.load(f)  #real need tracks,各个验证集pid
    currentPid=random.sample(list(topicOfPlaylist.index), 100)
    needTest=list(set(needVerify)&set(currentPid))
    csv=tracksWithDetail[tracksWithDetail['pid'].isin(currentPid)]
    #测试集中需要被移除的关系
#    for rt in needTest:
#        #Real[rt]这些歌曲-歌单关系需要被去掉
#        csv=csv[(~csv['track_uri'].isin(Real[rt]))&( csv['pid']==rt)]
    TrackNum=len(csv['track_uri'].unique())
    if(len(needTest)==0 ):#or TrackNum<500 or len(currentPid)>200 ):
        pass
    else:
        topic1="None topic"
        space = [
                        hp.uniform('pw', 0,1),
                         hp.uniform('aw', 0,1),
                          hp.uniform('tw', 0,1)]
                #TODO
        sparam['needTest']=needTest
        sparam['topic1']=topic1
        csv.to_csv("current.csv",index_label=False)
        with open('tmp.file', 'wb') as f:
            pickle.dump(sparam, f)
        best = fmin(goal,space,algo=tpe.suggest,max_evals=max_evals)
        bestAccuracys,MAP,mndcg=getgoal(topic1,csv,needTest,best['pw'],best['aw'],best['tw'])
        print(best['pw'],best['aw'],best['tw'],"没有主题下平均准确率",MAP,mndcg)
    print("Done!")                
    return
discuss(30)