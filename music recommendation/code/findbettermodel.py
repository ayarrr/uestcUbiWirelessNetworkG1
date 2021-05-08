# -*- coding: utf-8 -*-
import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
def gettxt(path,idlist):
    mylist=[]
    for i in idlist:
        inputpath=str(path+str(i)+".txt")
        with open(inputpath,encoding="utf-8") as f:
            txt=f.read()
            wordLst = nltk.word_tokenize(txt)
                        #去除停用词
            filtered = [w for w in wordLst if w not in stopwords.words('english')]
#            #仅保留名词或特定POS   
            refiltered =nltk.pos_tag(filtered)
            filtered = [w for w, pos in refiltered if pos.startswith('NN')]
            #词干化
            ps = PorterStemmer()
            filtered = [ps.stem(w) for w in filtered]
            mylist.append(" ".join(filtered))
    return mylist,len(idlist)
def save_top_words(model, feature_names, n_top_words):
    #保存每个主题下权重较高的term
    T_w=pd.DataFrame()
    for topic_idx, topic in enumerate(model.components_):
        T_w["topic"+str(topic_idx)]=[feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
    T_w.to_csv("topic_word.csv",index=False)
    return
#把词转换成词频向量
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
from sklearn.decomposition import LatentDirichletAllocation
#myindexcsv=pd.read_csv("cleanResult/WithPidAllpaname.csv")
#import random 
#pidlist=list(myindexcsv["pid"])
pidlist=list(pd.read_csv("../NeedConcern/pidlist.csv")["pid"])
#corpus,n = gettxt(r"ladprocesstest/Output2/",pidlist)
#corpus,n = gettxt(r"Output1/",pidlist)
corpus,n = gettxt(r"../Output1/",pidlist)

#corpus.extend(testcorpus)
cntVector = CountVectorizer(stop_words={'english'},encoding='utf-8')
cntTf = cntVector.fit_transform(corpus)
joblib.dump(cntVector,"cntVector.model" )
#joblib.dump(cntVector,"cntTf.model" )
print("cntV完成")
#LDA主题模型训练        K=20 -50
from sklearn.decomposition import LatentDirichletAllocation
n_topics = range(30, 100)
perplexityLst = [1.0]*len(n_topics)
#训练LDA并打印训练时间
lda_models = []
for idx, n_topic in enumerate(n_topics):
    lda = LatentDirichletAllocation(n_topics=n_topic,
                                    max_iter=30,
                                    learning_method='batch',
                                    evaluate_every=200,
                                    verbose=0
                    
                                    )
    lda.fit(cntTf)
    perplexityLst[idx] = lda.perplexity(cntTf)
    lda_models.append(lda)
#    print(n_topic,lda.perplexity(cntTf))
best_index = perplexityLst.index(min(perplexityLst))
best_n_topic = n_topics[best_index]
best_model = lda_models[best_index]
joblib.dump(best_model,"best_model.model" )
print ("Best # of Topic: ", best_n_topic)
    #打印主题-词语分布矩阵
#    print(model.components_)
n_top_words=10
tf_feature_names = cntVector.get_feature_names()
#每个主题下权重较高的词语
#print_top_words(lda, tf_feature_names, n_top_words)
save_top_words(best_model, tf_feature_names, n_top_words)
#print(lda.components_.shape)
#利用已有模型得到语料X中每篇文档的主题分布
#print(best_model.transform(cntTf))
doc_topic=pd.DataFrame(best_model.transform(cntTf))
doc_topic["pid"]=pd.Series(pidlist)
doc_topic.to_csv("doc_topic.csv",index=False)
pd.DataFrame(best_model.components_).to_csv("topic_word1.csv",index=False)
print(best_model.perplexity(cntTf))