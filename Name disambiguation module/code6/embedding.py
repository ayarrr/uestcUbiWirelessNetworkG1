import numpy as np
from utility import sigmoid
import pandas as pd
from gensim.models.doc2vec import Doc2Vec,TaggedDocument
from os import listdir
from os.path import isfile, join
from random import choice
class BprOptimizer():
    """
    latent_dimen: latent dimension
    alpha: learning rate
    matrix_1reg: regularization parameter of matrix 一阶的惩罚力度
    matrix_2reg: regularization parameter of matrix 二阶的惩罚力度
    when you have many type node to learn ,you can use many matrix
    We just need papers' embedding
    """
    def __init__(self, latent_dimen, alpha, matrix_1reg,matrix_2reg):
        self.latent_dimen = latent_dimen#最终要学多大维度的向量
        self.alpha = alpha#学习率
        self.matrix_1reg = matrix_1reg#更新过程的一阶相似的惩罚力度
        self.matrix_2reg = matrix_2reg#更新过程的一阶相似的惩罚力度
        
    def init_modelwithDOC2VEC(self, dataset):
        """
        使用doc2vec初始化embedding
        """
        dataPath=dataset.datapath+dataset.ego+"/"
        docLabels = []
        docLabels = [f for f in listdir(dataPath) if f.endswith('.txt')]
        data = []
        corpora_documents = []
        for doc in docLabels:
        #    print(doc)
            f=open(dataPath + doc,"r",encoding='utf-8')
            s=f.read()
            data.append(s)
            t=TaggedDocument(s,[doc])
            corpora_documents.append(t)
        
        
        model = Doc2Vec(vector_size=self.latent_dimen, min_count=1, window=10,  sample=1e-4, negative=2, workers=8)
        model.build_vocab(corpora_documents)
        model.train(corpora_documents,total_examples=len(docLabels),epochs=10)
        
        model.save(dataPath+"/"+dataset.ego+".model")
#        vector = model.infer_vector(data[0])  
#        print(vector)
#        print(len(docLabels),len(data),len(self.paper_latent_matrix.keys()))
        for paper_idx in range(len(dataset.paper_list)):
            pid=docLabels[paper_idx]#找到真实pid  eg '5bc6bfdd486cef66309a9f87.txt'
            pid=pid.split('.')[0]
            pidindex=dataset.paper_list.index(pid)
            cupdate=choice([True,False])
            if cupdate:
                self.paper_latent_matrix[pidindex]=model.infer_vector(data[paper_idx])
        
        
    def init_model(self, dataset):
        """
        initialize matrix using uniform [-0.2, 0.2]
        """
        self.paper_latent_matrix = {}#只需要文章的
        for paper_idx in range(len(dataset.paper_list)):
            self.paper_latent_matrix[paper_idx] = np.random.uniform(-0.2, 0.2,
                                                                    self.latent_dimen)
    def computeS(self, fst, snd):#计算隐空间中两个向量的距离，此处直接以内积算
        return np.dot(self.paper_latent_matrix[fst],
                          self.paper_latent_matrix[snd])
#    def computeS(self,fst, snd):
#        return np.linalg.norm(self.paper_latent_matrix[fst] - self.paper_latent_matrix[snd])
        
    def updateEndSGD(self, fst, snd, third):#j表示当前节点，k表示当前节点的邻居
        """
        SGD inference
        """
        x = self.computeS(fst, snd) - \
            self.computeS(fst, third)
        common_term = sigmoid(x) - 1

        grad_fst = common_term * (self.paper_latent_matrix[snd] - \
                                  self.paper_latent_matrix[third]) + \
                    2 * self.matrix_1reg * self.paper_latent_matrix[fst]
                    #以grad_fst同时更新三个作者向量
        self.paper_latent_matrix[fst] = self.paper_latent_matrix[fst] - \
                                         self.alpha * grad_fst

        grad_snd = common_term * self.paper_latent_matrix[fst] + \
                   2 * self.matrix_1reg * self.paper_latent_matrix[snd]
        self.paper_latent_matrix[snd]= self.paper_latent_matrix[snd] - \
                                        self.alpha * grad_snd

        grad_third = -common_term * self.paper_latent_matrix[fst] + \
                     2 * self.matrix_1reg * self.paper_latent_matrix[third]
        self.paper_latent_matrix[third] = self.paper_latent_matrix[third] - \
                                           self.alpha * grad_third
      
    def updateCenterSGD(self, fst, snd, third):#在原空间结构相似，更新二阶相似的
        """
        SGD inference
        """
        x = self.computeS(fst, snd) - \
            self.computeS(fst, third)
        common_term = sigmoid(x) - 1

        grad_fst = common_term * (self.paper_latent_matrix[snd] - \
                                  self.paper_latent_matrix[third]) + \
                    2 * self.matrix_2reg * self.paper_latent_matrix[fst]
                    #以grad_fst同时更新三个作者向量
        self.paper_latent_matrix[fst] = self.paper_latent_matrix[fst] - \
                                         self.alpha * grad_fst

        grad_snd = common_term * self.paper_latent_matrix[fst] + \
                   2 * self.matrix_2reg * self.paper_latent_matrix[snd]
        self.paper_latent_matrix[snd]= self.paper_latent_matrix[snd] - \
                                        self.alpha * grad_snd

        grad_third = -common_term * self.paper_latent_matrix[fst] + \
                     2 * self.matrix_2reg * self.paper_latent_matrix[third]
        self.paper_latent_matrix[third] = self.paper_latent_matrix[third] - \
                                           self.alpha * grad_third
        return
    def compute_onehop_loss(self, fst, snd, third):#loss includes ranking loss and model complexity
#        print("gghghh")

        x = self.computeS(fst, snd)-self.computeS(fst, third)
        ranking_loss = -np.log(sigmoid(x))
    
        complexity = 0.0
        complexity += self.matrix_1reg * np.dot(self.paper_latent_matrix[fst],
                                               self.paper_latent_matrix[fst])
        complexity += self.matrix_1reg * np.dot(self.paper_latent_matrix[snd],
                                               self.paper_latent_matrix[snd])
        complexity += self.matrix_1reg * np.dot(self.paper_latent_matrix[third],
                                               self.paper_latent_matrix[third])
        return ranking_loss + complexity
    def compute_2hop_loss(self, fst, snd, third):
        x = self.computeS(fst, snd)-self.computeS(fst, third)
        ranking_loss = -np.log(sigmoid(x))
    
        complexity = 0.0
        complexity += self.matrix_2reg * np.dot(self.paper_latent_matrix[fst],
                                               self.paper_latent_matrix[fst])
        complexity += self.matrix_2reg * np.dot(self.paper_latent_matrix[snd],
                                               self.paper_latent_matrix[snd])
        complexity += self.matrix_2reg * np.dot(self.paper_latent_matrix[third],
                                               self.paper_latent_matrix[third])
        return ranking_loss + complexity

  