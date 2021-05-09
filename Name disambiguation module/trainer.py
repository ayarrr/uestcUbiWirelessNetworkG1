from utility import save_embedding
import random
import pandas as pd
class TrainHelper():
    @staticmethod
    #num_epoch=8,dataset,optimizer,evaluator,metapathsample_dict
    def helper(num_epoch,dataset,optimizer,evaluator,metapathsample_dict):
        optimizer.init_model(dataset)
        optimizer.init_modelwithDOC2VEC(dataset)#引入语义网络中的向量表示
#        average_f1 = evaluator.compute_f1(dataset,optimizer.paper_latent_matrix)
#        print('init_f1 is ',str(average_f1))
        r=[]
        """
        所有路径采样后的样本都是3元组，长路径只保留两个端点和中心点
        """  
        for _ in range(num_epoch):#迭代这么多次
            bpr_loss = 0.0
            for k,samples in metapathsample_dict.items():
                start=k[0]
                if len(k)==3:
                    c=k[1]
                else:
                    c=k[2]
                
                if(start=='P'):#端点是目标节点，调一阶
                    metasaple=samples.generte(dataset)
                    for s in metasaple:#拿到各个样本，s是四元组如（P,A,P，pt）
#                        print("一阶更新",k,s[0],s[2],s[3])
                        optimizer.updateEndSGD(s[0],s[2],s[3])
                if(c=='P'):#中心点是目标节点，调二阶
                    if len(metasaple)==0:
                        metasaple=samples.generte(dataset)
                    #拿到各个样本，s是三元组如（P,A,P）
                    i=0
                    while(i<len(metasaple)):
                        m1=metasaple[i]
                        j=i+1
                        while j<len(metasaple):
                            if(m1[0]==metasaple[j][0] and m1[2]==metasaple[j][2])or(m1[0]==metasaple[j][2] and m1[2]==metasaple[j][0]):
#                                print("need adjust",k,m1,metasaple[j])
                                p_tmetapath = random.choice(metasaple)
                                while (m1[0]==p_tmetapath[0] and m1[2]==p_tmetapath[2])or(m1[0]==p_tmetapath[2] and m1[2]==p_tmetapath[0]):
                                    p_tmetapath = random.choice(metasaple)
                                optimizer.updateCenterSGD(m1[1],metasaple[j][1],p_tmetapath[1])
                                metasaple.remove(metasaple[j])
                            j=j+1
                        i=i+1
                metasaple.clear()
            average_f1 = evaluator.compute_f1(dataset,optimizer.paper_latent_matrix)
#            print('f1 is ',str(average_f1))
            r.append(average_f1)
        save_embedding(optimizer.paper_latent_matrix,
                       dataset.paper_list, optimizer.latent_dimen)    
        pd.DataFrame({"My":r}).to_csv("metaHb2.csv")
        print('f1 is ',str(average_f1))
                        
                        
                    
                     
                        
                    
               
   