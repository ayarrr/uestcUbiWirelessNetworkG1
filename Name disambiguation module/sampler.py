import numpy as np
import random

class PAPSampler():
    def generte(self,dataset):#都得叫generte
        """
        返回所有PAP样本（路径样本不重复）
        """
        pset=[]
#        print(len(dataset.paper_list))
        for P1 in range(len(dataset.paper_list)):
            if len(dataset.paper_authorlist_dict[P1])==0:
                continue
#            print(p1,P1)
            p_nei_list=list(dataset.AP_Graph.neighbors(P1))
            
            A=random.choice(p_nei_list)
            A_nei=list(dataset.AP_Graph.neighbors(A))
            A_nei.remove(P1)
            if len(A_nei)<1 :
                continue 
            P2=random.choice(A_nei)
            if ((P1,A,P2) in pset) or ((P2,A,P1) in pset):
                continue
            else:
                p_t = random.choice(range(len(dataset.paper_list)))
                if len(dataset.paper_authorlist_dict[p_t])==0:
                    pass
                else:
                    pt_nei=set(dataset.AP_Graph.neighbors(p_t))
                    while len(pt_nei.intersection(set(p_nei_list)))>0:
                        p_t = random.choice(range(len(dataset.paper_list)))
                        if len(dataset.paper_authorlist_dict[p_t])==0:
                            break
                        pt_nei=set(dataset.AP_Graph.neighbors(p_t))
                pset.append((P1,A,P2,p_t))
             
        return pset
class PVPSampler():#PVP路径采样
    def generte(self,dataset):#都得叫generte
        """
        返回所有PVP样本（路径样本不重复）
        """
        pset=[]
        for P1 in range(len(dataset.paper_list)):
            p_nei_list=list(dataset.PV_Graph.neighbors(P1))
            V=random.choice(p_nei_list)
            V_nei=list(dataset.PV_Graph.neighbors(V))
            V_nei.remove(P1)#移除自身
            if len(V_nei)<1 :
                continue 
            P2=random.choice(V_nei)
            if ((P1,V,P2) in pset) or ((P2,V,P1) in pset):
                continue
            else:
                p_t = random.choice(range(len(dataset.paper_list)))
                pt_nei=set(dataset.PV_Graph.neighbors(p_t))
                while len(pt_nei.intersection(set(p_nei_list)))>0:
                    p_t = random.choice(range(len(dataset.paper_list)))
                    pt_nei=set(dataset.PV_Graph.neighbors(p_t))
                pset.append((P1,V,P2,p_t))
        return pset
class POPSampler():#PAOAP路径采样简化为POP
    def generte(self,dataset):#都得叫generte
        """
        返回所有POP样本（路径样本不重复）
        """
        pset=[]
        for P1 in range(len(dataset.paper_list)):
            p_nei_list=list(dataset.OP_Graph.neighbors(P1))
            O=random.choice(p_nei_list)
            O_nei=list(dataset.OP_Graph.neighbors(O))
            O_nei.remove(P1)#移除自身
            if len(O_nei)<1 :
                continue 
            P2=random.choice(O_nei)
            if ((P1,O,P2) in pset) or ((P2,O,P1) in pset):
                continue
            else:
                p_t = random.choice(range(len(dataset.paper_list)))
                pt_nei=set(dataset.OP_Graph.neighbors(p_t))
                while len(pt_nei.intersection(set(p_nei_list)))>0:
                    p_t = random.choice(range(len(dataset.paper_list)))
                    pt_nei=set(dataset.OP_Graph.neighbors(p_t))
                pset.append((P1,O,P2,p_t))
        return pset
class APASampler():#APA路径采样
    def generte(self,dataset):#都得叫generte
        """
        返回所有APA样本（路径样本不重复）
        """
        pset=[]
        for A1 in dataset.author_list:
            a_nei_list=list(dataset.AP_Graph.neighbors(A1))
            if len(a_nei_list)==1:#只有一个作品的作者不看
                continue                
            P=random.choice(a_nei_list)
            P_nei=list(dataset.AP_Graph.neighbors(P))
            P_nei.remove(A1)#移除自身
            if len(P_nei)<1 :
                continue 
            A2=random.choice(P_nei)
            if ((A1,P,A2) in pset) or ((A2,P,A1) in pset):
                continue
            else:
                
                pset.append((A1,P,A2))
        return pset
class OPOSampler():#OPO路径采样
    def generte(self,dataset):#都得叫generte
        """
        返回所有OPO样本（路径样本不重复）
        """
        pset=[]
        for O1 in dataset.org_list:
            o_nei_list=list(dataset.OP_Graph.neighbors(O1))
            if len(o_nei_list)==1:#只有一个作品的单位不看
                continue                
            P=random.choice(o_nei_list)
            P_nei=list(dataset.OP_Graph.neighbors(P))
            P_nei.remove(O1)#移除自身
            if len(P_nei)<1 :
                continue 
                
                
            O2=random.choice(P_nei)
            if ((O1,P,O2) in pset) or ((O2,P,O1) in pset):
                continue
            else:
                pset.append((O1,P,O2))
        return pset
    
class PAPAPSampler():#PAPAP路径采样
    def generte(self,dataset):#都得叫generte
        """
        返回所有PAPAP样本（路径样本不重复）
        """
        pset=[]
        for P1 in range(len(dataset.paper_list)):
            if len(dataset.paper_authorlist_dict[P1])==0:
                continue
            p1_nei_list=list(dataset.AP_Graph.neighbors(P1))
            A1=random.choice(p1_nei_list)
            A_nei=list(dataset.AP_Graph.neighbors(A1))
            A_nei.remove(P1)
            if len(A_nei)<1:
                continue
            P2=random.choice(A_nei)
            p2_nei_list=list(dataset.AP_Graph.neighbors(P2))
            p2_nei_list.remove(A1)
            if len(p2_nei_list)<1:
                continue
            A2=random.choice(p2_nei_list)
            A2_nei=list(dataset.AP_Graph.neighbors(A2))
            A2_nei.remove(P2)
            if P1 in A2_nei:#去除环装的
                A2_nei.remove(P1)
            if len(A2_nei)<1:
                continue
            P3=random.choice(A2_nei)
            if ((P1,A1,P2,A2,P3) in pset) or ((P3,A2,P2,A1,P1) in pset):
                continue
            else:
                p_t = random.choice(range(len(dataset.paper_list)))
                if len(dataset.paper_authorlist_dict[p_t])==0:
                    pass
                else:
                    one_hop=set()
                    for p1_a in p1_nei_list:
                        one_hop=one_hop|set(dataset.AP_Graph.neighbors(p1_a))
                    one_auhop=set()
                    for p1_p in one_hop:
                        one_auhop=one_auhop|set(dataset.AP_Graph.neighbors(p1_p))
                    pt_nei=set(dataset.AP_Graph.neighbors(p_t))
                    while len((set(p1_nei_list)|one_auhop).intersection(pt_nei))>0:
                        p_t = random.choice(range(len(dataset.paper_list)))
                        if len(dataset.paper_authorlist_dict[p_t])==0:
                            break                  
                        pt_nei=set(dataset.AP_Graph.neighbors(p_t))
                pset.append((P1,A1,P2,A2,P3,p_t))
        res=[]
        for p1,a1,p2,a2,p3,p_t in pset:#只保留中心和端点
             res.append((p1,p2,p3,p_t))
            
        return res
class PVPAPSampler():#PVPAP路径采样
    def generte(self,dataset):#都得叫generte
        """
        返回所有PVPAP样本（路径样本不重复）
        """
        pset=[]
        for P1 in range(len(dataset.paper_list)):
            if len(dataset.paper_authorlist_dict[P1])==0:
                continue
            p1_nei_list=list(dataset.AP_Graph.neighbors(P1))
            A1=random.choice(p1_nei_list)
            A_nei=list(dataset.AP_Graph.neighbors(A1))
            A_nei.remove(P1)
            if len(A_nei)<1:
                continue
            P2=random.choice(A_nei)
            p2_nei_list=list(dataset.PV_Graph.neighbors(P2))#候选V
            if len(p2_nei_list)<1:
                continue
            V=random.choice(p2_nei_list)
            V_nei=list(dataset.PV_Graph.neighbors(V))
            V_nei.remove(P2)
            if P1 in V_nei:#去除环装的
                V_nei.remove(P1)
            if len(V_nei)<1:
                continue
            P3=random.choice(V_nei)
            if (P1,A1,P2,V,P3) in pset:
                continue
            else:
                p_t = random.choice(range(len(dataset.paper_list)))
                if len(dataset.paper_authorlist_dict[p_t])==0:
                    pass
                else:
                    one_hop=set()
                    for p1_a in p1_nei_list:
                        one_hop=one_hop|set(dataset.AP_Graph.neighbors(p1_a))
                    one_vhop=set()
                    for p1_p in one_hop:
                        one_vhop=one_vhop|set(dataset.PV_Graph.neighbors(p1_p))
                    pt_nei=set(dataset.PV_Graph.neighbors(p_t))
                    while len(pt_nei.intersection(one_vhop))>0:
                        p_t = random.choice(range(len(dataset.paper_list)))
                        pt_nei=set(dataset.PV_Graph.neighbors(p_t))
                pset.append((P1,A1,P2,V,P3,p_t))
        res=[]
        for p1,a1,p2,v,p3,pt in pset:#只保留中心和端点
            res.append((p1,p2,p3,pt))
            
        return res