import networkx as nx
import os
import json
class DataSet():
    def __init__(self, paper,name, haslabel):
        self.ego=name #当前有歧义的作者的名字
        self.Data=paper#当前人的所有原始数据
        self.paper_authorlist_dict = {}#存当前文章有哪些作者
        self.paper_orglist_dict = {}#存当前文章有那些合作单位,包括
        self.paper_list = [] #存各种文章的真实id，下标是其编码
        self.venue_list = [] #存各种会议name
        self.author_list = []#存相关文章的所有作者
        self.org_list = []#存所有相关单位
        self.OP_Graph = nx.Graph()
        self.AP_Graph = nx.Graph()
        self.PV_Graph = nx.Graph()
        self.num_nnz = 0#决定采样的次数,使用异质网络中的总节点数
        self.datapath="alltextinformation/"#文本信息存储的位置
        self.haslabelP=haslabel#有label的文章id
        self.savetxt()#准备文本数据
        

    def reader_struct(self):#获取结构信息
        paper_index = 0
        author_set = set()
        venue_set=set()
        org_set=set()
        for p in self.Data:
            if p['id'] not in self.haslabelP:
                continue
#            print(p)
            self.paper_list.append(p['id'])
            pv=str(p['venue'])
            venue_set.add(pv)
            self.PV_Graph.add_edge(paper_index,pv)
            cau=[]
            corg=[]
#            if(paper_index==327):
#                        print("find 327!!!!!!!!!!!!!",p['authors'])
            for a in p['authors']:
                authorname=str(a['name'])
                if authorname!=self.ego:
                    cau.append(authorname)
                    self.AP_Graph.add_edge(paper_index,authorname)
                    
                orgname=str(a['org'])
                corg.append(orgname)#也要获取有歧义作者的单位
                self.OP_Graph.add_edge(paper_index,orgname)
            self.paper_authorlist_dict[paper_index]=cau
            self.paper_orglist_dict[paper_index]=corg
            author_set=author_set|set(cau)
            org_set=org_set|set(corg)
            paper_index=paper_index+1; 
        self.author_list=list(author_set)
        self.venue_list=list(venue_set)
        self.org_list=list(org_set)
        self.num_nnz=len(self.paper_list)+len(self.author_list)
        +len(self.venue_list)+len(self.org_list)
        return
    def savetxt(self):#把文本存入TXT
        txtdir=self.datapath
        name=self.ego
        if not(os.path.exists(txtdir+name)):
            os.mkdir(txtdir+name) 
        for p in self.Data:
            if p['id'] not in self.haslabelP:
                continue
#            print(p["id"])
            keys=p.keys()
            f=open(txtdir+name+"/"+str(p["id"])+".txt",'w',encoding='utf-8') 
            if 'title' in keys:
                f.write(str(p['title'])+'\n')
            if 'keywords' in keys:
                f.write(str(p['keywords'])+'\n')
            if 'abstract' in keys:
                f.write(str(p['abstract'])+'\n')
            f.close       
  

