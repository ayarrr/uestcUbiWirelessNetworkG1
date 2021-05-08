import Myparser
import embedding
import numpy as np
import json
import pandas as pd
import sampler
import trainer
import labelreader
import eval_
#当前看那个有歧义的人的文章
def main(path,name,labelpath):
    
    """
    pipeline for representation learning for all papers for a given name reference
    """
    with open(path,'r', encoding='gbk') as f:
        temp = json.loads(f.read())[name]
        label=labelreader.labelR(name,labelpath)#读label
        label.readlabel()
        name=' '.join([l.capitalize() for l in name.split('_')])
#        print(len(temp),name)
        ll = np.array(label.detail)
        ll=list(np.hstack(ll.flat))
        dataset = Myparser.DataSet(temp,name,ll)#先读一个人的数据
        
       
        dataset.reader_struct()
        optimizer = embedding.BprOptimizer(20, 0.2,0.02,0.2) 
        #以下为所考虑的所有元路径
        PAP_sampler = sampler.PAPSampler()#基于PAP路径的采样
        PVP_sampler = sampler.PVPSampler()#基于PVP路径的采样
        POP_sampler = sampler.POPSampler()#基于POP路径的采样
        APA_sampler = sampler.APASampler()#基于APA路径的采样
        OPO_sampler = sampler.OPOSampler()#基于OPO路径的采样
        PAPAP_sampler = sampler.PAPAPSampler()#基于PAPAP路径的采样
        PVPAP_sampler = sampler.PVPAPSampler()#基于PVPAP路径的采样
        run_helper = trainer.TrainHelper()
        evaluator=eval_.Evaluator(label)
        #开始训练更新embedding
        run_helper.helper(30,dataset,optimizer,evaluator,
                          metapathsample_dict={
                                  "PAP":PAP_sampler,
                                  "PVP":PVP_sampler,
                                  "POP":POP_sampler,
                                  "APA":APA_sampler,
                                  "OPO":OPO_sampler,
                                  "PAPAP":PAPAP_sampler,
                                  "PVPAP":PVPAP_sampler})


    return
labelpath="../assignment_validate.json"
main("../pubs_validate.json","bing_chen",labelpath)
#for num in range(0,180,6):
#    main("../pubs_validate.json","bing_chen",labelpath,num)
#    print(num/1000)