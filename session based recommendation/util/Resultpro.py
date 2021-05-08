# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 17:23:21 2019

@author: Administrator
"""

import numpy as np
import pandas as pd


def cal_result(preds, labels, cut_off=20):
    """
    cut_off专门用于计算ReCall@5
    """
    recall = []
    mrr = []
    rank_l = []
    request_extra=[]
    for pred, b_label in zip(preds, labels):
        crank = list(np.argsort(pred)[::-1]).index(b_label) + 1
        rank_l.append(crank)
        recall.append(crank <= cut_off)
        if crank >cut_off:
            request_extra.append(1)
        else:
            request_extra.append(0)
            
        mrr.append(1 / float(crank))

    return recall, mrr, rank_l,request_extra
def cal_result1(preds,labels,cut_off=20):#输入preds,labels都为batch,25
    """
    cut_off专门用于计算ReCall@5
    """
    recall = []
    mrr = []
    rank_l = []
    for pred, b_label in zip(preds,labels):
        rellClick_item=list(b_label).index(1)
        crank=list(np.argsort(pred)[::-1]).index(rellClick_item)+1
        rank_l.append(crank)
        recall.append(crank <= cut_off)
        mrr.append(1/float(crank))
        
    return recall, mrr, rank_l