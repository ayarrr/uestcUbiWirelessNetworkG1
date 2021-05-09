# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 14:46:00 2019

@author: Administrator
"""
import numpy as np
import json
import pandas as pd
class labelR():
    def __init__(self,name,path):
        self.path=path
        self.name=name
        
    def readlabel(self):
        with open(self.path) as f:
            temp = json.loads(f.read())
            clasify=temp[self.name]
            self.n_clusfer=len(clasify)
            self.detail=clasify
            return 
        
