# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 19:00:17 2021

@author: Utilisateur
"""

import numpy as np
from os import listdir
from datetime import date
from distutils.dir_util import copy_tree

def create_checkpoint():
    today = str(date.today())
    print("Today's date:", today)
    
    savepath= '../../checkpoints'
    checkpoints=listdir(savepath)
    print("Number of versions :", len(checkpoints))
    
    if len(checkpoints)==0:
        copy_tree('.', savepath+'/'+today)
    else:
        last_version=checkpoints[len(checkpoints)-1]
        if last_version != today:
            copy_tree('.', savepath+'/'+today)
    
def get_closest_factors(x): #Given c, find a, b such that a*b=c and return the closest a,b each others
    l=[]
    if int(np.sqrt(x)) == np.sqrt(x):
        return int(np.sqrt(x))
    else:
        for i in range(1, x + 1):
            if x % i == 0:
                l.append(i)
        if len(l) % 2 != 0:
            return l[int(len(l)/2)+1]
        else:
            return l[int(len(l)/2)]
    
def hexa_repartition(h, w):
    res=np.zeros(2)
    res[0]=int(h[0],w)*w+w/2
    res[1]=int(h[1],w)*w+w/2
    return res