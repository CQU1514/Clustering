# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 10:44:30 2017

@author: xiadong
"""

import matplotlib.pyplot as plt
import pandas as pd;
from datetime import datetime;
import numpy as np
from sklearn.cluster import DBSCAN
import math

def judge_neibour(data0,data1):
    if abs(data0[0]-data1[0])!=1:
        return False
    else:
        shu0=data0[1].split(",")
        shu1=data1[1].split(",")
        sum_shu=0
        for i in xrange(len(shu0)):
            sum_shu+=abs(int(shu0[i])-int(shu1[i]))
        if sum_shu==1:
            return True
        else:
            return False


def get_cluster(grid_set):
    grid_cluster={}
    mycluster_area={}
    i=0
    for ind0 in xrange(0,len(grid_set)):
        temp1=grid_set[ind0]
        res=[]
        res.append(temp1)
        for ind1 in xrange(ind0+1,len(grid_set)):            
            temp2=grid_set[ind1]
            if temp2[0]-temp1[0]>1:
                break
            else:
                if judge_neibour(temp1,temp2):
                    res.append(temp2)
        id_set=[]
        for grid_temp in res:
            if grid_cluster.has_key(grid_temp):
                if grid_cluster[grid_temp] not in id_set:
                    id_set.append(grid_cluster[grid_temp])
        if len(id_set)==0:
            myid=i
            i+=1
            mycluster_area[myid]=res
            for grid_temp in res:
                grid_cluster[grid_temp]=myid       
        else:
            myid=min(id_set)
            for id_temp in id_set:
                res+=mycluster_area.pop(id_temp)
            res=list(set(res))
            mycluster_area[myid]=res
            for grid_temp in res:
                grid_cluster[grid_temp]=myid
    return mycluster_area,grid_cluster


class GRIDBSCAN():
    def __init__(self, gridsize=0.004, min_samples=150):                
        self.gridsize =gridsize
        self.min_samples=min_samples


    def fit(self,X):
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        min_value=[]
        for i in xrange(self.n_features):
            min_value.append(min(X[:,i]))
        self.min_v=min_value
        index=[]
        dic_grid={}
        for i in xrange(self.n_samples):
            a0=0
            a1=[]
            for j in xrange(self.n_features):
                index_temp=int((X[i,j]-min_value[j])/self.gridsize)
                a0+=index_temp
                a1.append(str(index_temp))
            r=(a0,",".join(a1))
            index.append(r)
            if dic_grid.has_key(r):
                dic_grid[r]=dic_grid[r]+1
            else:
                dic_grid[r]=1
        hot_grid=[]
        self.index=index
        grid_set=dic_grid.keys()
        for temp in grid_set:
            if dic_grid[temp]>=self.min_samples:
                hot_grid.append(temp)
        self.hot_grid=hot_grid
        hot_grid.sort()
        self.cluster_set,grid_label=get_cluster(hot_grid)
        labels=[]
        for temp in index:
            if grid_label.has_key(temp):
                labels.append(grid_label[temp])
            else:
                labels.append(-1)
        self.labels_=labels
