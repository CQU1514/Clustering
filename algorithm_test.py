# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 13:40:08 2017

@author: xiadong
"""

import sys
import scipy
import pickle
import math
import OPTICS
import denclue
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import time
import GRIDBSCAN

from sklearn.neighbors import BallTree

from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

volume=200000

v_set=[]
G_set=[]
De_set=[]
OP_set=[]
DB_set=[]
while volume<1000000:
    v_set.append(volume)
    if volume<=500:
        m_sample=1
    elif volume<=100000:
        m_sample=int(math.ceil(volume/500.0)-1)
    else:
        m_sample=200
    
    centers = [[1, 1], [-1, -1], [1, -1]]
    X, labels_true = make_blobs(n_samples=volume, centers=centers, cluster_std=0.4,
                                random_state=1)

#    X = StandardScaler().fit_transform(X)
#    plt.figure()
#    plt.scatter(X[:,0],X[:,1])
#    plt.show()

    #GRIDBSCAN
    start=time.time()
    GDB=GRIDBSCAN.GRIDBSCAN(gridsize=0.08,min_samples=m_sample)
    GDB.fit(X)
    time_d=round(time.time()-start,3)
    G_set.append(time_d)
    print 'GRIDBSCAN程序耗时:%s',time_d; 

#    #Denclue
#    start=time.time()
#    dc=denclue.DENCLUE(eps=0.01)
#    dc.fit(X)
#    time_d=round(time.time()-start,3)
#    De_set.append(time_d)
#    print 'Denclue程序耗时:%s',time_d;

#    #OPTICS
#    start=time.time()
#    testtree = OPTICS.setOfObjects(X)
#    OPTICS.prep_optics(testtree,m_sample*2,m_sample)
#    OPTICS.build_optics(testtree,m_sample*2,m_sample,'./testing_may6.txt')
#    OPTICS.ExtractDBSCAN(testtree,0.2)
#    time_d=round(time.time()-start,3)
#    OP_set.append(time_d)
#    print 'OPTICS程序耗时:%s',time_d;
    
    #DBSCAN
    start=time.time()
    db = DBSCAN(eps=0.04, min_samples=m_sample).fit(X)
    time_d=round(time.time()-start,3)
    DB_set.append(time_d)
    print 'DBSCAN程序耗时:%s',time_d; 
     
    volume+=50000