# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 15:54:54 2024

@author: cdarc
"""

import time
import numpy as np
from less_squares import LessSquares
#import matplotlib.pyplot as plt

def speed_test():
    m = 100
    n = 500
    A = np.random.random(size=(m,n))
    model = LessSquares(A)
    axis = 0
    times = []
    E = []
    for k in range(400):
        u = np.random.random(size=(A.shape[((axis+1)%2)],1))
        u = A[0, :, np.newaxis] + 0.00001 * u
        so = time.time()
        A = np.vstack((A,u.T))
        np.linalg.pinv(A)
        sof = time.time()
        s1 = time.time()
        model.append(u,axis)
        s2 = time.time()
        times.append([s2-s1,sof-so])
        if k%10 == 9:
            #model.clean()
            #print(model.matrix-A)
            print(k)
            #e = np.linalg.norm(model.pinv-np.linalg.pinv(model.A))
            e = model.check(mode='full')
            E.append(e)
            print(e)
    t = np.cumsum(times,axis = 0)
    return t,E,model,A

t,E,model,A = speed_test()
#plt.plot(np.log(E))
#plt.plot(np.log(np.array([np.max(e) for e in E]))/np.log(10))
#plt.plot((-t[:-10,:]+t[10:,:])/10);plt.legend(['new','old'])
#plt.xlim(xmin=0.0)
#plt.ylim(ymin=0.0)
#plt.scatter(np.linalg.norm(np.linalg.pinv(A)-model.pseudo,axis=1),model.norms)