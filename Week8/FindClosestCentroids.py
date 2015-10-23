from __future__ import division
import numpy as np

def FindClosestCentroids(X, centroids):
    K = np.shape(centroids)[0]
    m = np.shape(X)[0]
    idx = np.zeros((m,1))
    for i in range(m):
        temp = np.zeros((K,1))
        for j in range(K):
            temp[j,0] = np.sum((X[i,:]-centroids[j,:])**2)
        index = np.where(temp == temp.min())[0]
        idx[i] = index[0]
    idx = np.array(idx, dtype = 'i') + 1
    return idx

