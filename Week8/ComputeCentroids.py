from __future__ import division
import numpy as np



def ComputeCentroids(X, idx, K):
    m, n = np.shape(X)
    centroids = np.zeros((K,n))
    
    for j in range(K):
        t = 0
        for i in range(m):
            if idx[i]-1 == j:
                t = t+1
                centroids[j,:] = centroids[j,:] + X[i,:]
        centroids[j,:] = centroids[j,:]/t
    return centroids