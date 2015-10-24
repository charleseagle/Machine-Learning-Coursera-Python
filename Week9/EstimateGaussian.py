from __future__ import division
import numpy as np

def EstimateGaussian(X):
    m, n = np.shape(X)
    mu = np.zeros((n,1))
    sigma2 = np.zeros((n,1))
    
    mu = 1/m*X.sum(axis=0)
    sigma2 = 1/m*((X-np.tile(mu,m).reshape(m,n))**2).sum(axis=0)
    
    return [mu, sigma2]