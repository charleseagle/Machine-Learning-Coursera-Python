from __future__ import division
import numpy as np

def PolyFeatures(X, p):
    m = np.shape(X.flatten())[0]
    X_poly = np.zeros((m,p))
    
    for i in range(p):
        X_poly[:,i] = X[:,0]**(i+1)
      
    return X_poly
