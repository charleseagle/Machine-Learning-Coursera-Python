from __future__ import division
import numpy as np

def normalEqn(X, y):
    theta = np.matrix(np.zeros((np.shape(X)[1],1)))
    theta = np.linalg.inv((X.T*X))*X.T*y
    
    return theta