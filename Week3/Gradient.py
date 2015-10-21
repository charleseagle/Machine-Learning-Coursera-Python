from __future__ import division
import numpy as np
import Sigmoid

def Gradient(theta, X, y):
    m = np.shape(y)[0]
    
    grad = np.zeros((np.shape(theta)))
    
    
    grad =(1/m)*(X.T.dot(np.transpose((Sigmoid.Sigmoid(X.dot(theta))).T-y)))
    
    return grad
