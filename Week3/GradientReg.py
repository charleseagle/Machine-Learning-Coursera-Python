from __future__ import division
import numpy as np
import Sigmoid

def GradientReg(theta, X, y, Lambda):
    m = np.shape(y)[0]
    
    grad = np.zeros((np.shape(theta)))
    
    grad =(1/m)*(X.T.dot(np.transpose((Sigmoid.Sigmoid(X.dot(np.append(\
            0,theta[1:].ravel())))).T-y))) +\
            (Lambda/m)*theta
    
    return grad
