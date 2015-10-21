from __future__ import division
import numpy as np
import Sigmoid



def LrGradient(theta, X, y, Lambda):
    m = len(y)
    
    grad = np.zeros((np.shape(theta)))
    
    
    grad = (1/m)*(X.T.dot(Sigmoid.Sigmoid(X.dot(theta))-y)) + \
            (Lambda/m)*np.append(0,theta[1:].ravel()).reshape(np.shape(theta))
    
    return grad
