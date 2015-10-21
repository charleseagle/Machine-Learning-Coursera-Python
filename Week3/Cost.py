from __future__ import division
import numpy as np
import Sigmoid

def Cost(theta, X, y):
    m = np.shape(y)[0]
    J = 0
        
    J = -(1/m)*(np.log(Sigmoid.Sigmoid(X.dot(theta))).T.dot(y) + \
        np.log(1-Sigmoid.Sigmoid(X.dot(theta))).T.dot((1-y)))
        
       
    return J