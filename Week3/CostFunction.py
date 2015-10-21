from __future__ import division
import numpy as np
import Sigmoid

def CostFunction(theta, X, y):
    m = np.shape(y)[0]
    J = 0
    grad = np.zeros((np.shape(theta)))
    
    J = -(1/m)*(np.log(Sigmoid.Sigmoid(X.dot(theta))).T.dot(y) + \
        np.log(1-Sigmoid.Sigmoid(X.dot(theta))).T.dot((1-y)))
        
    grad =(1/m)*(X.T.dot(np.transpose((Sigmoid.Sigmoid(X.dot(theta))).T-y)))
    
    return [J, grad]