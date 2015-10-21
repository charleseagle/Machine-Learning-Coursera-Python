from __future__ import division
import numpy as np
import Sigmoid

def CostFunctionReg(theta, X, y, Lambda):
    m = np.shape(y)[0]
    J = 0
    grad = np.zeros((np.shape(theta)))
    
    J = -(1/m)*(np.log(Sigmoid.Sigmoid(X.dot(theta))).T.dot(y) + \
        np.log(1-Sigmoid.Sigmoid(X.dot(theta))).T.dot((1-y))) +\
        (Lambda/(2*m))*np.sum(theta[1:np.shape(theta)[0]].ravel().T\
        .dot(theta[1:].ravel()))
    
    grad =(1/m)*(X.T.dot(np.transpose((Sigmoid.Sigmoid(X.dot(np.append(\
            0,theta[1:].ravel())))).T-y))) +\
            (Lambda/m)*theta
    
    return [J, grad]