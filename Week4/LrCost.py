from __future__ import division
import numpy as np
import Sigmoid



def LrCost(theta, X, y, Lambda):
    m = len(y)
    J = 0
#    grad = np.zeros((np.shape(theta)))
    
    J = -(1/m)*(np.log(Sigmoid.Sigmoid(X.dot(theta))).T.dot(y) + \
            np.log(1-Sigmoid.Sigmoid(X.dot(theta))).T.dot(1-y)) \
            + (Lambda/(2*m))*sum(theta[1:]**2)
#    grad = (1/m)*(X.T.dot(Sigmoid.Sigmoid(X.dot(theta))-y)) + \
#            (Lambda/m)*np.insert(0,theta[1:np.shape(theta)[0]].ravel(),0)
    
    return J
