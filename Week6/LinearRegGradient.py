from __future__ import division
import numpy as np

def LinearRegGradient(theta, X, y, Lambda):
    
    theta = theta.reshape(np.shape(X)[1],1)
    m = np.shape(X)[0]
    
    grad = np.zeros((np.shape(X)[1],1))
    
    grad = 1/m*(X.T.dot((X.dot(theta)-y))) + \
        Lambda/m*np.append(0,theta[1:,0].flatten()).reshape(np.shape(X)[1],1)
    
    return grad
    