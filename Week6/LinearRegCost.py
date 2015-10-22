from __future__ import division
import numpy as np

def LinearRegCost(theta, X, y, Lambda):
    m = np.shape(X)[0]
    theta = theta.reshape(np.shape(X)[1],1)
    
    J = 0
    
    J = 1/(2*m)*np.sum((((X.dot(theta))-y)**2)) + \
            Lambda/(2*m)*np.sum(theta[1:,0].flatten()**2)
    
    return J
    