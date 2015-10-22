from __future__ import division
import numpy as np

def LinearRegCostFunction(X, y, theta, Lambda):
    m = np.shape(X)[0]
    theta = theta.reshape(np.shape(X)[1],1)
    J = 0
    grad = np.zeros((np.shape(theta)))
    J = 1/(2*m)*np.sum((((X.dot(theta))-y)**2)) + \
            Lambda/(2*m)*np.sum(theta[1:,0].flatten()**2)
    grad = 1/m*(X.T.dot((X.dot(theta)-y))) + \
        Lambda/m*np.append(0,theta[1:,0].flatten()).reshape(np.shape(X)[1],1)
    
    return [J, grad]
    

