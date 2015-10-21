from __future__ import division
import numpy as np
import computeCost

def gradientDescentMulti(X, y, theta, alpha, num_iters):
    m = np.shape(y)[0]
    J_history = np.zeros(num_iters)
    for iter in range(num_iters):
        theta = theta - (alpha/m)*X.T*(X*theta - y)
        J_history[iter] = computeCost.computeCost(X, y, theta)

    return [theta, J_history]
    
    
