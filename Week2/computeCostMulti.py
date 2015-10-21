from __future__ import division
import numpy as np
# Compute cost for linear regression with multiple variables
def computeCostMulti(X, y, theta):
    m = np.shape(y)[0]  #number of tranning examples
    J = 0
    J = 1/(2*m)*((X*theta) - y).T*((X*theta) - y)
    return J
