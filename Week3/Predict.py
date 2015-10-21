from __future__ import division
import numpy as np
import Sigmoid

def Predict(theta, X):
    p = theta.dot(X.T)
    p = np.where(p >= 0.5, 1, 0)
    
#    m = np.shape(X)[0]
#    p = np.zeros((m,1))
#    
#    for i in range(m):
#        a = 0
#        for j in range(np.shape(theta)[0]):
#            a = a + theta[j]*X[i,j]
#        if Sigmoid.Sigmoid(a) >= 0.5:
#            p[i] = 1;
#        else:
#            p[i] = 0

    return p
