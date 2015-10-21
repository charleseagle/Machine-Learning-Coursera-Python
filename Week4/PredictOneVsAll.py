from __future__ import division
import numpy as np
import Sigmoid

def PredictOneVsAll(all_theta, X):
    m = np.shape(X)[0]
    #num_labels = np.shape(all_theta)[0]
    
    p = np.zeros((np.shape(X)[0],1))
    X = np.append(np.ones((m,1)), X, 1)
    a = Sigmoid.Sigmoid(X.dot(all_theta.T))
    for i in range(np.shape(a)[0]):
        p[i] = np.where(a == a[i,:].max())[1]
    return p + 1
        
    
    
