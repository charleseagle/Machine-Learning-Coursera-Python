from __future__ import division
import numpy as np
import Sigmoid

def Predict(theta1, theta2, X):
    m = np.shape(X)[0]
    #num_labels = np.shape(theta2)[0]
    p = np.zeros((m,1))
    
    h1 = Sigmoid.Sigmoid(np.append(np.ones((m,1)), X, 1).dot(theta1.T))
    h2 = Sigmoid.Sigmoid(np.append(np.ones((m,1)), h1, 1).dot(theta2.T))
    for i in range(np.shape(h2)[0]):
        p[i] = np.where(h2 == h2[i,:].max())[1]
        
    return p + 1