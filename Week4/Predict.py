from __future__ import division
import numpy as np
import Sigmoid

def Predict(theta1, theta2, X):
    m = np.shape(X)[0]
    #num_labels = np.shape(theta2)[0]
    p = np.zeros((m,1))
    X = np.append(np.ones((m,1)), X, 1)
    A = Sigmoid.Sigmoid(X.dot(theta1.T))
    A = np.append(np.ones((m,1)), A, 1)
    B = Sigmoid.Sigmoid(A.dot(theta2.T))
    for i in range(np.shape(B)[0]):
        p[i] = np.where(B == B[i,:].max())[1]
    return p + 1