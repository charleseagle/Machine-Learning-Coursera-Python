from __future__ import division
import numpy as np

def MapFeature(X1, X2):
    X1 = X1[:,np.newaxis]
    X2 = X2[:,np.newaxis]
    degree = 6
    out = np.ones((np.shape(X1)[0], (degree+1)*(degree+2)//2))
    col_num = 0
    for i in range(degree+1):
        for j in range(i+1):
            el = (X1**(i-j))*(X2**j)
            out[:,col_num] = el[:,0]
            col_num = col_num + 1
    return out
            