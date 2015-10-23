from __future__ import division
import numpy as np


def ProjectData(X, U, K):
    Z = np.zeros((np.shape(X)[0],K))
    Z = X.dot(U[:,0:K])
    return Z