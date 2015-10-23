from __future__ import division
import numpy as np


def RecoverData(Z, U, K):
    X_rec = np.zeros((np.shape(Z)[0],np.shape(U)[0]))
    X_rec = Z.dot(U[:,0:K].T)
    return X_rec