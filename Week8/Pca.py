from __future__ import division
import numpy as np

def Pca(X):
    m, n = np.shape(X)
    U = np.zeros(n)
    S = np.zeros(n)
    sigma = 1/m*(X.T.dot(X))
    U, S, V = np.linalg.svd(sigma)
    return [U, S]