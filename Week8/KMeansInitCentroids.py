from __future__ import division
import numpy as np


def KMeansInitCentroids(X, K):
    centroids = np.zeros((K, np.shape(X)[1]))
    randidx = np.random.permutation(np.shape(X)[0])
    centroids = X[randidx[0:K], :]
    return centroids