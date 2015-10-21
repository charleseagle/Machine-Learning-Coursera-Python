from __future__ import division
import numpy as np

def featureNormalize(X):
    X_norm = X
    mu = np.matrix(np.zeros((1, np.shape(X)[1])))
    sigma = np.matrix(np.zeros((1, np.shape(X)[1])))
    for i in range(np.shape(X)[1]):
        mu[0,i] = np.mean(X[:,i])
        for j in range(np.shape(X)[0]):
            X_norm[j,i] = X[j,i] - mu[0,i]
    for i in range(np.shape(X)[1]):
        sigma[0,i] = np.std(X[:,i])
        for j in range(np.shape(X)[0]):
            X_norm[j,i] = X[j,i]/sigma[0,i]
    X = X_norm
    return [X_norm, mu, sigma]
   