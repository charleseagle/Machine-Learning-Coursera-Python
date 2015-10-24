from __future__ import division
import numpy as np


def MultivariateGaussian(X, mu, sigma2):
    k = len(mu)
    if np.shape(sigma2.reshape(len(sigma2),1))[1] == 1 or \
            np.shape(sigma2.reshape(len(sigma2),1))[0] == 1:
        sigma2 = np.diag(sigma2)
    X = X - mu
    p = (2*np.pi)**(-k/2)*np.linalg.det(sigma2)**(-0.5)*np.exp(\
        -0.5*((X.dot(np.linalg.pinv(sigma2)))*X).sum(axis=1))
    return p