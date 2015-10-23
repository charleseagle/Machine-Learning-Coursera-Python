from __future__ import division
#import numpy as np

def FeatureNormalize(X):
    mu = X.mean(axis = 0)
    X_norm = X - mu
    
    sigma = X_norm.std(axis = 0, ddof=1)
    X_norm = X_norm/sigma
    return [X_norm, mu, sigma]
