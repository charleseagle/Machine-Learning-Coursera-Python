# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 02:17:18 2015

@author: Charleseagle
"""
from __future__ import division
import numpy as np


def CofiCost(params, Y, R, num_users, num_movies, num_features, Lambda):
    X = params[0:num_movies*num_features].reshape(num_movies, num_features)
    theta = params[num_movies*num_features:].reshape(num_users, num_features)
    J = 0
        
    J = 1/2*np.sum((X.dot(theta.T)*R-Y*R)**2) + \
            Lambda/2*((np.sum(theta**2)) + np.sum(X**2))
    return J