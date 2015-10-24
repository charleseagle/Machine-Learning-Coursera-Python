# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 21:12:03 2015

@author: Charleseagle
"""
from __future__ import division
import numpy as np


def CofiCostFunc(params, Y, R, num_users, num_movies, num_features, Lambda):
    X = params[0:num_movies*num_features].reshape(num_movies, num_features)
    theta = params[num_movies*num_features:].reshape(num_users, num_features)
    J = 0
    X_grad = np.zeros((np.shape(X)))
    theta_grad = np.zeros((np.shape(theta)))
    
    J = 1/2*np.sum((X.dot(theta.T)*R-Y*R)**2) + \
            Lambda/2*((np.sum(theta**2)) + np.sum(X**2))
    X_grad = (X.dot(theta.T)*R-Y*R).dot(theta) + Lambda*X
    theta_grad = (X.dot(theta.T)*R-Y*R).T.dot(X) + Lambda*theta
    
    grad = np.append(X_grad.flatten(), theta_grad.flatten(), 1)
    return [J, grad]