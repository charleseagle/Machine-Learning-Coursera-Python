# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 02:19:39 2015

@author: Charleseagle
"""
from __future__ import division
import numpy as np


def CofiGradient(params, Y, R, num_users, num_movies, num_features, Lambda):
    X = params[0:num_movies*num_features].reshape(num_movies, num_features)
    theta = params[num_movies*num_features:].reshape(num_users, num_features)
    
    X_grad = np.zeros((np.shape(X)))
    theta_grad = np.zeros((np.shape(theta)))
    
    
    theta_grad = (X.dot(theta.T)*R-Y*R).T.dot(X) + Lambda*theta
    
    grad = np.append(X_grad.flatten(), theta_grad.flatten(), 1)
    return grad