# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 22:44:19 2015

@author: Charleseagle
"""
from __future__ import division
import numpy as np
import ComputeNumericalGradient, CofiCostFunc

def CheckCostFunction(Lambda = None):
    if Lambda == None:
        Lambda = 0
    X_t = np.random.rand(4,3)
    theta_t = np.random.rand(5,3)
    
    Y = X_t.dot(theta_t.T)
    Y[np.random.rand(np.shape(Y)[0]) > 0.5] = 0
    R = np.zeros((np.shape(Y)))
    R[Y != 0] = 1
    
    m, n = np.shape(X_t)
    X = np.random.randn(m,n)
    a, b = np.shape(theta_t)
    theta = np.random.randn(a,b)
    num_users = np.shape(Y)[1]
    num_movies = np.shape(Y)[0]
    num_features = np.shape(theta_t)[1]
    def J(t):
        return CofiCostFunc.CofiCostFunc(t, Y, R, num_users, num_movies, \
                                num_features, Lambda)
     
    numgrad = ComputeNumericalGradient.ComputeNumericalGradient(J, \
            np.append(X.flatten(), theta.flatten(), 1))
    cost, grad = CofiCostFunc.CofiCostFunc(np.append(X.flatten(), \
            theta.flatten(), 1), Y, R, num_users, \
                          num_movies, num_features, Lambda)
    print numgrad, grad
    print 'The above two columns you get should be very similar.'
    print '(Left-Your Numerical Gradient, Right-Analytical Gradient)'
    diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)
    print 'If your backpropagation implementation is correct, then \
           the relative difference will be small (less than 1e-9).\
           Relative Difference: ', diff
   
                     
    
    