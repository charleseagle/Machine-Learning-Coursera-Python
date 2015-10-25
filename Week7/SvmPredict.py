# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 23:19:56 2015

@author: Charleseagle
"""
from __future__ import division
import numpy as np


def SvmPredict(model, X):
    m, n = np.shape(X)
    
    if n == 1:
        X = X.reshape(m,1)
    p = np.zeros((m,1))
    pred = np.zeros((m,1))
    
    if model['KernelFunction'].__name__ == 'LinearKernel':
        p = X.dot(model['w']) + model['b']
    elif model['KernelFunction'].__name__ == 'GaussianKernel':
        X1 = (X**2).sum(axis = 1)
        X2 = (model['X']**2).sum(axis = 1).T
        K = X1.reshape(len(X1),1) + (X2 - 2* X.dot(model['X'].T))
        K = model['KernelFunction'](1, 0)**K
        K = model['y'].T * K
        K = model['alphas'].T * K
        p = K.sum(axis = 1)
    else:
        for i in range(m):
            prediction = 0
            for j in range(np.shape(model['X'])[0]):
                prediction = prediction + model['alphas'][j]*model['y'][j]\
                    *model['KernelFunction'](X[i,:],model['X'][j,:])
            p[i] = prediction + model['b']
    pred[p >= 0] = 1
    pred[p < 0] = 0
    return pred
    
    