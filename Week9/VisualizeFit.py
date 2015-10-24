# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 12:47:25 2015

@author: Charleseagle
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import MultivariateGaussian

def VisualizeFit(X, mu, sigma2):
    X1, X2 = np.meshgrid(np.arange(0,35.5,0.5),np.arange(0,35.5,0.5))
    m, n = np.shape(X1)
    Z = MultivariateGaussian.MultivariateGaussian(np.append(X2.flatten()\
    .reshape(m*n,1), X1.flatten().reshape(m*n,1), 1), mu, sigma2)
    Z = Z.reshape(np.shape(X1))
    
    
    plt.plot(X[:,0], X[:,1], 'x', color = 'r')
    if np.sum(np.isinf(Z)) == 0:
        plt.contour(X2, X1, Z)
    