# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 23:09:39 2015

@author: Charleseagle
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import PlotData, SvmPredict


def VisualizeBoundary(X, y, model):
    PlotData.PlotData(X,y)
    x1plot = np.linspace(np.min(X[:,0]), np.max(X[:,0]), 100).T
    x2plot = np.linspace(np.min(X[:,1]), np.max(X[:,1]), 100).T
    X1, X2 = np.meshgrid(x1plot, x2plot)
    vals = np.zeros((np.shape(X1)))
    for i in range(np.shape(X1)[1]):
        this_X = np.append(X1[:,i].reshape(np.shape(X1)[0],1), \
                X2[:,i].reshape(np.shape(X2)[0],1), 1)
        vals[:,i] = SvmPredict.SvmPredict(model, this_X).flatten()
    plt.contour(X1, X2, vals, color = 'k')