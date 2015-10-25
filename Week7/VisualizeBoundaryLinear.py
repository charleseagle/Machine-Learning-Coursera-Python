# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 13:46:52 2015

@author: Charleseagle
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import PlotData

def VisualizeBoundaryLinear(X, y, model):
    w = model['w']
    b = model['b']
    xp = np.linspace(np.min(X[:,0]), np.max(X[:,0]), 100)
    yp = -(w[0]*xp + b)/w[1]
    

    PlotData.PlotData(X, y)
    plt.plot(xp, yp, '-', color = 'b')
   