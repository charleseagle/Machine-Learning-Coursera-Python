# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 02:04:00 2015

@author: Charleseagle
"""
from __future__ import division
import numpy as np

def NormalizeRatings(Y, R):
    m, n = np.shape(Y)
    Ymean = np.zeros((m,1))
    Ynorm = np.zeros((m,n))
    for i in range(m):
        idx = np.where(R[i,:] == 1)
        Ymean[i] = np.mean(Y[i,idx])
        Ynorm[i,idx] = Y[i,idx] - Ymean[i]
    return [Ynorm, Ymean]