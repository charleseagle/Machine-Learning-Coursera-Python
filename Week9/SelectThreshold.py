# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 15:43:50 2015

@author: Charleseagle
"""
from __future__ import division
import numpy as np


def SelectThreshold(yval, pval):
    bestEpsilon = 0
    bestF1 = 0
    F1 = 0
    
    #stepsize = (pval.max(axis=0) - pval.min(axis=0))/1000
    stepsize = (np.max(pval) - np.min(pval))/1000
    for epsilon in np.arange(np.min(pval), np.max(pval), stepsize):
        predictions = pval < epsilon
        predictions = predictions + 0
        tp = np.sum((predictions == 1)+0 & (yval == 1)+0)
        fp = np.sum((predictions == 1)+0 & (yval == 0)+0)
        fn = np.sum((predictions == 1)+0 & (yval == 0)+0)
        prec = tp/(tp+fp)
        rec = tp/(tp+fn)
        F1 = 2*prec*rec/(prec+rec)
        if F1 > bestF1:
            bestF1 = F1
            bestEpsilon = epsilon
    return [bestEpsilon, bestF1]