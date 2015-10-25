# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 12:49:53 2015

@author: Charleseagle
"""
from __future__ import division
import numpy as np

class Functions(object):
#    def __init__(self, x1, x2):
#        self.x1 = x1
#        self.x2 = x2
    def GaussianKernel(self, x1, x2, sigma):
        # Ensure that x1 and x2 are column vectors
        x1 = x1
        x2 = x2
        # Ensure that x1 and x2 are column vectors
        sim =  np.exp(-np.sum((x1 - x2)**2)/(2*sigma**2))
        return sim
    def LinearKernel(self, x1, x2):
        # Ensure that x1 and x2 are column vectors
        x1 = x1.reshape(np.shape(x1)[0],1)
        x2 = x2.reshape(np.shape(x2)[0],1)
        # Ensure that x1 and x2 are column vectors
        sim = x1.T.dot(x2)
        return sim