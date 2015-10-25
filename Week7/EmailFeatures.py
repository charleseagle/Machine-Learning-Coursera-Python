# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 13:55:10 2015

@author: Charleseagle
"""
from __future__ import division
import numpy as np
 

def EmailFeatures(word_indices):
    n =1899
    x = np.zeros((n,1))
    for j in range(len(word_indices)):
        for i in range(n):
            if i == word_indices[j]-1:
                x[i] = 1
    return x