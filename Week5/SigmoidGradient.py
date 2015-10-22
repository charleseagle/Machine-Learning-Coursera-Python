from __future__ import division
import numpy as np


def SigmoidGradient(z):
    g = np.zeros(np.shape(z))
    g = 1/(1+np.exp(-z))*(1-1/(1+np.exp(-z)))
    return g
    
