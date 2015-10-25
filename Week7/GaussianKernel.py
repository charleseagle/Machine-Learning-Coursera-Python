from __future__ import division
import numpy as np

def GaussianKernel(x1, x2, sigma):
    # Ensure that x1 and x2 are column vectors
    x1 = x1
    x2 = x2
    # Ensure that x1 and x2 are column vectors
    sim =  np.exp(-np.sum((x1 - x2)**2)/(2*sigma**2))
    return sim
    