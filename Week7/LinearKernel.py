from __future__ import division
import numpy as np

def LinearKernel(x1, x2):
    # Ensure that x1 and x2 are column vectors
    x1 = x1.reshape(np.shape(x1)[0],1)
    x2 = x2.reshape(np.shape(x2)[0],1)
    # Ensure that x1 and x2 are column vectors
    sim = x1.T.dot(x2)
    return sim
    
    
