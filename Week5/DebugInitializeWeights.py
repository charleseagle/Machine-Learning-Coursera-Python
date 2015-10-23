from __future__ import division
import numpy as np


def DebugInitializeWeights(fan_out, fan_in):
    W = np.zeros((fan_out, 1 + fan_in))
    W = np.sin(np.arange(len(W.flatten()))+1).reshape(np.shape(W))/10
    
    return W