from __future__ import division
import numpy as np

def RandInitializeWeights(L_in, L_out):
    W = np.zeros((L_out, 1+L_in))
    epsilon_init = 0.12
    W = np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init
    return W
    

