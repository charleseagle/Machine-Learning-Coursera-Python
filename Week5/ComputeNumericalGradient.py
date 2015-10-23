from __future__ import division
import numpy as np

def ComputeNumericalGradient(J, theta):
    numgrad = np.zeros((np.shape(theta)))
    perturb = np.zeros((np.shape(theta)))
    e = 1e-4
    
    for p in range(len(theta.flatten())):
        perturb[p] = e
        loss1, grad1 = J(theta - perturb)
        loss2, grad2 = J(theta + perturb)
        
        numgrad[p] = (loss2 - loss1)/(2*e)
        perturb[p] = 0
    return numgrad
