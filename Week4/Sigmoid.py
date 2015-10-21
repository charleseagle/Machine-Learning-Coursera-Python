from __future__ import division
import numpy as np

def Sigmoid(z):
    g = np.zeros((np.shape(z)))
    
    g = 1/(1+np.exp(-z))
    
#    for i in range(np.shape(z)[0]):        
#        for j in range(np.shape(z)[1]):
#            g[i,j] = 1/(1+np.exp(-z[i,j]))
            
    return g


