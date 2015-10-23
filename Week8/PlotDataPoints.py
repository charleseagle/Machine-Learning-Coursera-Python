from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import colorsys

def PlotDataPoints(X, idx, K):
    def to_rgb(x):
        return colorsys.hsv_to_rgb(*x)
    h = np.arange(K+1).reshape(K+1,1)/(K+1)
    h = np.append(h, np.ones((K+1, 2)), 1)
    palette = np.apply_along_axis(to_rgb, axis = 1, arr = h)
    
#    m = np.shape(idx)[0]
#    n = np.shape(palette)[1]
#    colors = np.zeros((m,n))
#    for i in range(m):
#        colors[i,:]= palette[idx[i]-1,:]
    colors= palette[idx.flatten()-1,:]
    
    plt.scatter(X[:,0], X[:,1], s= 15, c = colors)
    
    