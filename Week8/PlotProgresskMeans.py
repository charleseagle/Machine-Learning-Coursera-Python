from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import PlotDataPoints, DrawLine

def PlotProgresskMeans(X, centroids, previous, idx, K, i):
    PlotDataPoints.PlotDataPoints(X, idx, K)
    plt.plot(centroids[:,0], centroids[:,1], 'x', color = 'k', 
             linewidth = 15)
    for j in range(np.shape(centroids)[0]):
        DrawLine.DrawLine(centroids[j,:], previous[j,:])
    
    plt.suptitle('Iteration number')

