from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import FindClosestCentroids, PlotProgresskMeans, ComputeCentroids

def RunkMeans(X, initial_centroids, max_iters, plot_progress = None):
    if plot_progress == None:
        plot_progress = False
    if plot_progress is True:
        plt.figure(1)
    
    m, n = np.shape(X)
    K = np.shape(initial_centroids)[0]
    centroids = initial_centroids
    previous_centroids = centroids
    idx = np.zeros((m,1))
    
    for i in range(max_iters):
        print 'K-Means iteration ', i, max_iters
        idx = FindClosestCentroids.FindClosestCentroids(X, centroids)
        
        if plot_progress is True:
            
            PlotProgresskMeans.PlotProgresskMeans(X, centroids, \
                    previous_centroids, idx, K, i)
            previous_centroids = centroids
            
            plt.show()
            print "'Program paused. Press enter to continue."
            raw_input("Press ENTER to continue")
        centroids = ComputeCentroids.ComputeCentroids(X, idx, K)
    return [centroids, idx]