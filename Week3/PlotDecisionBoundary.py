from __future__ import division
import matplotlib.pylab as plt
import numpy as np
import PlotData, MapFeature

def PlotDecisionBoundary(theta, X, y):
    
    if np.shape(X)[1] <= 3:
        # 2 points to define a line, so choose two endpoints
    
        plot_x = np.zeros(2)
        plot_x[0] = np.min(X[:,1])-2
        plot_x[1] = np.max(X[:,1])+2
        
        # Calculate the decision boundary line
        plot_y = (-1/theta[2])*(theta[1]*plot_x + theta[0])
        
        plt.figure()
        PlotData.PlotData(X[:,1:3], y)
        plt.plot(plot_x, plot_y, label = 'Decision Boundary')
        plt.legend(bbox_to_anchor=(0.4, 1.35), loc = 1)
        plt.xlim(30, 100)
        plt.ylim(30, 100)
        plt.show()
    else:
        # grid range
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        
        z = np.zeros((len(u), len(v)))
        for i in range(len(u)):
            for j in range(len(v)):
                z[i,j] = MapFeature.MapFeature(np.array([u[i]]), np.array([v[j]])).dot(theta)
        z = z.T
        plt.figure()
        PlotData.PlotData(X[:,1:3], y)
        cs = plt.contour(u, v, z, levels = [0.0, 0.0], colors = 'k')
       
        cs.collections[0].set_label('Decision Boundary')
        plt.legend(bbox_to_anchor=(0.4, 1.35), loc = 1)
#        plt.xlim(30, 100)
#        plt.ylim(30, 100)
        plt.show()
        
        
    
   
