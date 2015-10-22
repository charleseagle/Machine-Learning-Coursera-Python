from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import PolyFeatures

def PlotFit(min_x, max_x, mu, sigma, theta, p):
    x = np.arange(min_x - 2, max_x + 5.05, 0.05)
    x = x.reshape(len(x),1)
    X_poly = PolyFeatures.PolyFeatures(x, p)
    X_poly = X_poly - mu
    X_poly = X_poly/sigma
    X_poly = np.append(np.ones((np.shape(x)[0],1)), X_poly, 1)
    
    plt.plot(x, X_poly.dot(theta), '--', linewidth =2, color = 'blue')
    
