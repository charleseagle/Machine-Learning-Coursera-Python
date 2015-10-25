from __future__ import division
import numpy as np
import matplotlib.pyplot as plt


def PlotData(X, y):
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    
    plt.plot(X[pos,0], X[pos,1], 'x', linewidth = 1, color = 'red')
    plt.plot(X[neg,0], X[neg,1], 'o', linewidth = 1, color = 'red')
