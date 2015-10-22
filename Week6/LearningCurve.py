from __future__ import division
import numpy as np
import TrainLinearReg, LinearRegCostFunction

def LearningCurve(X, y, Xval, yval, Lambda):
    m = np.shape(X)[0]
    #n = np.shape(Xval)[0]
    
    error_train = np.zeros((m,1))
    error_val = np.zeros((m,1))
    
    for i in range(m):
        theta = TrainLinearReg.TrainLinearReg(X[0:i+1,:], y[0:i+1,:], \
                Lambda).reshape(np.shape(X)[1],1)
        error_train[i,0], a = LinearRegCostFunction.LinearRegCostFunction(\
                     X[0:i+1,:], y[0:i+1,:], theta, Lambda)
        error_val[i,0], b = LinearRegCostFunction.LinearRegCostFunction(\
                     Xval, yval, theta, Lambda)
    return [error_train, error_val]