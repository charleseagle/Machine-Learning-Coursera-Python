from __future__ import division
import numpy as np
import TrainLinearReg, LinearRegCostFunction

def ValidationCurve(X, y, Xval, yval):
    lambda_vec = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])
    a = len(lambda_vec)
    lambda_vec = lambda_vec.reshape(a,1)
    error_train = np.zeros((a, 1))
    error_val = np.zeros((a, 1))
    
    m = np.shape(X)[0]
    n = np.shape(Xval)[0]
    X = np.append(np.ones((m,1)), X, 1)
    Xval = np.append(np.ones((n,1)), Xval, 1)
    
    for i in range(a):
        theta1 = TrainLinearReg.TrainLinearReg(X, y, lambda_vec[i,0])
        error_train[i], a = LinearRegCostFunction.LinearRegCostFunction(X, y, theta1, 0)
        theta2 = TrainLinearReg.TrainLinearReg(Xval, yval, lambda_vec[i,0])        
        error_val[i], a = LinearRegCostFunction.LinearRegCostFunction(Xval, yval, theta2, 0)
    return [lambda_vec, error_train, error_val]

