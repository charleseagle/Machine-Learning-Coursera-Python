from __future__ import division
import numpy as np
import scipy.optimize as opt

import LinearRegCost, LinearRegGradient

def TrainLinearReg(X, y, Lambda):
    initial_theta = np.zeros((np.shape(X)[1],1))
    
    
    result = opt.minimize(fun = LinearRegCost.LinearRegCost, x0 = initial_theta ,\
         args = (X, y, Lambda), method = 'Powell', jac =  \
         LinearRegGradient.LinearRegGradient, options = {'maxiter':1000})  
    theta = result.x
    
#    result = opt.fmin_cg(LinearRegCost.LinearRegCost, \
#            fprime = LinearRegGradient.LinearRegGradient, x0 = initial_theta, 
#		args = (X, y, Lambda), maxiter = 200, disp = True, full_output = True )
#
#    theta = result[1]
    return theta
        
            
            
            