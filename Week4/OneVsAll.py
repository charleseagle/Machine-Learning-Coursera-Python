from __future__ import division
import numpy as np
import scipy.optimize as opt
import LrCost, LrGradient


def OneVsAll(X, y, num_labels, Lambda):
    m, n = np.shape(X)
    all_theta = np.zeros((num_labels, n+1))
    X = np.append(np.ones((m,1)), X, 1)
    
    for c in range(num_labels):
        initial_theta = np.zeros((n+1,1))
        # Run optimize to obtain the optimal theta
        result = opt.optimize.fmin_cg( LrCost.LrCost, fprime=LrGradient.LrGradient, x0=initial_theta.ravel(), \
	   args=(X, ((y == c+1) + 0).ravel(), Lambda), maxiter=50, disp=False, full_output=True )        
        all_theta[c,:] = result[0]
#        result = opt.minimize(fun = LrCost.LrCost, x0 = initial_theta ,\
#         args = (X, ((y == c+1) + 0), Lambda), method = 'TNC', jac =  \
#         LrGradient.LrGradient, options = {'gtol':1e-6, 'maxiter':1000})
#         
#        theta = result.x
#        all_theta[c,:] = theta
    
    return all_theta
        
