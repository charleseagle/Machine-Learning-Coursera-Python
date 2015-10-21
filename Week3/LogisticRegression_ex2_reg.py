from __future__ import division
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import axes3d, Axes3D 
#from matplotlib import cm
import numpy as np
import scipy.optimize as op
import pandas as pd
import PlotData, CostFunctionReg, MapFeature, GradientReg, CostReg, PlotDecisionBoundary, Sigmoid, Predict
# 
#data = pd.read_csv("ex2data2.txt", header = None)
#
#X = data[:,0:2]
#y = data[:,2]

data = pd.read_csv('ex2data2.txt', header = None)

y = np.array(data[len(data.columns)-1])
m = len(y)
X = np.array(np.array(data.values[:,:-1]))
PlotData.PlotData(X, y)
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')

X = MapFeature.MapFeature(X[:,0],X[:,1])

#initial fitting parameters
initial_theta = np.zeros((np.shape(X)[1],1))

# Set regularization parameter lambda to 1
Lambda = 1

cost, grad = CostFunctionReg.CostFunctionReg(initial_theta, X, y, Lambda)

print 'Cost at initial theta (zeros): ', cost

# Initialize fitting parameters
#initial_theta = np.zeros((np.shape(X)[1],1))
initial_theta = 1*np.random.randn(np.shape(X)[1])
Lambda = 1;

result = op.minimize(fun = CostReg.CostReg, x0 = initial_theta ,\
         args = (X, y, Lambda), method = 'COBYLA', jac =  \
         GradientReg.GradientReg, options = {'gtol':1e-6, 'maxiter':1000})   

theta = result.x

print 'Cost at theta found by fminunc: ', CostReg.CostReg(result.x, X , y, Lambda)
print 'Theta:'
print theta

#theta, j_min = op.fmin_bfgs(CostReg.CostReg, initial_theta ,\
#         args = (X, y, Lambda), fprime = GradientReg.GradientReg, \
#         full_output = True, disp = False, maxiter = 1000000)[0:2]        
#
#print 'Cost at theta found by fminunc: ', j_min
#print 'Theta:'
#print theta

PlotDecisionBoundary.PlotDecisionBoundary(theta, X, y)


p = Predict.Predict(theta, X)

print 'Train Accuracy:', sum(np.where(p==y, 1, 0))/100





















