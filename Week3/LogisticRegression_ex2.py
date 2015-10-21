from __future__ import division
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import axes3d, Axes3D 
#from matplotlib import cm
import numpy as np
import scipy.optimize as op
import PlotData, CostFunction, Gradient, Cost, PlotDecisionBoundary, Sigmoid, Predict
 


data = np.loadtxt('ex2data1.txt', delimiter = ',')
X = data[:,0:2]
y = data[:,2]

print 'Plotting data with + indicating (y = 1) examples and o \
       indicating (y = 0) examples.'

PlotData.PlotData(X, y)


m, n = np.shape(X)

X = np.append(np.ones((m,1)), X, 1)

initial_theta = np.zeros((n+1,1))

cost, grad = CostFunction.CostFunction(initial_theta, X, y)
print 'Cost at initial theta (zeros):', cost
print 'Gradient at initial theta (zeros):'
print grad



#res = op.minimize(fun, \
#    x0, args = (X, y), method = 'TNC', jac = None)
        
result = op.minimize(fun = Cost.Cost, x0 = initial_theta ,\
         args = (X, y), method = 'TNC', jac =  Gradient.Gradient)        
theta = result.x

print 'Cost at theta found by fminunc: ', Cost.Cost(result.x, X , y)
print 'Theta:'
print theta


PlotDecisionBoundary.PlotDecisionBoundary(theta, X, y)

a = np.array([1, 45, 85])
prob = Sigmoid.Sigmoid(a.dot(theta))
print 'For a student with scores 45 and 85, \
    we predict an admission probability of', prob

p = Predict.Predict(theta, X)

print 'Train Accuracy:', sum(np.where(p==y, 1, 0))












