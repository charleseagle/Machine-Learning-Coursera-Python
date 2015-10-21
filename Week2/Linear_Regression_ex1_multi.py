from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import featureNormalize, warmUpExercise, normalEqn, gradientDescentMulti

print "Loading data"
data = np.loadtxt('ex1data2.txt', delimiter = ',')
X = np.matrix(data[:,0:2])
y = np.matrix(data[:,2]).T
m = np.shape(y)[0]

print 'First 10 examples from the dataset:'
print X[0:9,:], y[0:9,:]

print "'Program paused. Press enter to continue."
raw_input("Press ENTER to continue")

# Scale freatures and set them to zero mean
print "Normalizing Features"
X, mu, sigma = featureNormalize.featureNormalize(X)

# add intercept term to X
X = np.matrix(np.append(np.ones((m,1)), X, 1)) #add a column of ones to X

print "Running gradient descent"

#Choose some alpha value
alpha = 0.1
num_iters = 50

#initial theta and run gredient descent
theta = np.matrix(np.zeros((3,1)))
theta, J_history = gradientDescentMulti.gradientDescentMulti(X, y, theta, alpha, num_iters)

plt.figure()
plt.plot(np.linspace(1,np.shape(J_history)[0]), J_history, 'o', label = 'Iteration', color = 'red')
plt.xlabel('Number of iterations')
plt.ylabel('Cost fuction J')
plt.legend(loc = 1)
plt.show()

print 'Theta computed from gradient descent'
print theta

price = theta[0,0] + theta[1,0]*(1650-mu[0,0])/sigma[0,0] + theta[2,0]*(3-mu[0,1])/sigma[0,1]

print 'Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):'
print price
print 'Program paused. Press enter to continue.'

print "'Program paused. Press enter to continue."
raw_input("Press ENTER to continue")

print 'Solving with normal equations'

data = np.loadtxt('ex1data2.txt', delimiter = ',')
X = np.matrix(data[:,0:2])
y = np.matrix(data[:,2]).T
m = np.shape(y)[0]

X = np.matrix(np.append(np.ones((m,1)), X, 1))

theta = normalEqn.normalEqn(X, y)

print 'Theta computed from the normal equations:'
print theta

price = theta[0,0] + theta[1,0]*1650 + theta[2,0]*3

print 'Predicted price of a 1650 sq-ft, 3 br house (using normal equations):' 
print price























