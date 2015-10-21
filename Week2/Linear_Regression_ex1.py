from __future__ import division
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
#from matplotlib import cm
import numpy as np
import warmUpExercise, computeCost, gradientDescent

print "Running warmUpExercise"
print "5x5 Indentity Matrix:"

print warmUpExercise.warmUpExercise()

print "Program paused. Press enter to continue."
raw_input("Press ENTER to continue")

print ("Plotting data")

#infile = open('ex1data1.txt', 'r')
#for line in infile:
#    data.append(line)
#infile.close()

data = np.loadtxt('ex1data1.txt', delimiter = ',')
X = data[:,0]
y = np.matrix(data[:,1]).T
m = len(y)

plt.figure()
plt.plot(X, y, 'o')
#plt.savefig('temp.png')
plt.show()

print "'Program paused. Press enter to continue."
raw_input("Press ENTER to continue")

print "Running gradient descent."

X = np.matrix([np.ones(m), data[:,0]]).T #add a column of ones to X
theta = np.matrix(np.zeros(2)).T #initialize fitting parameters

# gradient descent setting
iterations = 1500
alpha = 0.01

# compute and display initial cost
computeCost.computeCost(X, y, theta)

# Run gradietn descent
theta, J_hisotry = gradientDescent.gradientDescent(X, y, theta, alpha, iterations)

# Print theta to screen
print "Theta found by gradient descent: ", theta[0], theta[1]

plt.figure()
plt.plot(X[:,1], y, 'o', label = 'Training data', color = 'blue')
plt.plot(X[:,1], X*theta, '-', label = 'Linear regression', color = 'red')
plt.legend(loc = 4)
plt.show()

#Predict values for population sizes of 35,000 and 70,000
predict1 = [1, 3.5]*theta
print 'For population = 35,000, we predict a profit of ', predict1*1000
predict2 = [1, 7]*theta
print "For population = 70,000, we predict a profit of", predict2*10000

print "'Program paused. Press enter to continue."
raw_input("Press ENTER to continue")

# Visualizing J(theta_0, theta_1)
print "Visualizing J(theta_0, theta_1)"

theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = np.matrix([theta0_vals[i], theta1_vals[j]]).T
        J_vals[i,j] = computeCost.computeCost(X, y, t)
# transpose J_vals
J_vals = J_vals.T

#surface plot
fig = plt.figure()
ax = Axes3D(fig)
#ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(theta0_vals, theta1_vals, J_vals)
ax.set_xlabel('theta_0')
ax.set_ylabel('theta_1')
ax.set_zlabel('J Values')
plt.show()


plt.figure(1)
plt.contour(theta0_vals, theta1_vals, J_vals, np.logspace(-2, 3, 20))
plt.xlabel('theta_0')
plt.ylabel('theta_1')
plt.show()




