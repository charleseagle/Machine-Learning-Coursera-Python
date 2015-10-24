# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 19:10:58 2015

@author: Charleseagle
"""

from __future__ import division
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import EstimateGaussian, MultivariateGaussian, VisualizeFit, \
        SelectThreshold, CofiCostFunc, CheckCostFunction, \
        LoadMovieList, NormalizeRatings, CofiGradient, CofiCost

print 'Loading movie ratings dataset.'
data = scipy.io.loadmat('E:\Machine learning\Week_9\Python\ex8_movies.mat')
R = data['R']
Y = data['Y']
print 'Average rating for movie 1 (Toy Story): %8.8f/5 ' \
        %np.mean(Y[0,np.where(R[0,:] -1 == 0)])

plt.figure(figsize=(5, 5))
plt.imshow(Y)
plt.show()

data1 = scipy.io.loadmat('E:\Machine learning\Week_9\Python\ex8_movieParams.mat')
X = data1['X']
theta = data1['Theta']
# Reduce the data set size so that this runs faster
num_users = 4
num_movies = 5
num_features = 3

X = X[0:num_movies, 0:num_features]
theta = theta[0:num_users, 0:num_features]
Y = Y[0:num_movies, 0:num_users]
R = R[0:num_movies, 0:num_users]

J, grad = CofiCostFunc.CofiCostFunc(np.append(X.flatten(), theta.flatten(), 1), \
        Y, R, num_users, num_movies, num_features, 0)
print 'Cost at loaded parameters: %2.2f (this value should be about 22.22)' %J

print 'Checking Gradients (without regularization) ...'
CheckCostFunction.CheckCostFunction()

J, grad = CofiCostFunc.CofiCostFunc(np.append(X.flatten(), theta.flatten(), 1), \
        Y, R, num_users, num_movies, num_features, 1.5)
print 'Cost at loaded parameters (lambda = 1.5): %2.2f \
        (this value should be about 31.34)' %J
        
print 'Checking Gradients (with regularization) ...'
CheckCostFunction.CheckCostFunction(1.5)

movielist = LoadMovieList.LoadMovieList()
my_ratings = np.zeros((1682, 1))

#Check the file movie_idx.txt for id of each movie in our dataset
#For example, Toy Story (1995) has ID 1, so to rate it "4", you can set
my_ratings[0] = 4

#Or suppose did not enjoy Silence of the Lambs (1991), you can set
my_ratings[97] = 2
# rate other movies
my_ratings[6] = 3
my_ratings[11] = 5
my_ratings[53] = 4
my_ratings[63] = 5
my_ratings[65] = 3
my_ratings[68] = 5
my_ratings[182] = 4
my_ratings[225] = 5
my_ratings[354] = 5

print 'New user ratings:'
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print 'Rated {} for {}'.format(my_ratings[i], movielist[i])

print 'Training collaborative filtering...'
data = scipy.io.loadmat('E:\Machine learning\Week_9\Python\ex8_movies.mat')
R = data['R']
Y = data['Y']
Y = np.append(my_ratings, Y, 1)
R = np.append((my_ratings!=0)+0, R, 1)

Ynorm, Ymean =NormalizeRatings.NormalizeRatings(Y,R)
num_users = np.shape(Y)[1]
num_movies = np.shape(Y)[0]
num_features = 10

X = np.random.randn(num_movies, num_features);
theta = np.random.randn(num_users, num_features);

initial_parameters = np.append(X.flatten(), theta.flatten(), 1)
Lambda = 10

result = opt.minimize(fun = CofiCost.CofiCost, x0 = initial_parameters,\
         args = (Y, R, num_users, num_movies, num_features, Lambda), \
         method = 'CG', jac = CofiGradient.CofiGradient, \
         options = {'maxiter':100})

theta = result.x
# Unfold the returned theta back into U and W
X = theta[0:num_movies*num_features].reshape(num_movies, num_features)
theta = theta[num_movies*num_features:].reshape(num_users, num_features)

print 'Recommender system learning completed.'


p = X.dot(theta.T)
my_predictions = p[:,0].reshape(np.shape(p)[0],1) + Ymean

movielist = LoadMovieList.LoadMovieList()
ix = np.argsort(my_predictions, axis=0, kind ='mergesort')[::-1]
my_predictions = my_predictions[ix]
my_predictions = my_predictions.flatten()

print 'Top recommendations for you:'
for i in range(10):
    j = ix[i,0] 
    print 'Predicting rating %1.1f for movie %s' %(my_predictions[i],\
                        movielist[j])
        
print 'Original ratings provided:'
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print 'Rated {} for {}'.format(my_ratings[i], movielist[i])
    
    
    
    
    
    
    
    
    




