from __future__ import division
import scipy.io
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import FindClosestCentroids, ComputeCentroids, RunkMeans, KMeansInitCentroids



print 'Finding closest centroids.'

data = scipy.io.loadmat('E:\Machine learning\Week_8\Python\ex7data2.mat')
X = data['X']

# Select an initial set of centroids
K = 3
initial_centroids = np.array([[3,3],[ 6,2],[8,5]])
# Find the closest centroids for the examples using the initial centroids
idx = FindClosestCentroids.FindClosestCentroids(X, initial_centroids)
print '(the closest centroids should be 1, 3, 2 respectively)'
print 'Closest centroids for the first 3 examples: ', idx[0:3,0]

print 'Computing centroids means.'
centroids = ComputeCentroids.ComputeCentroids(X, idx, K)
print '(the centroids should be)'
print '[ 2.428301 3.157924 ]'
print '[ 5.813503 2.633656 ]'
print '[ 7.119387 3.616684 ]'
print 'Centroids computed after initial finding of closest centroids: ', centroids

print 'Running K-Means clustering on example dataset.'
max_iters = 10
centroids, idx = RunkMeans.RunkMeans(X, initial_centroids, max_iters, True)

print 'K-Means Done.'


print 'Running K-Means clustering on pixels from an image.'

A = misc.imread('E:/Machine learning/Week_8/Python/bird_small.png')
A = A/255
img_size = np.shape(A)

X = A.reshape(img_size[0]*img_size[1],3)

K = 16
max_iters = 10

initial_centroids = KMeansInitCentroids.KMeansInitCentroids(X, K)
centroids, idx = RunkMeans.RunkMeans(X, initial_centroids, max_iters)

print 'Applying K-Means to compress an image.'

idx = FindClosestCentroids.FindClosestCentroids(X, centroids)
X_recovered = centroids[idx-1,:]
X_recovered = X_recovered. reshape(img_size[0], img_size[1], 3)

fig = plt.figure(2)
ax = fig.add_subplot(1, 2, 1)
ax.imshow(A)
ax = fig.add_subplot(1,2,2)
ax.imshow(X_recovered)
fig.show()





















