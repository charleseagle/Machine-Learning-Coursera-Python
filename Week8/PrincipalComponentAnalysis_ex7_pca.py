from __future__ import division
import scipy.io
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy import misc
import matplotlib.pyplot as plt
import colorsys
import FeatureNormalize, Pca, DrawLine, ProjectData, RecoverData, \
        DisplayData, KMeansInitCentroids, RunkMeans, PlotDataPoints


print 'Visualizing example dataset for PCA.'

data = scipy.io.loadmat('E:\Machine learning\Week_8\Python\ex7data1.mat')
X = data['X']

plt.figure(1)
plt.plot(X[:,0], X[:,1], 'o')
plt.xlim(0.5,6.5)
plt.ylim(2,8)


X_norm, mu, sigma = FeatureNormalize.FeatureNormalize(X)

U, S = Pca.Pca(X_norm)

DrawLine.DrawLine(mu, mu+1.5*S[0]*U[:,0])
DrawLine.DrawLine(mu, mu+1.5*S[1]*U[:,1])
plt.show()

print 'Top eigenvector:'
print ' U(:,1) = %f %f', U[0,0], U[1,0]
print '(you should expect to see -0.707107 -0.707107)'

print 'Dimension reduction on example dataset.'
plt.figure(2)
plt.plot(X_norm[:,0], X_norm[:,1], 'o')
plt.xlim(-4,3)
plt.ylim(-4,3)
plt.show()

# Project the data onto K = 1 dimension
K =1
Z = ProjectData.ProjectData(X_norm, U, K)
print 'Projection of the first example: ', Z[0]
print '(this value should be about 1.481274)'

X_rec = RecoverData.RecoverData(Z, U, K)
print 'Approximation of the first example: ', X_rec[0,0], X_rec[0,1]
print '(this value should be about  -1.047419 -1.047419)'

# Draw lines connecting the projected points to the original points
plt.figure(3)
plt.plot(X_norm[:,0], X_norm[:,1], 'o')
plt.plot(X_rec[:,0], X_rec[:,1], 'o', color = 'red')
for i in range(np.shape(X_norm)[0]):
    DrawLine.DrawLine(X_norm[i,:], X_rec[i,:])
plt.show()

print 'Loading face dataset.'

data1 = scipy.io.loadmat('E:\Machine learning\Week_8\Python\ex7faces.mat')

X = data1['X']

DisplayData.DisplayData(X[0:100, :])

print 'Running PCA on face dataset. (this mght take a minute or two ...)'

X_norm, mu, sigma = FeatureNormalize.FeatureNormalize(X)
U, S = Pca.Pca(X_norm)

DisplayData.DisplayData(U[:,0:36].T)

print 'Dimension reduction for face dataset.'

K = 100
Z = ProjectData.ProjectData(X_norm, U, K)

print 'The projected data Z has a size of: ', np.shape(Z)

print 'Visualizing the projected (reduced dimension) faces.'

K = 100;
X_rec  = RecoverData.RecoverData(Z, U, K)

plt.figure(5)
plt.suptitle('Original faces')
DisplayData.DisplayData(X_norm[0:100,:])
plt.suptitle('Recovered faces')
DisplayData.DisplayData(X_rec[0:100,:])
plt.show()

A = misc.imread('E:/Machine learning/Week_8/Python/bird_small.png')
A = A/255
img_size = np.shape(A)

X = A.reshape(img_size[0]*img_size[1],3)

K = 16
max_iters = 10

initial_centroids = KMeansInitCentroids.KMeansInitCentroids(X, K)
centroids, idx = RunkMeans.RunkMeans(X, initial_centroids, max_iters)

sel = np.array(np.floor(np.random.rand(1000,1)*np.shape(X)[0]), dtype = 'i')
def to_rgb(x):
    return colorsys.hsv_to_rgb(*x)
h = np.arange(K+1).reshape(K+1,1)/(K+1)
h = np.append(h, np.ones((K+1, 2)), 1)
palette = np.apply_along_axis(to_rgb, axis = 1, arr = h)
colors = palette[idx[sel].flatten()-1,:]

# Visualize the data and centroid memberships in 3D
fig = plt.figure(5)
ax = Axes3D(fig)
#ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[sel, 0], X[sel,1], X[sel,2], c = colors)
plt.show()

#plt.figure(5)
#plt.scatter(X[sel, 0], X[sel,1], X[sel,2], c = colors)
#plt.show()


X_norm, mu, sigma = FeatureNormalize.FeatureNormalize(X)
U, S = Pca.Pca(X_norm)
Z = ProjectData.ProjectData(X_norm, U, 2);

plt.figure(6)
PlotDataPoints.PlotDataPoints(Z[sel,:].reshape(np.shape(sel)[0],2), idx[sel], K)
plt.suptitle('Pixel dataset plotted in 2D, using PCA for dimensionality reduction')
plt.show()
























