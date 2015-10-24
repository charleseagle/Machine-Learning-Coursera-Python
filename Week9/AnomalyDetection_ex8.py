from __future__ import division
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import EstimateGaussian, MultivariateGaussian, VisualizeFit, SelectThreshold

print 'Visualizing example dataset for outlier detection.'
data = scipy.io.loadmat('E:\Machine learning\Week_9\Python\ex8data1.mat')

X = data['X']
Xval = data['Xval']
yval= data['yval']


plt.figure(1)
plt.plot(X[:,0], X[:,1], 'x', color = 'r')
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')
plt.xlim(0,30)
plt.ylim(0,30)
plt.show()

print 'Visualizing Gaussian fit.'
mu, sigma2 = EstimateGaussian.EstimateGaussian(X)

p = MultivariateGaussian.MultivariateGaussian(X, mu, sigma2)

plt.figure(2)
VisualizeFit.VisualizeFit(X, mu, sigma2)
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')


pval = MultivariateGaussian.MultivariateGaussian(Xval, mu, sigma2)
epsilon, F1 = SelectThreshold.SelectThreshold(yval, pval)

print 'Best epsilon found using cross-validation: ', epsilon
print 'Best F1 on Cross Validation Set: ', F1
print '(you should see a value epsilon of about 8.99e-05)'

#Find the outliers in the training set and plot the
outliers = np.where(p < epsilon)

plt.plot(X[outliers, 0], X[outliers, 1], 'o', color = 'b')
plt.show()

data1 = scipy.io.loadmat('E:\Machine learning\Week_9\Python\ex8data2.mat')
X = data1['X']
Xval = data1['Xval']
yval= data1['yval']

mu, sigma2 = EstimateGaussian.EstimateGaussian(X)

p = MultivariateGaussian.MultivariateGaussian(X, mu, sigma2)
pval = MultivariateGaussian.MultivariateGaussian(Xval, mu, sigma2)
epsilon, F1 = SelectThreshold.SelectThreshold(yval, pval)

print 'Best epsilon found using cross-validation: ', epsilon
print 'Best F1 on Cross Validation Set: ', F1
print '# Outliers found: ', np.sum(sum(p < epsilon)+0)
print '(you should see a value epsilon of about 1.38e-18)'



