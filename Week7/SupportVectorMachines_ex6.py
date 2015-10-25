from __future__ import division
import scipy.io 
import numpy as np
import matplotlib.pyplot as plt
import PlotData, SvmTrain, Functions, VisualizeBoundaryLinear, \
        VisualizeBoundary, Dataset3Params


print 'Loading and Visualizing Data ...'

data1 = scipy.io.loadmat('E:\Machine learning\Week_7\Python\ex6data1.mat')
X = data1['X']
y = np.array(data1['y'], dtype = 'i')
plt.figure(1)
PlotData.PlotData(X,y)

print 'Training Linear SVM ...'
funcs = Functions.Functions()
C = 1
model = SvmTrain.SvmTrain(X, y, C, funcs.LinearKernel, 1e-3, 20)
VisualizeBoundaryLinear.VisualizeBoundaryLinear(X, y, model)
plt.show()


print 'Evaluating the Gaussian Kernel ...'
x1 = np.array([1, 2, 1])
x2 = np.array([0, 4, -1])
sigma = 2
sim = funcs.GaussianKernel(x1, x2, sigma)
print 'Gaussian Kernel between x1 = [1, 2, 1], x2 = [0, 4, -1],\
         sigma = 0.5 : %8.10f'% sim
print '(this value should be about 0.324652)'

print'Loading and Visualizing Data ...'
data2 = scipy.io.loadmat('E:\Machine learning\Week_7\Python\ex6data2.mat')
X = data2['X']
y = np.array(data2['y'], dtype = 'i')

plt.figure(2)
PlotData.PlotData(X, y)
plt.xlim(0,1.03)
plt.ylim(0.38,1.02)

print 'Training SVM with RBF Kernel (this may take 1 to 2 minutes) ...'

C = 1
sigma = 0.1

GaussianKernel = lambda x1, x2: funcs.GaussianKernel(x1,x2,sigma)
GaussianKernel.__name__ = 'GaussianKernel'

model= SvmTrain.SvmTrain(X, y, C, GaussianKernel)
VisualizeBoundary.VisualizeBoundary(X, y, model)
plt.show()

print 'Loading and Visualizing Data ...'

data3 = scipy.io.loadmat('E:\Machine learning\Week_7\Python\ex6data3.mat')
X = data3['X']
y = np.array(data3['y'], dtype = 'i')
Xval = data3['Xval']
yval = data3['yval']

plt.figure(3)
PlotData.PlotData(X, y);

C, sigma = Dataset3Params.Dataset3Params(X, y, Xval, yval)

model= SvmTrain.SvmTrain(X, y, C, GaussianKernel)
VisualizeBoundary.VisualizeBoundary(X, y, model)
plt.show()









