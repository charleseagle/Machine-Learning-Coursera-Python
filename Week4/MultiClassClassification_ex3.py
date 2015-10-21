from __future__ import division
#import matplotlib.pyplot as plt
import scipy.io
#from matplotlib import cm
import numpy as np
import DisplayData, OneVsAll, PredictOneVsAll

input_layer_size = 400
num_labels = 10

print 'Loading and Visualizing Data'


data = scipy.io.loadmat('ex3data1.mat')
X = data['X']
y = data['y']

m = np.shape(X)[0]

# Randomly select 100 data points to display
rand_indices = np.random.permutation(range(m))
sel = X[rand_indices[0:100],:]

DisplayData.DisplayData(sel)

print 'Training One-vs-All Logistic Regression...'

Lambda = 0.1
all_theta = OneVsAll.OneVsAll(X, y, num_labels, Lambda)

pred = PredictOneVsAll.PredictOneVsAll(all_theta, X)

print 'Training Set Accuracy: ', sum(np.where(pred == y, 1, 0))/5000

















