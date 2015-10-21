from __future__ import division
#import matplotlib.pyplot as plt
import scipy.io
#from matplotlib import cm
import numpy as np
import DisplayData, OneVsAll, Predict

input_layer_size  = 400;  # 20x20 Input Images of Digits
hidden_layer_size = 25;   # 25 hidden units
num_labels = 10;          # 10 labels, from 1 to 10 

print 'Loading and Visualizing Data ...'

data = scipy.io.loadmat('ex3data1.mat')
X = data['X']
y = data['y']
m = np.shape(X)[0]

#Randomly select 100 data points to display
rand_indices = np.random.permutation(range(m))
sel = X[rand_indices[0:100],:]

DisplayData.DisplayData(sel)

data1 = scipy.io.loadmat('ex3weights.mat')
theta1 = data1['Theta1']
theta2 = data1['Theta2']

pred = Predict.Predict(theta1, theta2, X)
print 'Training Set Accuracy: ', np.mean(np.where(pred == y, 1, 0))

rp = np.random.permutation(range(m))
for i in range(m):
    print 'Displaying Example Image'
    DisplayData.DisplayData(X[rp[i],:].reshape(1,400))
    pred = Predict.Predict(theta1, theta2, X[rp[i],:].reshape(1,400))
    
    print 'Neural Network Prediction: ', pred, np.mod(pred,10)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

























