from __future__ import division
#import matplotlib.pyplot as plt
import scipy.io
import scipy.optimize as opt
import numpy as np
import DisplayData, NnCostFunction, SigmoidGradient, RandInitializeWeights, \
        CheckNNGradients, NnCost, NnGradient, Predict

input_layer_size  = 400;  # 20x20 Input Images of Digits
hidden_layer_size = 25;   # 25 hidden units
num_labels = 10;          # 10 labels, from 1 to 10 

print 'Loading and Visualizing Data ...'

data = scipy.io.loadmat('E:\Machine learning\Week_5\Python\ex4data1.mat')
X = data['X']
y = data['y']
m = np.shape(X)[0]
rand_indices = np.random.permutation(range(m))
sel = X[rand_indices[0:100],:]

DisplayData.DisplayData(sel)

print 'Loading Saved Neural Network Parameters ...'
data1 = scipy.io.loadmat('E:\Machine learning\Week_5\Python\ex4weights.mat')
theta1 = data1['Theta1']
theta2 = data1['Theta2']

# Unroll parameters 
nn_params= np.append(theta1.flatten(), theta2.flatten(), 1)

print 'Feedforward Using Neural Network ...'

# Weight regularization parameter (we set this to 0 here).
Lambda = 0
J , grad = NnCostFunction.NnCostFunction(nn_params, input_layer_size, \
                    hidden_layer_size, num_labels, X, y, Lambda)
print 'Cost at parameters (loaded from ex4weights): (about 0.287629)', J

# Weight regularization parameter (we set this to 1 here).
Lambda = 1
J , grad = NnCostFunction.NnCostFunction(nn_params, input_layer_size, \
                    hidden_layer_size, num_labels, X, y, Lambda)
print 'Cost at parameters (loaded from ex4weights): (about 0.383770)', J

print 'Evaluating sigmoid gradient...'
g = SigmoidGradient.SigmoidGradient(np.array([1, -0.5, 0, 0.5, 1]))
print 'Sigmoid gradient evaluated at [1, -0.5, 0, 0.5, 1]:', g

print 'Initializing Neural Network Parameters ...'
initial_Theta1 = RandInitializeWeights.RandInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = RandInitializeWeights.RandInitializeWeights(hidden_layer_size, num_labels)

initial_nn_params = np.append(initial_Theta1.flatten(), initial_Theta2.flatten())

print 'Checking Backpropagation...'

CheckNNGradients.CheckNNGradients()

print 'Checking Backpropagation (w/ Regularization) ...'
#Check gradients by running checkNNGradients
Lambda = 3;
CheckNNGradients.CheckNNGradients(Lambda);

debug_J, grad_J  = NnCostFunction.NnCostFunction(nn_params, input_layer_size, \
                          hidden_layer_size, num_labels, X, y, Lambda)
    
print 'Cost at (fixed) debugging parameters (w/ lambda = 10) '
print '(this value should be about 0.576051): ', debug_J

print 'Training Neural Network...'
Lambda = 1

result = opt.minimize(fun = NnCost.NnCost, x0 = initial_nn_params ,\
         args = (input_layer_size, hidden_layer_size, num_labels, X, y, \
         Lambda), method = 'TNC', jac =  \
         NnGradient.NnGradient, options = {'maxiter':100})   

nn_params = result.x
theta1 = nn_params[0:hidden_layer_size * (input_layer_size + 1)].reshape(hidden_layer_size, (input_layer_size + 1))

theta2 = nn_params[(hidden_layer_size * (input_layer_size + 1)):].reshape\
            (num_labels, (hidden_layer_size + 1))
            
            
print 'Visualizing Neural Network...'

DisplayData.DisplayData(theta1[:,1:])

pred = Predict.Predict(theta1, theta2, X)
print 'Training Set Accuracy: ', np.mean(np.where(pred == y, 1, 0))
          
        
















