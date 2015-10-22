from __future__ import division
import numpy as np
import DebugInitializeWeights, NnCostFunction, ComputeNumericalGradient


def CheckNNGradients(Lambda = None):
    if Lambda == None:
        Lambda = 0
    
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5
    
    theta1 = DebugInitializeWeights.DebugInitializeWeights(hidden_layer_size, input_layer_size)
    theta2 = DebugInitializeWeights.DebugInitializeWeights(num_labels, hidden_layer_size)
    
    X  = DebugInitializeWeights.DebugInitializeWeights(m, input_layer_size - 1)
    y  = 1 + np.mod(np.arange(m)+1, num_labels)
    nn_params = np.append(theta1.flatten(), theta2.flatten())
    
    def CostFunc(p):
        cost, grad = NnCostFunction.NnCostFunction(p,input_layer_size, \
                hidden_layer_size, num_labels, X, y, Lambda)
        return [cost, grad]
    
    cost, grad = CostFunc(nn_params)

    numgrad = ComputeNumericalGradient.ComputeNumericalGradient(CostFunc,nn_params)
    
    #print numgrad, grad
    print 'The above two columns you get should be very similar.'
    print '(Left-Your Numerical Gradient, Right-Analytical Gradient)'
    
    diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)
    print 'If your backpropagation implementation is correct,'
    print 'then the relative difference will be small (less than 1e-9).'
    print 'Relative Difference: ', diff
    