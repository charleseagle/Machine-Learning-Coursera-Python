from __future__ import division
import numpy as np
import Sigmoid

def NnCost(nn_params, input_layer_size, hidden_layer_size,\
                     num_labels, X, y, Lambda):
    
    theta1 = nn_params[0:hidden_layer_size * (input_layer_size + 1)].\
                reshape(hidden_layer_size, (input_layer_size + 1))
    theta2 = nn_params[(hidden_layer_size * (input_layer_size + 1)):].\
                reshape(num_labels, (hidden_layer_size + 1))
                
    m = np.shape(X)[0]
    J = 0
#    theta1_grad = np.zeros((np.shape(theta1)))
#    theta2_grad = np.zeros((np.shape(theta2)))
    a1 = np.append(np.ones((m,1)), X, 1)
    z2 = a1.dot(theta1.T)
    a2 = Sigmoid.Sigmoid(z2)
    a2 = np.append(np.ones((m,1)), a2, 1)
    z3 = a2.dot(theta2.T)
    a3 = Sigmoid.Sigmoid(z3)
    
    y_matrix = np.zeros((m,num_labels))
    for i in range(m):
        y_matrix[i,y[i]-1] = 1
        
    J = -(1/m)*(np.sum(np.sum(y_matrix*np.log(a3))) + \
        np.sum(np.sum((1-y_matrix)*np.log(1-a3)))) + \
        Lambda/(2*m)*np.sum(np.sum(theta1[:,1:]**2))+ \
        Lambda/(2*m)*np.sum(np.sum(theta2[:,1:]**2));
#    d3 = a3 - y_matrix
#    d2 = d3.dot(theta2)*a2*(1-a2)
#    d2 = d2[:,1:]
#    theta1_grad = 1/m*(d2.T.dot(a1) + Lambda*np.append(\
#        np.zeros((hidden_layer_size,1)), theta1[:,1:], 1))
#    theta2_grad = 1/m*(d3.T.dot(a2)+ Lambda*np.append(\
#        np.zeros((num_labels,1)), theta2[:,1:], 1))
#    grad = np.append(theta1_grad.flatten(), theta2_grad.flatten(), 1)
#    
    return J
    






















