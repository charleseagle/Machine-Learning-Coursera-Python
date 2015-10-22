from __future__ import division
import scipy.io
import matplotlib.pyplot as plt
#import scipy.optimize as opt
import numpy as np
import LinearRegCostFunction, TrainLinearReg, LearningCurve, \
    PolyFeatures, FeatureNormalize, PlotFit, ValidationCurve

print 'Loading and Visualizing Data ...'

data = scipy.io.loadmat("E:\Machine learning\Week_6\Python\ex5data1.mat")

X = data['X']
y = data['y']

m = np.shape(X)[0]

plt.figure(1)
plt.plot(X, y, 'x', color = 'red')
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.show()

theta = np.ones((2,1))
X1 = np.append(np.ones((m,1)), X, 1)
J, grad = LinearRegCostFunction.LinearRegCostFunction(X1, y, theta, 1)
print 'Cost at theta = [1 ; 1]: (this value should be about 303.993192)', J

print 'Gradient at theta = [1 ; 1]: (this value should \
        be about [-15.303016; 598.250744])', grad[0], grad[1]
        
       

Lambda = 0
theta = TrainLinearReg.TrainLinearReg(X1, y, Lambda)
    
# Plot fit over the data
    
plt.figure(2)
plt.plot(X, y, 'x', linewidth = 1.5, color = 'red')
plt.plot (X, X1.dot(theta), '--', linewidth = 2, color = 'blue')
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.show()

Lambda = 0
Xval = data['Xval']
yval = data['yval']
n = np.shape(Xval)[0]
Xval1 = np.append(np.ones((n,1)), Xval, 1)

error_train, error_val = LearningCurve.LearningCurve(X1, y, Xval1, yval, Lambda)

plt.figure(3)
plt.suptitle('Learning curve for linear regression', fontsize=15)
line1 = plt.plot(np.arange(m)+1, error_train, '-', color ='red', label = 'Train')
line2 = plt.plot(np.arange(m)+1, error_val, '-', color ='blue', label = 'Cross Validation')
plt.legend(loc = 1)
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.xlim(0, 13)
plt.ylim(0,150)
plt.show()

print 'Training Examples\tTrain Error\tCross Validation Error'
for i in range(m):
    print i+1, error_train[i], error_val[i]
    
p = 8
# Map X onto Polynomial Features and Normalize

X_poly = PolyFeatures.PolyFeatures(X, p)

X_poly, mu, sigma = FeatureNormalize.FeatureNormalize(X_poly)
X_poly = np.append(np.ones((m,1)), X_poly, 1)

# Map X_poly_test and normalize (using mu and sigma)
Xtest = data['Xtest']
X_poly_test = PolyFeatures.PolyFeatures(Xtest, p)
X_poly_test = X_poly_test - mu
X_poly_test = X_poly_test/sigma
X_poly_test = np.append(np.ones((np.shape(X_poly_test)[0],1)), X_poly_test, 1)

# Map X_poly_val and normalize (using mu and sigma)
X_poly_val = PolyFeatures.PolyFeatures(Xval, p);
X_poly_val = X_poly_val - mu;
X_poly_val = X_poly_val/sigma;
X_poly_val = np.append(np.ones((np.shape(X_poly_val)[0],1)), X_poly_val, 1); 

print 'Normalized Training Example 1: ', X_poly[0, :]

Lambda = 1
theta = TrainLinearReg.TrainLinearReg(X_poly, y, Lambda)

plt.figure(4)
plt.plot(X, y, 'x', linewidth = 1.5, color = 'red')
PlotFit.PlotFit(np.min(X), np.max(X), mu, sigma, theta, p)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.suptitle('Polynomial Regression Fit Lambda')
plt.show()

error_train, error_val = LearningCurve.LearningCurve(X_poly, \
                    y, X_poly_val, yval, Lambda)
plt.figure(5)
plt.plot(np.arange(m)+1, error_train, '-', color = 'red', label ='Train')
plt.plot(np.arange(m)+1, error_val, '-', color = 'blue', label ='Cross Validation')
plt.legend(loc = 1)
plt.suptitle('Polynomial Regression Learning Curve')
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.show()

print 'Polynomial Regression lambda = ', Lambda
print '# Training Examples\tTrain Error\tCross Validation Error'
for i in range(m):
    print i, error_train[i], error_val[i]

lambda_vec, error_train, error_val = ValidationCurve.ValidationCurve(\
            X_poly, y, X_poly_val, yval)

plt.figure(6)
plt.plot(lambda_vec, error_train, '-', color = 'red', label ='Train')
plt.plot(lambda_vec, error_val, '-', color = 'red', label ='Cross Validation')
plt.xlabel('lambda')
plt.ylabel('Error')
plt.show()

print 'lambda\tTrain Error\tValidation Error'
for i in range(len(lambda_vec)):
    print lambda_vec[i], error_train[i], error_val[i]


















