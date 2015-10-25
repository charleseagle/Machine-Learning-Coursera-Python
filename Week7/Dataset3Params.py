# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 00:44:58 2015

@author: Charleseagle
"""
from __future__ import division
import numpy as np
import SvmTrain, SvmPredict, Functions

def Dataset3Params(X, y, Xval, yval):
    C = 1
    sigma = 0.3
    
    A = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30])
    prediction_error = np.zeros(((len(A),len(A))))
    
    for i in range(len(A)):
        for j in range(len(A)):
            funcs = Functions.Functions()
            GaussianKernel = lambda x1, x2: funcs.GaussianKernel(x1,x2,A[j])
            GaussianKernel.__name__ = 'GaussianKernel'
            model = SvmTrain.SvmTrain(X, y, A[i], GaussianKernel)
            predictions = SvmPredict.SvmPredict(model, Xval)
            prediction_error[i,j] = np.mean(np.where(predictions != yval, 1, 0))
    
    colmin = prediction_error.min(axis=0)
    coli = np.where(prediction_error == prediction_error.min(axis=0))[0]
    rowmin = min(colmin)
    rowj = np.where(prediction_error == rowmin)[1]
    C = A[coli[rowj]]
    sigma = A[rowj]
#    coli = np.where(prediction_error == prediction_error.flatten().min())[0]
#    rowj = np.where(prediction_error == prediction_error.flatten().min())[1]
    return [C, sigma]