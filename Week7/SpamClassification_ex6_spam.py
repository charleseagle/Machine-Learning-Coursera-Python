# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 09:05:26 2015

@author: Charleseagle
"""
from __future__ import division
import scipy.io 
import numpy as np
import matplotlib.pyplot as plt
import ProcessEmail, EmailFeatures, SvmTrain, Functions, SvmPredict, \
        GetVocabList


print 'Preprocessing sample email (emailSample1.txt)'

file_contents = '' 
with open('E:/Machine learning/Week_7/Python/emailSample1.txt', 'r') as fid:
        file_contents = fid.read()

word_indices = ProcessEmail.ProcessEmail(file_contents)

print 'Word Indices: {}'.format(word_indices)

print 'Extracting features from sample email (emailSample1.txt)'

file_contents = '' 
with open('E:/Machine learning/Week_7/Python/emailSample1.txt', 'r') as fid:
        file_contents = fid.read()
word_indices = ProcessEmail.ProcessEmail(file_contents)
features = EmailFeatures.EmailFeatures(word_indices)

print 'Length of feature vector: {}'.format(len(features))
print 'Number of non-zero entries: ', np.sum(features[np.where(features > 0)[0]])


data = scipy.io.loadmat('E:\Machine learning\Week_7\Python\spamTrain.mat')
X = np.array(data['X'], dtype = 'f')
y = np.array(data['y'], dtype = 'i')
print 'Training Linear SVM (Spam Classification)'
print '(this may take 1 to 2 minutes) ...'

C = 0.1
funcs = Functions.Functions()
model = SvmTrain.SvmTrain(X, y, C, funcs.LinearKernel)
p = SvmPredict.SvmPredict(model, X)
p[p == 0] = -1
print 'Training Accuracy: %2.10f' % np.mean(np.where(p == y, 1, 0))

data1 = scipy.io.loadmat('E:\Machine learning\Week_7\Python\spamTest.mat')
print 'Evaluating the trained Linear SVM on a test set ...'
Xtest = np.array(data1['Xtest'], dtype = 'f')
ytest = np.array(data1['ytest'], dtype = 'i')
p = SvmPredict.SvmPredict(model, Xtest)
print 'Test Accuracy: %2.10f' % np.mean(np.where(p == ytest, 1, 0))

idx = np.argsort(model['w'], axis = None)[::-1]
vocablist = GetVocabList.GetVocabList()
vocablist = dict( (v, k) for (k, v) in vocablist.iteritems() )
print 'Top predictors of spam: '
for i in range(1, 16):
    print '{}  {}'.format(vocablist[idx[i-1]+1], model['w'][idx[i-1]])


file_contents = '' 
with open('E:/Machine learning/Week_7/Python/spamSample1.txt', 'r') as fid:
        file_contents = fid.read()
word_indices = ProcessEmail.ProcessEmail(file_contents)
x = EmailFeatures.EmailFeatures(word_indices).T
p = SvmPredict.SvmPredict(model, x)
print 'Processed spamSample1.txt Spam Classification: ', int(p)
print '(1 indicates spam, 0 indicates not spam)'



