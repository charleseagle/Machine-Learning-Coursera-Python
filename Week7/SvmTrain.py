from __future__ import division
import numpy as np
import sys



def SvmTrain(X, Y, C, KernelFunction, tol = None, max_passes = None):
    if tol == None:
        tol = 1e-3
    if max_passes == None:
        max_passes = 5
    m, n = np.shape(X)
    # Map 0 to -1
    Y[Y==0] = -1
    
    alphas = np.zeros((m,1))
    b = 0
    E = np.zeros((m, 1))
    passes = 0
    eta = 0
    L = 0
    H = 0
        
    if KernelFunction.__name__ == "LinearKernel":
        K = X.dot(X.T)
        
    elif KernelFunction.__name__ == 'GaussianKernel':
        X2 = (X**2).sum(axis = 1)
        K = X2.reshape(len(X2),1) + (X2 - 2 * (X.dot(X.T)))
        K = KernelFunction(1, 0)**K
        
    else:
        K = np.zeros(m)
        for i in range(m):
            for j in range(m):
                K[i,j] = KernelFunction(X[i,:].T, X[j,:].T)
                K[j,i] = K[i,j]
    print 'Training ...'
    dots = 12
    while passes < max_passes:
        num_changed_alphas = 0
        for i in range(m):
            E[i] = b + np.sum(alphas*Y*K[:,i].reshape(m,1)) - Y[i,0]
            if (Y[i,0]*E[i,0] < -tol and alphas[i,0] < C) or \
                    (Y[i,0]*E[i] > tol and alphas[i,0] > 0):
                j = int(np.ceil(m*np.random.rand())-1)
                while j == i:
                    j = int(np.ceil(m*np.random.rand())-1)
                E[j,0] = b + np.sum(alphas*Y*K[:,j].reshape(m,1)) - Y[j,0]
                
                alphas_i_old = alphas[i,0] # very important!!!: the assigned value will change with the array change
                alphas_j_old = alphas[j,0] # unless using the element instead.
# when do the element operation, please always use full index like two indeces for 2d array.                
                
                if Y[i,0] == Y[j,0]:
                    L = max(0, (alphas[j,0] + alphas[i,0] - C))
                    H = min(C, (alphas[j,0] + alphas[i,0]))
                else:
                    L = max(0, (alphas[j,0] - alphas[i,0]))
                    H = min(C, (C + alphas[j,0] - alphas[i,0]))
                if L == H:
                    continue
                eta = 2*K[i,j] - K[i,i] - K[j,j]
                if eta >= 0:
                    continue
                alphas[j,0] = alphas[j,0] - (Y[j,0]*(E[i,0]-E[j,0]))/eta
                
                alphas[j,0] = min(H, alphas[j,0])
                alphas[j,0] = max(L, alphas[j,0])
                if np.abs(alphas[j,0]-alphas_j_old) < tol:
                    alphas[j,0] = alphas_j_old
                    continue
                
                alphas[i,0] = alphas[i,0] + Y[i,0]*Y[j,0]*(alphas_j_old - alphas[j,0])
                b1 = b - E[i,0] -Y[i,0]*(alphas[i,0]-alphas_i_old)*K[i,j].T - \
                    Y[j,0]*(alphas[j,0]-alphas_j_old)*K[i,j].T
                b2 = b - E[j,0] -Y[i,0]*(alphas[i,0]-alphas_i_old)*K[i,j].T - \
                    Y[j,0]*(alphas[j,0]-alphas_j_old)*K[j,j].T
                if 0 < alphas[i,0] and alphas[i,0] < C:
                    b = b1
                elif 0 < alphas[j,0] and alphas[j,0] < C:
                    b = b2
                else:
                    b = (b1+b2)/2
                num_changed_alphas = num_changed_alphas + 1
        
        if num_changed_alphas == 0:
            passes = passes +1
        else:
            passes = 0
        sys.stdout.write('.')
        dots = dots + 1
        if dots > 78:
            dots = 0
    print 'Done!'
    idx = (alphas > 0) + 0
    model = {}
    model['X']= X[np.where(idx != 0)[0],:]
    model['y']= Y[np.where(idx != 0)[0]]
    model['KernelFunction'] = KernelFunction
    model['b']= b
    model['alphas'] = alphas[np.where(idx != 0)[0]]
    model['w'] = ((alphas*Y).T.dot(X)).T;
    return model

                
                
                        
                    
        
    
    
