import matplotlib.pyplot as plt
import numpy as np

def PlotData(X, y):
    
    pos = np.where(y == 1)
    neg = np.where(y == 0)
    
    plt.plot(X[pos,0].ravel(), X[pos,1].ravel(), 'o', label = 'Admitted', color = 'red')
    plt.plot(X[neg,0].ravel(), X[neg,1].ravel(), '+', label = 'Not admitted', color = 'blue')
    plt.legend(bbox_to_anchor=(0.4, 1.25), loc = 1)
    
#    l1 = plt.plot(X[pos,0].ravel(), X[pos,1].ravel(), 'o', color = 'red')
#    l2 = plt.plot(X[neg,0].ravel(), X[neg,1].ravel(), '+', color = 'blue')
#    plt.legend((l1,l2), ('Admitted','Not admitted'), bbox_to_anchor=(0.4, 1.25), loc = 1)
#    
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    
