# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 17:14:00 2017

@author: karln
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from process import get_binary_data

X, Y = get_binary_data()

def sigmoid(a):
    return 1 / (1 + np.exp(-a))

def forward(X, W, b):
    return sigmoid(X.dot(W) + b)

def classification_rate(Y, P):
    return np.mean( Y == P)

def cross_entropy(T, pY):
    return -np.mean(T*np.log(pY) + (1 - T)*np.log(1 - pY))

D = X.shape[1]
W = np.random.randn(D)
b = 0

learning_rate = 0.001

numi = 1000

class_rate = np.arange(numi, dtype=np.float)

for k in range(numi):
    if ( k % 100 == 0 ):
        X, Y = shuffle(X, Y)
        
        Xtrain = X[:-100]
        Ytrain = Y[:-100]
        Xtest=X[-100:]
        Ytest=Y[-100:]
        
        train_costs = []
        test_costs = []

#    for i in range(10):
    pYtrain = forward(Xtrain, W, b)
    pYtest = forward(Xtest, W, b)
    
    ctrain = cross_entropy(Ytrain, pYtrain)
    ctest = cross_entropy(Ytest, pYtest)
    train_costs.append(ctrain)
    test_costs.append(ctest)
    
    W -= learning_rate*Xtrain.T.dot(pYtrain - Ytrain)
    b -= learning_rate*(pYtrain - Ytrain).sum()
#        if ( i % 1000 == 0 ):
#            print( i, ctrain, ctest)
        
#    print("Final train classification rate:", classification_rate(Ytrain, np.round(pYtrain)))
#    print("Final test classification rate:", classification_rate(Ytest, np.round(pYtest)))
    pYtotal = forward(X, W, b)
    class_rate[k] = classification_rate(Y, np.round(pYtotal))
#    if ( k % 100 == 0 ):
#        print("Final total classification rate:", k, class_rate[k])
    
#    legend1, = plt.plot(train_costs, label='train cost')
#    legend2, = plt.plot(test_costs, label='test cost')
#    plt.legend([legend1, legend2])
#    plt.show()

#print("Final train classification rate:", classification_rate(Ytrain, np.round(pYtrain)))
#print("Final test classification rate:", classification_rate(Ytest, np.round(pYtest)))
#pYtotal = forward(X, W, b)
#print("Final total classification rate:", classification_rate(Y, np.round(pYtotal)))
plt.plot(class_rate)
plt.show()
