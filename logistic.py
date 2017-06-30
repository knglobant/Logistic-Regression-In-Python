# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 17:14:00 2017

@author: karln
"""
import numpy as np

N = 100
D = 2

X = np.random.randn(N,D)

X[:50,:] = X[:50,:] - 2*np.ones((50,D))
X[50:,:] = X[50:,:] + 2*np.ones((50,D))

T = np.array([0]*50 + [1]*50)

Xb= np.concatenate((np.array([[1]*N]).T, X), axis=1)

w = np.random.randn(D + 1)

def sigmoid(z):
    return 1/(1 + np.exp(-z))

Y = sigmoid(Xb.dot(w))

def cross_entropy(T, Y):
    E = 0
    for i in range(N):
        if T[i] == 1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1-Y[i])
    return E

learning_rate= 0.1


for i in range(N):
#    w += learning_rate * np.dot((T - Y).T, Xb)
    w += learning_rate * Xb.T.dot(-(Y-T))
    Y = sigmoid(Xb.dot(w))
    
print ("Final w:", w)
print ("Final ce:", cross_entropy(T, Y))

"""
print(cross_entropy(T,Y))
print(cross_entropy(T,Y))

w = np.array([0,4,0])

z = Xb.dot(w)
Y = sigmoid(z)

print(cross_entropy(T,Y))
"""