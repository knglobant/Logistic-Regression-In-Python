# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 17:14:00 2017

@author: karln
"""
import numpy as np
import matplotlib.pyplot as plt

N = 100
D = 2

X = np.random.randn(N,D)

X[:50,:] = X[:50,:] - 2*np.ones((50,D))
X[50:,:] = X[50:,:] + 2*np.ones((50,D))

T = np.array([0]*50 + [1]*50)

ones = np.array([[1]*N]).T
Xb= np.concatenate((ones, X), axis=1)

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def cross_entropy(T, Y):
    E = 0
    for i in range(N):
        if T[i] == 1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1-Y[i])
    return E

w = np.array([0,4,4])

plt.scatter(X[:,0], X[:,1], c=T, s=100, alpha=.5)

x_axis = np.linspace(-6,6,100)
y_axis = -x_axis

plt.plot(x_axis, y_axis)

plt.show()

z = Xb.dot(w)
Y = sigmoid(z)

print(cross_entropy(T,Y))
