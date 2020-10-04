#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 15:36:41 2020

@author: kpinitas
"""

import cupy as cp
def sigmoid(x):
   return 1/(1+cp.exp(-x))


def sigmoid_der(x):
    sigm = 1/(1+cp.exp(-x))
    return sigm*(1-sigm)

def relu(x):
    return x * (x > 0)

def relu_der(x):
    return 1.0*(x>0)

def tanh(x):
    return cp.tanh(x)
def tanh_der(x):
    return 1-x**2

def softmax(x,axis=0):
    tmp=cp.exp(x)
    return tmp/cp.sum(tmp,axis=axis)

def sigmoidal_relu(x):
    xr=relu(x)
    return sigmoid(xr)

def sigmoidal_relu_der(x):
    return sigmoid_der(relu(x))*relu_der(x)

def activation(func = 'sigmoid', x=None):
    x=cp.array(x);
    assert x.all().tolist()!= None,"At least one \"None\" in input data"
    if func == 'sigmoid':
        return sigmoid(x)
    elif func == 'softmax':
        return softmax(x, axis=0)
    elif func=='sigmoid_der':
        return sigmoid_der(x)
    elif func == 'relu':
        return relu(x)
    elif func == 'relu_der':
        return relu_der(x)
    elif func=='tanh':
        return tanh(x)
    elif func== 'tanh_der':
        return tanh_der(x)
    elif func=='sigmoidalrelu':
        return sigmoidal_relu(x)
    elif func=='sigmoidalrelu_der':
        return sigmoidal_relu_der(x)
    else:
        return None