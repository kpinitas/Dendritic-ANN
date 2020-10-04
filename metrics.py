#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 03:14:45 2020

@author: kpinitas
"""

import cupy as cp
from sklearn.metrics import accuracy_score as acc
def accuracy(x,y):
    x = cp.argmax(x, axis=0).tolist()
    y = cp.argmax(y, axis=0).tolist()
    a = acc(x, y)
    print('acc: '+str(a))
    return a