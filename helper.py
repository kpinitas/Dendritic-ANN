#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 17:27:52 2020

@author: kpinitas
"""
import cupy as cp
#import skimage.measure as sk
import numpy as np
def is_int(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False
    
#    
#def max_pooling(X, kernel=(2,2)):
#    try:
#        dims3 = X.shape[2]
#    except IndexError:
#        dims3= False
#    if bool(dims3):
#
#        return np.array([ sk.block_reduce(a, kernel, np.max).tolist() for a in X])
#    return sk.block_reduce(X, kernel, np.max)
#        
#        