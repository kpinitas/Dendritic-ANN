#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 18:25:48 2020

@author: kpinitas
"""
from metrics import accuracy
from activations import activation,sigmoid_der
import cupy as cp
from losses import  loss as loss_func
class ANN:
    def __init__(self,neurons_per_layer,activation_functions, X_train=None,y_train=None, lr=0.01,initializer='random'):

        self.neurons_per_layer=neurons_per_layer
        self.lr = lr
        self.initializer=initializer
        self.activation_functions = activation_functions
        self.weights={}
        self.biases={}
        self.zeta={}
        self.layers={}
        self.deltas={};
        self.dW={};
        self.dB={};
        self.X_train = X_train
        self.y_train=y_train
        self.init_params(self.initializer)
        
        
    def init_params(self, initializer):
        for i in range(len(self.neurons_per_layer[1:])):
          
          if initializer == 'random':  
               self.weights[i] = cp.random.normal(size=(self.neurons_per_layer[i+1],self.neurons_per_layer[i]))
               # self.weights[i] = self.weights[i]/cp.sqrt(cp.sum(self.weights[i]))
               self.biases[i]=cp.random.random((self.neurons_per_layer[i+1],1))
          
          elif initializer == 'zero':
                
               self.weights[i] = cp.zeros(shape=(self.neurons_per_layer[i+1],self.neurons_per_layer[i]))
               # self.weights[i] = self.weights[i]/cp.sqrt(cp.sum(self.weights[i]))
               self.biases[i]=cp.zeros(shape=(self.neurons_per_layer[i+1],1))
           
     
    def forward_propagation(self,X):
        try:
            batch_size = X.shape[1]
        except IndexError:
            batch_size=1
            X = cp.reshape(X,(X.shape[0],batch_size))
        
        for i in range(len(self.neurons_per_layer)):
            if i==0:
                self.layers[i] = X
                self.zeta[i] = X
            elif i==len(self.neurons_per_layer)-1:
                self.zeta[i] = cp.dot(self.weights[i-1],self.layers[i-1])+self.biases[i-1]
                self.layers[i] = activation(func='softmax',x=self.zeta[i])
            else:
                self.zeta[i] = cp.dot(self.weights[i-1],self.layers[i-1])+self.biases[i-1]
                self.layers[i] = activation(func=self.activation_functions[i-1],x=self.zeta[i])

    
    
    def backward_propagation(self,y):
        
        layer_names =  len(list(self.layers.keys()))-1
        try:
            batch_size = y.shape[1]
        except IndexError as e:
            batch_size=1
            y = cp.reshape(y,(y.shape[0],batch_size))
        
        for i in range(layer_names,0,-1):
            if i==0:
                continue
            if i  == list(self.layers.keys())[-1]:
                self.dB[i-1] = (1/batch_size)*cp.sum(cp.subtract(self.layers[i],y),axis=1)
                self.dB[i-1]= cp.reshape(self.dB[i-1],(self.dB[i-1].shape[0],1))
                self.deltas[i] = cp.subtract(self.layers[i],y)
                self.dW[i-1] = (1/batch_size)*cp.dot(self.deltas[i],self.layers[i-1].T)
            #i=3...1
            else:
                self.deltas[i]=cp.multiply(cp.matmul(self.weights[i].T,self.deltas[i+1]),activation(str(self.activation_functions[i-1])+'_der',self.zeta[i]))
                self.dB[i-1] = (1/batch_size)*cp.sum(self.deltas[i], axis=1)
                self.dB[i-1]= cp.reshape(self.dB[i-1],(self.dB[i-1].shape[0],1))
                self.dW[i-1] = (1/batch_size)*cp.dot(self.deltas[i],self.layers[i-1].T)
#
    
    
    
    def update_params(self,lr=None):
        if lr==None:
            lr = self.lr
        for i in range(len(list(self.weights.keys()))):
            self.weights[i]-=lr*self.dW[i]
            self.biases[i]-=lr*self.dB[i]
   
    
    
    
    def check_param_dims(self):
        for i in list(self.weights.keys()):
            print(str(i)+': '+str(self.weights[i].shape)+' '+str(self.biases[i].shape)+' '+str(self.layers[i].shape))
        print(self.layers[list(self.layers.keys())[-1]].shape)
   
    def train(self, epochs=10000, batch_size=5,lr=None, X=None, y=None,validation_set=None, lr_decay = 0):

        if type(X) == type(None) and type(y)==type(None):
            X=self.X_train
            y=self.y_train
            lr = self.lr
        cost=[]
        acr = []
        for i in range(epochs):
            loss=[]
            batch_remainder = X.shape[1]%batch_size
            num_of_batches = (X.shape[1]//batch_size) if batch_remainder==0 else (X.shape[1]//batch_size)+1
            for j in range(num_of_batches):
                start = batch_size*j
                end = start+batch_size
                
                if batch_remainder==0 or (batch_remainder!=0 and i!=num_of_batches-1):
                    Xb = X[:,start:end]
                    yb = y[:,start:end]
                else:
                    Xb = X[:,X.shape[1]-batch_remainder:]
                    yb = y[:,X.shape[1]-batch_remainder:]
                
                self.forward_propagation(Xb)
                self.backward_propagation(yb)
                if (i+1)%(int(epochs/5)) ==0:
                    lr = (1-lr_decay)*lr
                self.update_params(lr)
                loss.append(loss_func(x=self.layers[list(self.layers.keys())[-1]],y=yb))
            epoch_loss = cp.mean(cp.array(loss))
            print('=== Epoch: '+str(i+1)+' ===')
            print('cost: ',epoch_loss)
            cost.append(epoch_loss.tolist())
            if validation_set!=None:
                val_pred = self.predict(validation_set[0])
                a=accuracy(val_pred,validation_set[1])
                acr.append(a)
            else:
                acr=None
        return cost, acr

    def predict(self, X=None):
        self.forward_propagation(X)
        return self.layers[list(self.layers.keys())[-1]]