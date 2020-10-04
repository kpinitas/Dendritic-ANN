#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 18:10:00 2020

@author: kpinitas
"""
import cupy as cp
from activations import activation as activation_func
from losses import  loss as loss_func
from metrics import accuracy


class Layer:
    
    def __init__(self,layer_type = 's',input_dims = None,neurons = None, activation = 'sigmoid', dendrites=3):
        self.layer_type=layer_type
        self.input_dims = input_dims
        self.activation = activation
        self.neurons =neurons
        self.dendrites = dendrites
        if self.layer_type == 'd':
            self.neurons = self.dendrites*self.neurons
        
        self.W = None
        self.b=None
        self.init_params()
        self.prev_layer=None
        self.zeta = None
        self.a = None
        self.delta = None
        self.dW=None
        self.dB = None
        
    def init_params(self):
        self.W = cp.random.uniform(size=( self.neurons, self.input_dims))
        self.b = cp.random.uniform(size=(self.neurons, 1))
        
        if self.layer_type == 's':
            self.w_mask = cp.zeros(shape=( self.neurons, self.input_dims))
            
            for i in range(0,self.w_mask.shape[0]):
                self.w_mask[i,i*self.dendrites:(i+1)*self.dendrites]=1
            self.W[self.w_mask ==0]=0
            
            
    def forward(self,X):
        try:
            batch_size = X.shape[1]
        except IndexError:
            batch_size=1
            X = cp.reshape(X,(X.shape[0],batch_size))
        self.prev_layer = X 
        self.zeta =  cp.dot(self.W,X)+self.b
        self.a = activation_func(func=self.activation, x = self.zeta)
        
        
    def backward(self, y,w,d):
       try:
            batch_size = y.shape[1]
       except IndexError:
            batch_size=1
            y = cp.reshape(y,(y.shape[0],batch_size))
            
       if type(w)==type(None) and type(d)==type(None):
            self.delta = cp.subtract(self.a,y)
            self.dB = (1/batch_size)*cp.sum(self.delta,axis=1)
            self.dB= cp.reshape(self.dB,(self.dB.shape[0],1))
            self.dW=(1/batch_size)*cp.dot(self.delta,self.prev_layer.T) 
                    
       else:
            w=cp.array(w)
            deltaW = cp.matmul(w.T,d)
            self.delta=cp.multiply(deltaW,activation_func(str(self.activation)+'_der',self.zeta))
            self.dB = (1/batch_size)*cp.sum(self.delta, axis=1)
            self.dB= cp.reshape(self.dB,(self.dB.shape[0],1))
            self.dW=(1/batch_size)*cp.dot(self.delta,self.prev_layer.T)
        
 
    def update(self,lr):
       if self.layer_type == 's': 
           self.dW[self.w_mask ==0]=0
       self.W-=lr*self.dW
       self.b-=lr*self.dB
   
    
class DNN:
    
    def __init__(self,hidden_layers=None, activations = None,dims_out=None):
 
        self.hidden_layers = hidden_layers;
        self.activations = activations
        self.model=None
        self.dims_out=dims_out
        self.init_model()
        self.y_pred = None
        
        
    def init_model(self):
        self.model=[]
        for i in range(len(self.hidden_layers)-1):
            inp = self.hidden_layers[i]
            neur = self.hidden_layers[i+1]
            self.model.append(Layer(input_dims=inp, neurons=neur, layer_type='d'))
            dns = self.model[-1].dendrites
            self.model.append(Layer(input_dims=neur*dns, neurons=neur, layer_type='s',activation=self.activations[i],dendrites=dns))
        self.model.append(Layer(input_dims=self.hidden_layers[-1], neurons=self.dims_out, layer_type='o',activation='softmax'))
        
    def forward(self,X):
         try:
            batch_size = X.shape[1]
         except IndexError:
            batch_size=1
            X = cp.reshape(X,(X.shape[0],batch_size))

         for i in range(len(self.model)):
            forward_data = X
            for i in range(len(self.model)):
                layer=self.model[i]
                layer.forward(forward_data)
                forward_data = layer.a
            self.y_pred = forward_data
            
    def backward(self,y):
        w=None
        d = None
        for i in range(len(self.model)-1,-1,-1): 
          layer = self.model[i]
          layer.backward(y,w,d)     
          w,d =layer.W, layer.delta  
          
    def update(self, lr = 0.01):
        for layer in self.model:
            layer.update(lr)
            
            
    def train(self, epochs=10000, batch_size=5,lr=None, X=None, y=None,validation_set=None, lr_decay = 0):

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
                
                self.forward(Xb)
                self.backward(yb)
                if (i+1)%(int(epochs/5)) ==0:
                    lr = (1-lr_decay)*lr
                self.update(lr)
                loss.append(loss_func(x=self.y_pred,y=yb))
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
        self.forward(X)
        return self.y_pred