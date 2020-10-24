#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 17:37:42 2020

@author: kpinitas
"""

from metrics import accuracy
from activations import activation
import cupy as cp
from losses import  loss as loss_func
from learning_rules import  learning_rule

class Layer:
   def  __init__(self,input_dims,neurons, activation,dendrites_per_neuron=7,dens_activation = 'sigmoid',initializer = 'random',rule='classic'):
        self.input_dims = input_dims
        self.activation= activation
        self.neurons= neurons
        self.dendrites_per_neuron = dendrites_per_neuron
        self.rule=rule
        self.prev_layer=None
        self.W={}
        self.b ={}
        self.zeta = {}
        self.a = {}
        self.y_pred = None
        self.den_activation = dens_activation
        self.delta={}
        self.dW={}
        self.dB={}
        self.init_params(initializer)
       
   def init_params(self,initializer):
       
       if initializer =='random':
           self.W['d'] = cp.random.normal(size=(self.neurons,self.dendrites_per_neuron,self.input_dims))
           self.W['s'] = cp.random.normal(size=(self.neurons,1,self.dendrites_per_neuron))
           
           self.b['d'] = cp.random.normal(size=(self.neurons,self.dendrites_per_neuron,1))
           self.b['s'] = cp.random.normal(size=(self.neurons,1,1))
       
        
       elif initializer == 'zero':
           self.W['d'] = cp.zeros(shape=(self.neurons,self.dendrites_per_neuron,self.input_dims))
           self.W['s'] = cp.zeros(shape=(self.neurons,1,self.dendrites_per_neuron))
           
           self.b['d'] = cp.zeros(shape=(self.neurons,self.dendrites_per_neuron,1))
           self.b['s'] = cp.zeros(shape=(self.neurons,1,1))
           
       elif initializer=='uniform':
           self.W['d'] = cp.random.uniform(size=(self.neurons,self.dendrites_per_neuron,self.input_dims))
           self.W['s'] = cp.random.uniform(size=(self.neurons,1,self.dendrites_per_neuron))
           
           self.b['d'] = cp.random.uniform(size=(self.neurons,self.dendrites_per_neuron,1))
           self.b['s'] = cp.random.uniform(size=(self.neurons,1,1))
           
   def forward(self,X):
       try:
            batch_size = X.shape[1]
       except IndexError:
            batch_size=1
            X=cp.reshape(X,(X.shape[0],batch_size))
       
       self.prev_layer =X 
       self.zeta['d']  = cp.dot(self.W['d'],self.prev_layer)+self.b['d']
       self.a['d'] = activation(func = self.den_activation,x=self.zeta['d'])
        
      
       self.zeta['s']  = cp.einsum('nik,nkj->nij',self.W['s'],self.a['d']) +self.b['s']
       self.a['s'] = activation(func = self.activation,x=self.zeta['s']).squeeze()
       return self.a['s']
        
   def backward(self, y,w=None,d=None,rule=None):
       
       rule=self.rule if type(rule) == type(None) else rule
       self.calculate_derivatives(y,w,d,rule)
       
       
   def update(self, lr=0.01,rule = None):
       rule=self.rule if type(rule) == type(None) else rule
       new_params = learning_rule[rule]['update'](lr,self.W,self.b, self.dW,self.dB)
       self.W= new_params[0]
       self.b = new_params[1]
        
 
        
 
   def calculate_derivatives(self,y,w=None,d = 'None', rule= None):
       rule=self.rule if type(rule) == type(None) else rule
       derivatives=learning_rule[rule]['calc'](self.W,self.b,self.zeta,self.a,self.prev_layer,self.activation,self.den_activation, y,w,d)
       self.dW = derivatives[0]
       self.dB=derivatives[1]
       self.delta = derivatives[2]
           
       




class DNN:
    def __init__(self, hidden_layers=[32,64,128], activations=['sigmoid', 'sigmoid', 'sigmoid'],initializer = 'random',rule='classic'):
        self.hidden_layers = hidden_layers
        self.activations = activations
        self.model=None
        self.rule = rule
        self.initializer =initializer
        self.init_model()
        self.y_pred=None
        
    def init_model(self):
        self.model = [Layer(input_dims = self.hidden_layers[i], neurons = self.hidden_layers[i+1],activation = self.activations[i],initializer = self.initializer,rule = self.rule) for i in range(len(self.hidden_layers)-1)]  
        
    def forward(self, X):
        try:
            batch_size = X.shape[1]
        except IndexError:
            batch_size=1
            X = cp.reshape(X,(X.shape[0],batch_size))

        for i in range(len(self.model)):
            forward_data = X
            for layer in self.model:
                layer.forward(forward_data) 
                forward_data = layer.a['s']
            self.y_pred = forward_data
            
        
    def backward(self,y,rule = None):
        w=None
        d = None
        rule = self.rule if type(rule)==type(None) else rule
        for i in range(len(self.model)-1,-1,-1):
            layer = self.model[i]
            layer.backward(y,w,d)
            w= layer.W['d']
            d= None if type(layer.delta)== type(None) else layer.delta['d']
    
    def update(self, lr=0.01,rule=None):
        rule = self.rule if type(rule)==type(None) else rule
        for layer in self.model:
            layer.update(lr)
    def train(self, epochs=10000, batch_size=5,lr=None, X=None, y=None,validation_set=None, lr_decay = 0,rule=None):
        
        rule = self.rule if type(rule)==type(None) else rule
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