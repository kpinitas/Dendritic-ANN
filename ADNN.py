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


class Layer:
   def  __init__(self,input_dims,neurons, activation,dendrites_per_neuron=10,dens_activation = 'sigmoid'):
        self.input_dims = input_dims
        self.activation= activation
        self.neurons= neurons
        self.dendrites_per_neuron = dendrites_per_neuron
        
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
        self.init_params()
       
   def init_params(self):
       self.W['d'] = cp.random.normal(size=(self.neurons,self.dendrites_per_neuron,self.input_dims))
       self.W['s'] = cp.random.normal(size=(self.neurons,1,self.dendrites_per_neuron))
       
       self.b['d'] = cp.random.normal(size=(self.neurons,self.dendrites_per_neuron,1))
       self.b['s'] = cp.random.normal(size=(self.neurons,1,1))
       
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
        
   def backward(self, y,w=None,d=None):
       try:
            batch_size = y.shape[1]
       except IndexError:
            batch_size=1
         
       is_last_layer  = (type(w) == type(d) )and (type(d) == type(None))
       
       if is_last_layer:
            
            self.delta['s'] = cp.subtract(self.a['s'],y)
            self.dB['s'] = (1/batch_size)*cp.sum(self.delta['s'],axis=1)
            self.dB['s']= cp.reshape(self.dB['s'],(self.dB['s'].shape[0],1,1))
            
            self.delta['s'] = cp.reshape(self.delta['s'],(self.delta['s'].shape[0],1,self.delta['s'].shape[1]) )
            
            self.dW['s']=(1/batch_size)* cp.einsum('nik,kjn->nij',self.delta['s'],self.a['d'].T) 
            
       else:
            w=cp.array(w)
            
            deltaW = cp.einsum('nik,kij->nj',w.T,d)
            deltaW=cp.reshape(deltaW,(deltaW.shape[0],1,deltaW.shape[1]))
            a_der = activation(str(self.activation)+'_der',self.zeta['s'])
            
            self.delta['s']=cp.multiply(deltaW,a_der)
            self.dB['s'] = (1/batch_size)*cp.sum(self.delta['s'].squeeze(), axis=1)
            self.dB['s']= cp.reshape(self.dB['s'],(self.dB['s'].shape[0],1,1))
            self.dW['s']=(1/batch_size)* cp.einsum('nik,kjn->nij',self.delta['s'],self.a['d'].T)
            
       
       deltaW=cp.einsum('nik,kij->knj',self.W['s'].T,self.delta['s']) 
       a_der = activation(self.den_activation+'_der',self.zeta['d'])
       self.delta['d']=cp.multiply(deltaW,a_der)
       self.dB['d'] = (1/batch_size)*cp.sum(self.delta['d'], axis=2)
       self.dB['d']= cp.reshape(self.dB['d'],(self.dB['d'].shape[0],self.dB['d'].shape[1],1))
       self.dW['d']=(1/batch_size)*cp.dot(self.delta['d'],self.prev_layer.T)
       
       
   def update(self, lr=0.01):
       for key in self.b.keys():
           self.W[key]-=lr*self.dW[key]
           self.b[key]-=lr*self.dB[key]
        
        
class DNN:
    def __init__(self, hidden_layers=[32,64,128], activations=['sigmoid', 'sigmoid', 'sigmoid']):
        self.hidden_layers = hidden_layers
        self.activations = activations
        self.model=None
        self.init_model()
        self.y_pred=None
        
    def init_model(self):
        self.model = [Layer(input_dims = self.hidden_layers[i], neurons = self.hidden_layers[i+1],activation = self.activations[i]) for i in range(len(self.hidden_layers)-1)]  
        
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
            
        
    def backward(self,y):
        w=None
        d = None
        for i in range(len(self.model)-1,-1,-1):
            layer = self.model[i]
            layer.backward(y,w,d)
            w= layer.W['d']
            d= layer.delta['d']
    
    def update(self, lr=0.01):
        for layer in self.model:
            layer.update(lr)
    def train(self, epochs=10000, batch_size=5,lr=None, X=None, y=None,validation_set=None, lr_decay = 0):

        cost=[]
        acr = []
        acr_train=[]
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
            tr_pred= self.predict(X)
            a=accuracy(tr_pred,y)
            lr = (1-a)/4
            acr_train.append(a)
            if validation_set!=None:
                val_pred = self.predict(validation_set[0])
                a=accuracy(val_pred,validation_set[1])
                acr.append(a)  
            
            
            else:
                acr=None
        return cost, acr,acr_train
    
    def predict(self, X=None):
        self.forward(X)
        return self.y_pred