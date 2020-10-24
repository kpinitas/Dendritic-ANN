#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 15:47:45 2020

@author: kpinitas
"""
import matplotlib.pyplot as plt
import cupy as cp
import numpy as np
import sys
from helper import is_int
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from ANN import ANN
from ADNN import DNN
from keras.datasets import mnist
from keras.utils import to_categorical
def create_sets(dname='IRIS.csv',test_size=0.3):    
    if dname=='mnist.csv':
        

        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        train_images = train_images.reshape((60000, 28 * 28))
        train_images = train_images.astype('float32') / 255
        test_images = test_images.reshape((10000, 28 * 28))
        test_images = test_images.astype('float32') / 255
        train_labels = to_categorical(train_labels)
        test_labels = to_categorical(test_labels)

        y_test = cp.array(test_labels.tolist())
        y_train = cp.array(train_labels.tolist())
        X_test = cp.array(test_images.tolist())
        X_train = cp.array(train_images.tolist())

        num_in = len(X_train[1])
        num_out = len(y_train[1])

        return num_in, num_out, X_train.T, X_test.T, y_train.T, y_test.T

    dataset = pd.read_csv(dname)
    dataset = shuffle(dataset)
    dataset.fillna(0)

    if dname == 'IRIS.csv':
        labels = dataset.iloc[:, -1]
        dataset= dataset.iloc[:,:-1]
        labels = pd.get_dummies(labels,drop_first=False)
        X = dataset.values.tolist()
        y = labels.values.tolist()


    num_in =len(X[1])
    num_out = len(y[1])


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    X_train = cp.array(X_train)
    X_test = cp.array(X_test)
    y_train = cp.array(y_train)
    y_test = cp.array(y_test)

    return num_in, num_out, X_train.T, X_test.T, y_train.T, y_test.T





DEBUG=1

if DEBUG==0:
    dname = sys.argv[1]
    hidden_layers =  sys.argv[2].strip('[]').split(',')
    hidden_layers = [int(h) for h in hidden_layers]
    activation_functions = sys.argv[3].strip('[]').split(',')
    learning_params =sys.argv[4].strip('[]').split(',')
    learning_params = [int(l) if is_int(l) else float(l) for l in learning_params]
else:
    dname = 'mnist.csv' #'IRIS.csv'
    hidden_layers=[30]
    activation_functions = ['sigmoid','softmax']#,'softmax']
    learning_params = [150,0.01,2,0.0]
neurons_per_layer=hidden_layers

num_in, num_out,X_train, X_test, y_train, y_test = create_sets(dname)

neurons_per_layer.append(num_out)
neurons_per_layer.insert(0,num_in)

#net = ANN(neurons_per_layer,activation_functions, X_train,y_train)
net = DNN(neurons_per_layer,activation_functions,initializer = 'random',rule='classic')
validation_set=[X_test,y_test]

learning_params[1] = -learning_params[1] if net.rule=='hebbian' else learning_params[1]


print('Training starts')
cost, acr=net.train(X=X_train, y=y_train, epochs=learning_params[0],validation_set=validation_set,batch_size=learning_params[2],lr=learning_params[1],lr_decay = learning_params[3] )

plt.title('Loss Function')
plt.plot(list(range(len(cost))),cost)
plt.show()

if acr!=None:
    plt.title('Accuracy')
    plt.plot(list(range(len(acr))),acr)
    plt.show()

