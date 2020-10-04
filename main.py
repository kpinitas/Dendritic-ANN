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
from helper import is_int, max_pooling
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from ANN import ANN
from ADNN import DNN
def create_sets(dname='IRIS.csv',test_size=0.3):    
    
    if dname=='fashion.csv':
        from keras.datasets import fashion_mnist as fmnist
        from keras.utils import to_categorical
        (train_images, train_labels), (test_images, test_labels) = fmnist.load_data()
        
        print(train_images.shape, test_images.shape)
        train_images = train_images.reshape((train_images.shape[0], 28* 28))
        # train_images = train_images.astype('float32') / 255
        test_images = test_images.reshape((test_images.shape[0], 28 * 28))
        # test_images = test_images.astype('float32') / 255
        train_labels = to_categorical(train_labels)
        test_labels = to_categorical(test_labels)

        y_test = cp.array(test_labels.tolist())
        y_train = cp.array(train_labels.tolist())
        X_test = cp.array(test_images.tolist())
        X_train = cp.array(train_images.tolist())

        num_in = len(X_train[1])
        num_out = len(y_train[1])

        return num_in, num_out, X_train.T, X_test.T, y_train.T, y_test.T

    
    if dname=='mnist.csv':
        from keras.datasets import mnist
        from keras.utils import to_categorical

        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        # print(train_images.shape,train_images[0].shape)

        # train_images = max_pooling(train_images,kernel=(4,4))
        # test_images = max_pooling(test_images,kernel=(4,4))
        # print(train_images.shape,train_images[0].shape)
        
        train_images = train_images.reshape((60000, 28* 28))
        # train_images = train_images.astype('float32') / 255
        test_images = test_images.reshape((10000, 28 * 28))
        # test_images = test_images.astype('float32') / 255
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





dname = 'mnist.csv' #'IRIS.csv'
hidden_layers=[15]
activation_functions = ['sigmoid','softmax']
learning_params = [20,0.01,128,0.0]
neurons_per_layer=hidden_layers

num_in, num_out,X_train, X_test, y_train, y_test = create_sets(dname)
print('Load Data: OK!')
neurons_per_layer.append(num_out)

neurons_per_layer.insert(0,num_in)





dnn = DNN(hidden_layers = neurons_per_layer,activations = activation_functions)
print('Model Initialization: OK!')

validation_set=[X_test,y_test]


print('=====Trainning=====')
dnn_cost, dnn_acr, dnn_tacr=dnn.train(X=X_train,y=y_train,epochs=learning_params[0],validation_set=validation_set,batch_size=learning_params[2],lr=learning_params[1],lr_decay = learning_params[3] )


hidden_layers=[150,15]
neurons_per_layer=hidden_layers
neurons_per_layer.append(num_out)

neurons_per_layer.insert(0,num_in)

activation_functions = ['sigmoid','sigmoid','softmax']


ann =   ANN(neurons_per_layer,activation_functions)
print('Model Initialization: OK!')

validation_set=[X_test,y_test]


print('=====Trainning=====')
ann_cost, ann_acr, ann_tacr=ann.train(X=X_train,y=y_train,epochs=learning_params[0],validation_set=validation_set,batch_size=learning_params[2],lr=learning_params[1],lr_decay = learning_params[3] )





hidden_layers=[15]
neurons_per_layer=hidden_layers
neurons_per_layer.append(num_out)

neurons_per_layer.insert(0,num_in)

activation_functions = ['sigmoid','softmax']



slp = ANN(neurons_per_layer,activation_functions)
print('Model Initialization: OK!')

validation_set=[X_test,y_test]


print('=====Trainning=====')
slp_cost, slp_acr, slp_tacr=slp.train(X=X_train,y=y_train,epochs=learning_params[0],validation_set=validation_set,batch_size=learning_params[2],lr=learning_params[1],lr_decay = learning_params[3] )



plt.figure(1)
plt.title('Loss Function')
ann_plt, =plt.plot(list(range(len(ann_cost))),ann_cost)
dnn_plt, =plt.plot(list(range(len(dnn_cost))),dnn_cost)
slp_plt, = plt.plot(list(range(len(slp_cost))),slp_cost)
plt.legend([ann_plt,dnn_plt,slp_plt], ['DNN', 'ANN', '1-Layer Perceptron'])
plt.show()
# plt.savefig('./images/MNIST/cost.py')


plt.figure(1)
plt.title('Accuracy on Train')
ann_plt, =plt.plot(list(range(len(ann_tacr))),ann_tacr)
dnn_plt, =plt.plot(list(range(len(dnn_tacr))),dnn_tacr)
slp_plt, = plt.plot(list(range(len(slp_tacr))),slp_tacr)
plt.legend([ann_plt,dnn_plt,slp_plt], ['DNN', 'ANN', '1-Layer Perceptron'])
plt.show()
# plt.savefig('./images/MNIST/train_acc.py')


plt.figure(2)
if validation_set!=None:
    plt.title('Accuracy on Test')
    ann_plt, = plt.plot(list(range(len(ann_acr))),ann_acr)
    dnn_plt, = plt.plot(list(range(len(dnn_acr))),dnn_acr)
    slp_plt, = plt.plot(list(range(len(slp_acr))),slp_acr)
    plt.legend([ann_plt,dnn_plt,slp_plt], ['DNN', 'ANN', '1-Layer Perceptron'])
    plt.show()
    # plt.savefig('./images/MNIST/test_acc.py')
