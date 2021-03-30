#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 13:07:13 2021

@author: saharbaribi
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt

from Ex1.ex1_dropout import run_with_dropout
from Ex1.ex1_functions import run_NN


def one_hot_transform(Y):
    ''' Transforms the data into one hot encoded matrix 
        Input: 
            Y - Data to transform
        Output: 
            Y_one_hot.T - a transposed one encoded matrix.
    '''
    Y_one_hot = np.zeros((Y.size, Y.max() + 1))
    Y_one_hot[np.arange(Y.size), Y] = 1
    return Y_one_hot.T

def load_data(): 
    ''' Reads the data from Keras dataset, transforms the y data into one hot encoded matrices, 
        flattens the X matrices, and normalizes them. 
        Output: 
            x_train, x_test - Flattend and normalized X matrices
            y_train, y_test - one hot encoded y matrices
    '''
    (x_train, y_train), (x_test, y_test)   = keras.datasets.mnist.load_data(path='mnist.npz')
    
    num_classes = 10
    y_train = one_hot_transform(y_train)
    y_test = one_hot_transform(y_test)
    
    x_train = np.array([x_train[i].flatten() for i in range(0,x_train.shape[0])])
    x_test = np.array([x_test[i].flatten() for i in range(0,x_test.shape[0])])
    x_train = x_train.T
    x_test = x_test.T
    
    ## Normalizing the input
    x_train = x_train /255.0
    x_test = x_test /255.0
    return x_train, x_test, y_train, y_test


def plot_costs(train_cost, val_cost, title): 
    '''
        Plots ths train and validation costs.
    '''
    plt.plot(np.arange(1, (len(train_cost) +1))*100, train_cost, label = "Train cost")
    plt.plot(np.arange(1, (len(val_cost) +1))*100, val_cost, label = "Val cost")
    plt.legend()
    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel("cost")
    plt.show()

def main(): 
    x_train, x_test, y_train, y_test = load_data()

    params = {
        "x_train":x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test" : y_test,
        "layers_dim" : [x_train.shape[0], 20, 7, 5, 10],
        "learning_rate" : 0.009
    }

    ## Running without batchnorm
    print("Running a network without Batch Normalization")
    params["num_iterations"] = 14000
    params["batch_size"] = 600
    run_NN(**params, use_batch_norm=False,title = "Without Batch Normalization")

    print("Running a network with Batch Normalization")
    params["num_iterations"] = 85000
    params["batch_size"] = 900
    run_NN(**params, use_batch_norm=True, title = "With Batch Normalization")

    print("Running a network with dropout")
    run_with_dropout(**params,dropout=[0,0.1,0,0])
    params["num_iterations"] = 11000
    params["batch_size"] = 800

if __name__ == "__main__":
    # execute only if run as a script
    main()