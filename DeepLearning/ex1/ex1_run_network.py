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
from ex1 import NeuralNetwork
import matplotlib.pyplot as plt

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


def plot_costs(train_cost, val_cost, BatchNorm = "With"): 
    '''
        Plots ths train and validation costs.
    '''
    plt.plot(np.arange(1, len(train_cost) +1), train_cost, label = "Train cost")
    plt.plot(np.arange(1, len(val_cost) +1), val_cost, label = "Val cost")
    plt.legend()
    plt.title(f"{BatchNorm} Batch Normalization")
    plt.xlabel("Iterations")
    plt.ylabel("cost")

def main(): 
    x_train, x_test, y_train, y_test = load_data()
    ## Running without batchnorm
    layers_dim = [x_train.shape[0], 20, 7, 5, 10]
    learning_rate = 0.009
    num_iterations = 3000
    batch_size = 512
    network = NeuralNetwork(use_batchnorm = True)
    parameters, costs, val_costs = network.L_layer_model(x_train, y_train, layers_dim,learning_rate, num_iterations, batch_size)
    accuracy = network.Predict(x_test, y_test, parameters)
    print(f"Test Accuracy is : {accuracy}")
    plot_costs(costs, val_costs)


if __name__ == "__main__":
    # execute only if run as a script
    main()