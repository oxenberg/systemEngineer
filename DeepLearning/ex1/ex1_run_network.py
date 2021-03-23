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

def load_data(): 
    (x_train, y_train), (x_test, y_test)   = keras.datasets.mnist.load_data(path='mnist.npz')
    
    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    y_train = y_train.T 
    y_test = y_test.T
    
    image_vector_size = 28*28
    x_train = x_train.reshape(image_vector_size, x_train.shape[0])
    x_test = x_test.reshape(image_vector_size, x_test.shape[0])
    
    ## Normalizing the input
    x_train = x_train /255.0
    x_test = x_test /255.0
    return x_train, x_test, y_train, y_test



def main(): 
    x_train, x_test, y_train, y_test = load_data()
    ## Running without batchnorm
    layers_dim = [x_train.shape[0], 20,7, 5, 10]
    # layers_dim = [x_train.shape[0], 4,4, 10]
    learning_rate = 0.009
    num_iterations = 10
    batch_size = 48000
    network = NeuralNetwork(use_batchnorm = False)
    parameters, costs, val_costs = network.L_layer_model(x_train, y_train, layers_dim,learning_rate, num_iterations, batch_size)
    accuracy = network.Predict(x_test, y_test, parameters)
    print(f"Test Accuracy is : {accuracy}")
    plt.plot(np.arange(1, len(costs) +1), costs)
    plt.plot(np.arange(1, len(val_costs) +1), val_costs)

if __name__ == "__main__":
    # execute only if run as a script
    main()