#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 17:40:57 2021

@author: saharbaribi
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow.keras as keras
from ex1 import NeuralNetwork
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def initialize_parameters_deep(layer_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)            # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01 
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters



# For softmax, Z should be a (# of classes, m) matrix
# Recall that axis=0 is column sum, while axis=1 is row sum
def softmax(Z):
    t = np.exp(Z)
    t = t / t.sum(axis=0, keepdims=True)
    return t

def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A

def relu(Z):
    A = np.maximum(0,Z)    
    assert(A.shape == Z.shape)
    return A

def linear_activation_forward(A_prev, W, b, activation):
    
    if activation == "sigmoid":
        Z = np.dot(W, A_prev) + b
        A = sigmoid(Z)
    
    elif activation == "relu":
        Z = np.dot(W, A_prev) + b
        A = relu(Z)
        
    elif activation == "softmax":
        Z = np.dot(W, A_prev) + b
        A = softmax(Z)
    
    # Some assertions to check that shapes are right
    assert(Z.shape == (W.shape[0], A.shape[1]))
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    
    # Cache the necessary values for back propagation later
    cache = (A_prev, W, b, Z)

    return A, cache

def L_model_forward(X, parameters):

    caches = []
    A = X
    L = len(parameters) // 2                  # number of hidden layers in the neural network
    
    # Hidden layers 1 to L-1 will be Relu.
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], activation="relu")
        caches.append(cache)
        
    # Output layer L will be softmax
    AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], activation="softmax")
    caches.append(cache)
    
    assert(AL.shape == (10, X.shape[1]))
            
    return AL, caches


def compute_cost(AL, Y):
    m = Y.shape[1]
    
    cost = -1/m * np.sum(np.multiply(Y, np.log(AL)))    
    cost = np.squeeze(cost)      # To coerce data from [[17]] into 17
    assert(cost.shape == ())
    
    return cost


def linear_backward(dZ, A_prev, W, b):
    m = A_prev.shape[1]

    dW = 1/m * np.dot(dZ, A_prev.T)
    db = 1/m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db

def relu_backward(dA, cache):
    A_prev, W, b, Z = cache
    
    # Compute dZ
    dZ = np.array(dA, copy=True) # convert dz to a numpy array
    dZ[Z <= 0] = 0
    assert (dZ.shape == Z.shape)
    
    # Compute dA_prev, dW, db
    dA_prev, dW, db = linear_backward(dZ, A_prev, W, b)
    return dA_prev, dW, db

def softmax_backward(AL, Y, cache):
    A_prev, W, b, Z = cache
    
    # Compute dZ
    dZ = AL - Y
    
    # Compute dA_prev, dW, db
    dA_prev, dW, db = linear_backward(dZ, A_prev, W, b)
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Backpropagation at layer L-1
    # The activation is softmax at layer L-1
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = softmax_backward(AL, Y, current_cache)
    
    # Backpropagation from layers L-2 to 1
    # The activations are relu at all these layers
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = relu_backward(grads["dA" + str(l+1)], current_cache)
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2 # number of layers in the neural network
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
    
    return parameters


def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    np.random.seed(1)
    costs = []                         

    # Step a: Initialise Parameters
    parameters = initialize_parameters_deep(layers_dims)
    
    # Iterative loops of gradient descent
    for i in range(0, num_iterations):

        # Step b: Forward Propagation
        AL, caches = L_model_forward(X, parameters)
        
        # Step c: Compute cost
        cost = compute_cost(AL, Y)
        
        # Step d: Backward Propagation
        grads = L_model_backward(AL, Y, caches)
        
        # Step e: Update parameters
        parameters = update_parameters(parameters, grads, learning_rate)
                
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 10 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters

def predict(X, parameters):
    # Forward propagation
    probabilities, caches = L_model_forward(X, parameters)
    
    # Calculate Predictions (the highest probability for a given example is coded as 1, otherwise 0)
    predictions = (probabilities == np.amax(probabilities, axis=0, keepdims=True))
    predictions = predictions.astype(float)

    return predictions, probabilities

def evaluate_prediction(predictions, Y):
    m = Y.shape[1]
    predictions_class = predictions.argmax(axis=0).reshape(1, m)
    Y_class = Y.argmax(axis=0).reshape(1, m)
    
    return np.sum((predictions_class == Y_class) / (m))


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


def one_hot_array(Y):
    b = np.zeros((Y.size, Y.max() + 1))
    b[np.arange(Y.size), Y] = 1
    return b.T

def plot_digit(index):
    im = train.iloc[index, 1:]
    digit = train.iloc[index, 0]
    im_reshape = im.values.reshape(28, 28)
    plt.imshow(im_reshape, cmap='Greys')
    plt.title("The label is: " + str(digit))



train = pd.read_csv("mnist_train.csv")
X = train.iloc[:, 1:].values.T
Y = train.iloc[:, 0]
Y_onehot = one_hot_array(Y.values)
print("Shape of X is: ", str(X.shape))
print("Shape of Y is: ", str(Y_onehot.shape))
x_train = X[:, 0:5000]
x_test = X[:, 5000:10000]
y_train = Y_onehot[:, 0:5000]
y_test = Y_onehot[:, 5000:10000]
print("Shape of X_train is: " + str(x_train.shape))
print("Shape of X_test is: " + str(x_test.shape))
print("Shape of Y_train is: " + str(y_train.shape))
print("Shape of Y_test is: " + str(y_test.shape))
# (x_train, y_train), (x_test, y_test)   = keras.datasets.mnist.load_data(path='mnist.npz')
# y_train = one_hot_array(y_train)
# y_test = one_hot_array(y_test)
# image_vector_size = 28*28
# x_train = np.array([x_train[i].flatten() for i in range(0,x_train.shape[0])])
# x_test = np.array([x_test[i].flatten() for i in range(0,x_test.shape[0])])
# x_train = x_train.T
# x_test = x_test.T
# x_train = x_train.flatten().reshape(image_vector_size, x_train.shape[0])
# x_test = x_test.flatten().reshape(image_vector_size, x_test.shape[0])

# plot_digit(20)
# x_train, x_test, y_train, y_test = load_data()
# X_train, X_val, Y_train, Y_val = train_test_split(x_train.T, y_train.T, test_size = 0.2, random_state = 1)
# x_train, X_val, y_train, Y_val = X_train.T, X_val.T, Y_train.T, Y_val.T
# parameters = initialize_parameters_deep([784, 20, 7, 5, 10])
# print(parameters.keys())
# AL, caches = L_model_forward(x_train, parameters)
# print(pd.DataFrame(AL[:, 0:5]))
# compute_cost(AL, y_train) # initial cost of 2.3

# grads = L_model_backward(AL, y_train, caches)
# grads.keys()

# parameters = update_parameters(parameters, grads, learning_rate=0.009)
# AL, caches = L_model_forward(x_train, parameters)
# print(pd.DataFrame(AL[:, 0:5]))
layers_dims = [784, 10, 10]
parameters = L_layer_model(x_train, y_train, layers_dims, learning_rate = 0.001, num_iterations = 500, print_cost=True)

pred_train, probs_train = predict(x_train, parameters)
print("Train set error is: " + str(evaluate_prediction(pred_train, y_train)))
pred_test, probs_test = predict(x_test, parameters)
print("Test set error is: " + str(evaluate_prediction(pred_test, y_test)))
