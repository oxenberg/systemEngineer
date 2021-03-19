#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 11:36:53 2021

@author: saharbaribi
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import tensorflow.keras as keras


class NeuralNetwork():
    def __init__(self, layers_dims):
        self.network = self.initialize_parameters(layers_dims)

    def fit(self, data):
        return

    #### Train and predict
    def L_layer_model(self, X, Y, layers_dims, learning_rate, num_iterations, batch_size):
        return

    def Predict(self, X, Y, parameters):
        return

    # Forward
    def initialize_parameters(self, layer_dims):
        '''

        create dict that represent the network layers
        {
            layer_n: {
                w: <matrix of weights from layer n-1 to n>
                row - current layer number of neuron, columns - previous
                b: <vector of bias from layer n-1 to n>
        }

        :param layer_dims: array of dimensions of each layer, layer 0 is size of input
        :return: network_weights: dictionary like in description that represent the network layers

        '''

        network_weights = {}
        for index in range(1, len(layer_dims)):
            layer_name = f"layer_{index}"

            prev_layer_dim = layer_dims[index-1]
            current_layer_dim = layer_dims[index]
            #: create weight matrix with current layer number of neuron, columns - previous
            w_matrix = np.random.rand(current_layer_dim, prev_layer_dim)
            bias_vector = np.random.rand(prev_layer_dim)

            network_weights[layer_name] = {"w": w_matrix, "b": bias_vector}

        return network_weights

    def linear_forward(self, A, W, b):
        '''

        :param A: activations of prev layer
        :param W: weight matrix of the current layer size of [row: current layer, columns: size of previous layer]
        :param b: bias vector of the current layer

        :return: Z: vector, the linear component of the activation function
        :return linear_cache: a dictionary containing A, W, b

        '''

        Z = np.dot(W, A) + b

        linear_cache = {"A": A, "W": W, "b": b}
        return Z, linear_cache

    def softmax(self, Z):
        '''
        softmax function

        :param Z: array of linear component

        :return: A: array of after activations
        :return: activation_cache: array of after activations


        '''
        A = []
        softmax_sum = 0
        # exponent for each output z
        for z_i in Z:
            a = np.exp(z_i)
            A.append(a)
            softmax_sum += a
        # transform to array and split by sum of exponents
        A = np.array(A)
        A = A/softmax_sum

        activation_cache = Z

        return A, activation_cache

    def relu(self, Z):
        '''

        relu function
        :param Z: array of linear component

        :return: A
        :return: activation_cache: same as softmax only with relu

        '''
        A = Z * (Z > 0)

        activation_cache = Z

        return A, activation_cache

    def linear_activation_forward(self, A_prev, W, B, activation):
        '''

        one layer forword propagation

        :param A_prev: activations of the previous layer
        :param W: the weights matrix of the current layer
        :param B: the bias vector of the current layer
        :param activation: the activation function to be used (a string, either “softmax” or “relu”)

        :return: A: array, activations of the current layer
        :return: cache: dictionary containing both linear_cache and activation_cache

        '''

        Z, linear_cache = self.linear_forward(A_prev, W, B)

        if activation == "softmax":
            A, activation_cache = self.softmax(Z)
        else:
            A, activation_cache = self.relu(Z)

        cache = {"linear_cache": linear_cache,
                 "activation_cache": activation_cache}
        return A, cache

    def L_model_forward(self, X, parameters, use_batchnorm):
        '''

        make forward propagation for epoch

        :param X:the data, numpy array of shape (input size, number of examples)
        :param parameters: dict, initialized W and b parameters of each layer
        :param use_batchnorm: a boolean flag used to determine whether to apply batchnorm
        :return: A the last post-activation value
        :return: caches

        '''

        caches = []

        #:TODO check if this is batch norm
        if use_batchnorm:
            col_sums = X.sum(axis=0)
            X = X / col_sums

        A = X.copy()

        for layer in parameters.values():

            A, cache = self.linear_activation_forward(
                A, layer["w"], layer["b"])

            if use_batchnorm:
                col_sums = A.sum(axis=0)
                A = A / col_sums

            caches.append(cache)

        return A, caches

    def compute_cost(self, AL, Y):
        '''

        :param AL: probability vector corresponding to your label predictions, shape (num_of_classes, number of examples)
        :param Y: the labels vector
        :return: cost the cross-entropy cost

        '''
        number_of_classes = AL.shape[0]
        number_of_examples = AL.shape[1]
        #: TODO check dim of AL

        # calculate by the cross entropy formula
        cost = 0
        for exemple in range(number_of_examples):
            for class_ in range(number_of_classes):
                y_tag = AL[number_of_classes][exemple]
                y_true = Y[number_of_classes][exemple]
                cost += y_true*np.log(y_tag)
        cost = -cost/number_of_examples

        return cost

    def apply_batchnorm(self, A):
        return

    # Backward:
    def Linear_backward(self, dZ, cache):
        # TODO: make sure shapes align

        dW = np.dot(dZ, np.transpose(cache["A_prev"]))
        db = dZ
        dA_prev = np.dot(dZ, np.transpose(cache['W']))

        return dA_prev, dW, db

    def linear_activation_backward(self, dA, cache, activation):
        '''
            activation is the activation function in the current layer.
            chache: will be a list of tuples - each tuple will have the activation cache as the firse element,
            and the linear cache as the second element
        '''
        linear_cache = cache['linear_cache']
        if activation == 'softmax':
            dZ = self.softmax_backward(dA, cache['activation_cache'])
            dA_prev, dW, db = self.Linear_backward(dZ, linear_cache)
        else:
            dZ = self.relu_backward(dA, cache['activation_cache'])
            dA_prev, dW, db = self.Linear_backward(dZ, linear_cache)

        return dA_prev, dW, db

    def relu_backward(self, dA, activation_cache):
        Z = activation_cache['Z']
        A, Z = self.relu(Z)
        dZ = dA * np.int64(A > 0)

        return dZ

    def softmax_backward(self, dA, activation_cache):
        dZ = dA
        return dZ

    def L_model_backward(self, AL, Y, caches):
        '''
        chaches: will be a list of dictionaries - each dictionart will have the activation cache 
        and the linear cache of each layer

        '''
        grads = {}
        layers = len(caches)

        dA = AL-Y
        grads["dA"+str(layers)]

        # The last layer:
        grads["dA"+str(layers-1)], grads["dW"+str(layers)], grads["db"+str(layers)] = \
            self.linear_activation_backward(dA, caches[layers-1], "softmax")

        # Rest of the layers:
        for layer in range(layers-1, 0, -1):
            grads["dA"+str(layer-1)], grads["dW"+str(layer)], grads["db"+str(layer)] = \
                self.linear_activation_backward(
                    grads["dA"+str(layer)], caches[layer-1], "relu")

        return grads

    def Update_parameters(self, parameters, grads, learning_rate):
        '''{
            layer_n: {
                w: <matrix of weights from layer n-1 to n>
                row - current layer number of neuron, columns - previous
                b: <vector of bias from layer n-1 to n>
        }'''
        layers = len(parameters)

        for layer in range(1, layers + 1):
            # update w
            parameters["layer_"+str(layer)]['w'] = parameters["layer_" +
                                                              str(layer)]['w'] - learning_rate*grads['dW'+str(layer)]
            # update b
            parameters["layer_"+str(layer)]['b'] = parameters["layer_" +
                                                              str(layer)]['b'] - learning_rate*grads['db'+str(layer)]

        return parameters
