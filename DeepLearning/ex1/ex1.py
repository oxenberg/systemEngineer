#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 11:36:53 2021

@author: saharbaribi
"""


import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import tensorflow as tf

def create_batches(X,Y, batch_size): 
    ''' Creates and returns two list of batches, for X and Y. 
        Input: 
            X, Y - The data 
            batch_size - The size of the required batch. 
        Output: 
            batches_X, batches_y - a list of batches in the required size. 
    '''
    data_size = X.shape[1]
    indexes = np.arange(data_size)
    np.random.shuffle(indexes)
    batches_X = []
    batches_y = []
    prev_i = 0
    for i in np.arange(batch_size,data_size+1, batch_size): 
        batch_indexes = indexes[prev_i:i]
        batches_X.append(X[:,batch_indexes])
        batches_y.append(Y[:,batch_indexes])
        
        prev_i = i
    
    return batches_X, batches_y

class NeuralNetwork():
    def __init__(self, use_batchnorm = False):
        self.use_batchnorm  = use_batchnorm
        self.epsilon = 0.00001
    
    #### Train and predict
    def L_layer_model(self, X, Y, layers_dims, learning_rate, num_iterations, batch_size):
        X_train, X_val, Y_train, Y_val = train_test_split(X.T, Y.T, test_size = 0.2, random_state = 1)
        X_train, X_val, Y_train, Y_val = X_train.T, X_val.T, Y_train.T, Y_val.T
        parameters = self.initialize_parameters(layers_dims)
        costs = []
        val_costs = []
        epochs = 1500
        num_labels = Y_train.shape[0]
        assert num_labels == layers_dims[-1]
        
        iterations = 0
        val_prev_cost = np.inf
        done = False
        
        # TODO: understand whether it should be epochs or num of iteration and the diff between them. 
        batches_X, batches_y = create_batches(X_train,Y_train, batch_size)
        
        for i in tqdm(range(epochs)): 
            for batch_x,batch_y in zip(batches_X, batches_y): 
                
                iterations+=1
                ## Predict values
                y_predicted, caches = self.L_model_forward(batch_x, parameters, self.use_batchnorm)
                cost = self.compute_cost(y_predicted, batch_y)
                grads = self.L_model_backward(y_predicted, batch_y, caches)
                parameters = self.Update_parameters(parameters, grads, learning_rate)
                
                if iterations % 100 == 0: 
                    costs.append(cost)

                    val_predicted, val_caches = self.L_model_forward(X_val, parameters, self.use_batchnorm)
                    val_cost = self.compute_cost(val_predicted, Y_val)
                    val_costs.append(val_cost)
                    
                    print(f"iteration number:{iterations}, train cost: {cost}, validation cost: {val_cost}")
                    # : Stopping criterion
                    if val_cost>val_prev_cost or iterations>num_iterations: 
                        done = True
                        break
                    val_prev_cost = val_cost
            
            #: Stopping criterion
            if done:
                 break
        #TODO: remove val costs from the return
        train_accuracy = self.Predict(X_train, Y_train, parameters)
        print(f"Train Accuracy is : {train_accuracy}")
        validation_accuracy = self.Predict(X_val, Y_val, parameters)
        print(f"Validation Accuracy is : {validation_accuracy}")
        return parameters, costs, val_costs

    def Predict(self, X, Y, parameters):
        y_predicted, caches = self.L_model_forward(X, parameters, self.use_batchnorm)
        y_predicted = np.argmax(y_predicted, axis = 0)
        Y = np.argmax(Y, axis = 0)
        matches = np.where(Y==y_predicted)[0]
        accuracy = len(matches)/len(Y)
        
        return accuracy 

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
        np.random.seed(3)

        network_weights = {}
        for index in range(1, len(layer_dims)):
            layer_name = f"layer_{index}"

            prev_layer_dim = layer_dims[index-1]
            current_layer_dim = layer_dims[index]
            # initializer = tf.keras.initializers.GlorotUniform()

            #: create weight matrix with current layer number of neuron, columns - previous
            w_matrix = np.random.randn(current_layer_dim, prev_layer_dim)
            # w_matrix = np.array(initializer(shape = (current_layer_dim, prev_layer_dim)))
            w_matrix = w_matrix*np.sqrt(1/prev_layer_dim)

            bias_vector = np.zeros((current_layer_dim,1))
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
        e_x = np.exp(Z - np.max(Z))
        A = e_x / e_x.sum(axis=0)
        
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
        # if use_batchnorm:
        #     A = self.apply_batchnorm(X)

        A = X.copy()
        parameters_values_list = list(parameters.values())
        
        for layer in parameters_values_list[:-1]:
            
            A, cache = self.linear_activation_forward(
                A, layer["w"], layer["b"], "relu")

            if use_batchnorm:
                A = self.apply_batchnorm(A)

            caches.append(cache)
            
        softmax_layer = parameters_values_list[-1]
        A, cache = self.linear_activation_forward(
                A, softmax_layer["w"], softmax_layer["b"], "softmax")
        caches.append(cache)
        
        return A, caches

    def compute_cost(self, AL, Y):
        '''

        :param AL: probability vector corresponding to your label predictions, shape (num_of_classes, number of examples)
        :param Y: the labels vector
        :return: cost the cross-entropy cost

        '''
        number_of_examples = AL.shape[1]
        cost = - (1 / number_of_examples) * np.sum(
        np.multiply(Y, np.log(AL)))
        
        return cost

    def apply_batchnorm(self, A):
        mu = np.mean(A, axis = 0)
        var = np.var(A, axis = 0)

        NA = (A-mu)/np.sqrt(var+self.epsilon)

        return NA

    # Backward:
    def Linear_backward(self, dZ, cache):
        num_examples = cache['A'].shape[1]
        dW = (1/num_examples)*np.dot(dZ, np.transpose(cache["A"]))
        db = (1/num_examples)*np.sum(dZ, axis = 1, keepdims = True)
        dA_prev = np.dot(cache['W'].T, dZ)
        
        assert dA_prev.shape == cache['A'].shape
        assert dW.shape == cache['W'].shape
        assert db.shape == cache['b'].shape
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
        Z = activation_cache

        dZ = dA.copy()
        dZ[Z <= 0] = 0

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

        for layer in range(1, layers+1):
            # update w
            parameters["layer_"+str(layer)]['w'] = parameters["layer_" +
                                                              str(layer)]['w'] - learning_rate*grads['dW'+str(layer)]
            # update b
            parameters["layer_"+str(layer)]['b'] = parameters["layer_" +
                                                              str(layer)]['b'] - learning_rate*grads['db'+str(layer)]

        return parameters





class NueralNetworkDropout(NeuralNetwork): 
    def __init__(self, use_batchnorm = False, dropout_rate = [1, 0.2, 1, 1]):
        super().__init__(use_batchnorm)
        self.droupout = dropout_rate 
        self.predict = False
        
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

        A = X.copy()
        parameters_values_list = list(parameters.values())
        
        for layer, dropout_rate in zip(parameters_values_list[:-1], self.droupout[:-1]):
            if self.predict: 
                dropout_mask = dropout_rate
            else: 
                dropout_mask = np.random.binomial(1, dropout_rate, size = A.shape)
            A*=dropout_mask
            A, cache = self.linear_activation_forward(
                A, layer["w"], layer["b"], "relu")
            
            if use_batchnorm:
                A = self.apply_batchnorm(A)
            
            caches.append(cache)
        
        softmax_layer = parameters_values_list[-1]
        
        A, cache = self.linear_activation_forward(
                A, softmax_layer["w"], softmax_layer["b"], "softmax")
        caches.append(cache)
        
        return A, caches

        
        
        