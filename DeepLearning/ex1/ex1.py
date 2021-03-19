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
        self.initialize_parameters(layers_dims)
    

    def fit(self,data): 
        return        
    
    #### Train and predict
    def L_layer_model(self,X, Y, layers_dims, learning_rate, num_iterations, batch_size): 
        return
    
    def Predict(self,X, Y, parameters): 
        return

    #### Forward
    def initialize_parameters(self,layer_dims): 
        return None 
    
    def linear_forward(self,A, W, b):
        return None 
    
    
    def softmax(self,Z): 
        return None 
    
    def relu(self,Z): 
        return None
    
    def linear_activation_forward(self,A_prev, W, B, activation): 
        return
    
    def L_model_forward(self,X, parameters, use_batchnorm): 
        return
    
    def compute_cost(self,AL, Y): 
        return
    
    def apply_batchnorm(self,A): 
        return
    
    #### Backward: 
    def Linear_backward(self,dZ, cache): 
        #TODO: make sure shapes align
    
        dW = np.dot(dZ, np.transpose(cache["A_prev"]))
        db = dZ
        dA_prev = np.dot(dZ, np.transpose(cache['W']))  
        
        return dA_prev, dW, db
    
    def linear_activation_backward(self,dA, cache, activation): 
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
    
    def relu_backward (self,dA, activation_cache): 
        Z = activation_cache['Z']
        A, Z = self.relu(Z)        
        dZ = dA * np.int64(A>0)  

        return dZ
    
    
    def softmax_backward (self,dA, activation_cache): 
        dZ = dA 
        return dZ
    
    def L_model_backward(self,AL, Y, caches): 
        '''
        chaches: will be a list of dictionaries - each dictionart will have the activation cache 
        and the linear cache of each layer
        
        '''
        grads = {}
        layers = len(caches)
        
        dA = AL-Y
        grads["dA"+str(layers)]
 
        ## The last layer: 
        grads["dA"+str(layers-1)], grads["dW"+str(layers)],grads["db"+str(layers)] = \
            self.linear_activation_backward(dA, caches[layers-1], "softmax")
        
        # Rest of the layers: 
        for layer in range(layers-1, 0, -1): 
            grads["dA"+str(layer-1)], grads["dW"+str(layer)],grads["db"+str(layer)] = \
    self.linear_activation_backward(grads["dA"+str(layer)], caches[layer-1], "relu")
        
        return grads
    
    def Update_parameters(self,parameters, grads, learning_rate): 
        '''{
            layer_n: {
                w: <matrix of weights from layer n-1 to n>
                row - current layer number of neuron, columns - previous
                b: <vector of bias from layer n-1 to n>
        }'''
        layers = len(parameters)
        
        for layer in range(1, layers + 1): 
            # update w
            parameters["layer_"+str(layer)]['w'] = parameters["layer_"+str(layer)]['w']- learning_rate*grads['dW'+str(layer)]
            #update b
            parameters["layer_"+str(layer)]['b'] = parameters["layer_"+str(layer)]['b']- learning_rate*grads['db'+str(layer)]
        
        return parameters
    
    
