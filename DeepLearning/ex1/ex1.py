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
        return
    
    def linear_activation_backward(self,dA, cache, activation): 
        return
    
    def relu_backward (self,dA, activation_cache): 
        return
    
    
    def softmax_backward (self,dA, activation_cache): 
        return
    
    def L_model_backward(self,AL, Y, caches): 
        return
    
    def Update_parameters(self,parameters, grads, learning_rate): 
        return
    
    
