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


#### Forward

def initialize_parameters(layer_dims): 
    return None 

def linear_forward(A, W, b):
    return None 


def softmax(Z): 
    return None 

def relu(Z): 
    return None

def linear_activation_forward(A_prev, W, B, activation): 
    return

def L_model_forward(X, parameters, use_batchnorm): 
    return

def compute_cost(AL, Y): 
    return

def apply_batchnorm(A): 
    return

#### Backward: 

def Linear_backward(dZ, cache): 
    return

def linear_activation_backward(dA, cache, activation): 
    return

def relu_backward (dA, activation_cache): 
    return


def softmax_backward (dA, activation_cache): 
    return

def L_model_backward(AL, Y, caches): 
    return

def Update_parameters(parameters, grads, learning_rate): 
    return


#### Train and predict

def L_layer_model(X, Y, layers_dims, learning_rate, num_iterations, batch_size): 
    return

def Predict(X, Y, parameters): 
    return