#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 17:31:28 2021

@author: saharbaribi
"""

import numpy as np
import pandas as pd 
import tensorflow as tf
# import keras
import os
import sys 


FILES_PATH = 'lfw2/lfw2/'
TRAIN_PATH = 'Train.txt'
TEST_PATH = 'Test.txt'

train_data = pd.read_csv(TRAIN_PATH, sep='\t', names = ['name', 'image1', 'image2'],)
test_data = pd.read_csv(TEST_PATH, sep='\t', names = ['name', 'image1', 'image2']) 

