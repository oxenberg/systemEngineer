#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 17:31:28 2021

@author: saharbaribi
"""

import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator as generator
# import keras
import os
import sys

FILES_PATH = 'lfw2/lfw2/'
TRAIN_PATH = 'Train.txt'
TEST_PATH = 'Test.txt'

IMAGE_SIZE = (250, 250, 3)


def create_gen(data):
    gen1 = generator.flow_from_dataframe(data, target_size=IMAGE_SIZE, x_col='image1',
                                                    y_col='label',
                                                    class_mode='sparse', subset='training',shuffle=False)


    gen2 = generator.flow_from_dataframe(data, target_size=IMAGE_SIZE, x_col='image2',
                                        y_col='label',
                                        class_mode='sparse', subset='training', shuffle=False)
    return gen1, gen2


def create_path(tup):
    name, num = tup
    padding = 4
    num = f'%0{padding}d' % int(num)
    file_name = name + '_' + num + '.jpg'
    path = os.path.join(FILES_PATH, name, file_name)
    return path


def read_data(path, n=1100):
    count = 0
    data = pd.DataFrame(columns=['image1', 'image2', 'label'])
    with open(path, 'r') as f:
        for line in f:
            line = tuple(line.split('\t'))
            if count < n:
                name, image1, image2 = line
                data.loc[count] = [create_path((name, image1)), create_path((name, image2)), 1]
            else:
                name1, image1, name2, image2 = line
                data.loc[count] = [create_path((name1, image1)), create_path((name2, image2)), 0]
            count += 1

    return data


# Data Exploration
train_data = read_data(TRAIN_PATH)
test_data = read_data(TEST_PATH, 500)

train_im1, train_im2 = create_gen(train_data)
test_im1, test_im2 = create_gen(test_data)

