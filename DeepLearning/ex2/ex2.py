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
import keras
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, Dense, MaxPooling2D, Flatten, Lambda
import os
import sys
from keras.optimizers import Adam



FILES_PATH = 'lfw2/lfw2/'
TRAIN_PATH = 'Train.txt'
TEST_PATH = 'Test.txt'

IMAGE_SIZE = (250, 250, 3)


def create_gen(data, train = True):
    if train:
        class_mode = 'binary'
        y_col = 'label'
    else:
        y_col = None
        class_mode = None
    datagen = generator(rescale=1. / 255)
    gen1 = datagen.flow_from_dataframe(dataframe = data, target_size=IMAGE_SIZE, x_col='image1',
                                                    y_col=y_col,
                                                    class_mode=class_mode, subset='training',shuffle=False)


    gen2 = datagen.flow_from_dataframe(dataframe = data, target_size=IMAGE_SIZE, x_col='image2',
                                        y_col=y_col,
                                        class_mode=class_mode, subset='training', shuffle=False)
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
                data.loc[count] = [create_path((name, image1)), create_path((name, image2)), '1']
            else:
                name1, image1, name2, image2 = line
                data.loc[count] = [create_path((name1, image1)), create_path((name2, image2)), '0']
            count += 1

    return data


def calculate_distance(tensors):
    return abs(tensors[0] - tensors[1])


## Creating the model:
def create_model():
    input1 = Input(IMAGE_SIZE)
    input2 = Input(IMAGE_SIZE)

    model = Sequential()
    model.add(Conv2D(64, (10, 10), activation='relu', input_shape=IMAGE_SIZE))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (7, 7), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (4, 4), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, (4, 4), activation='relu'))
    model.add(Flatten())
    model.add(Dense(4096, activation='sigmoid'))

    encoded1 = model(input1)
    encoded2 = model(input2)

    distance_layer = Lambda(calculate_distance)
    distance = distance_layer([encoded1, encoded2])

    output_layer = Dense(1, activation='sigmoid')(distance)
    inputs = [input1, input2]
    network = Model(inputs=inputs, outputs=output_layer)

    return network

# # Data Exploration
train_data = read_data(TRAIN_PATH)
test_data = read_data(TEST_PATH, 500)

train_im1, train_im2 = create_gen(train_data)
test_im1, test_im2 = create_gen(test_data)

siamese_network = create_model()
siamese_network.summary()
siamese_network.compile(optimizer=Adam(learning_rate=0.0001), loss ='binary_crossentropy')

# def n_way_one_shot(n, data):
#     sample_false = data[data['label'==0]]['image2'].sample(n=n-1)
#     sample_true = data[data['label'==0]].sample(n=1)
#     image_to_compare = sample_true['image1']
#     samples = sample_false.append(sample_true['image2'])
#     for image in samples:
#         pair = (image_to_compare, image)
#     return None


# Training the model:
siamese_network.fit_generator([train_im1, train_im2], epochs = 10)
siamese_network.evaluate_generator([test_im1, test_im2])











