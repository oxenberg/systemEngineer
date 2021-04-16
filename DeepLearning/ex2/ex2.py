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

IMAGE_SIZE = (250, 250)
TRAIN_SIZE = 2200
BATCH_SIZE = 32


def create_gen(data, train=True):
    if train:
        class_mode = 'binary'
        y_col = 'label'
    else:
        y_col = None
        class_mode = None
    datagen = generator(rescale=1. / 255)


    gen1 = datagen.flow_from_dataframe(dataframe=data, target_size=IMAGE_SIZE, x_col='image1',
                                       y_col=y_col, color_mode="rgb",
                                       class_mode=class_mode, subset='training', shuffle=False,
                                       batch_size=BATCH_SIZE,seed=42)

    gen2 = datagen.flow_from_dataframe(dataframe=data, target_size=IMAGE_SIZE, x_col='image2',
                                       y_col=y_col, color_mode="rgb",
                                       class_mode=class_mode, subset='training', shuffle=False,
                                       batch_size=BATCH_SIZE,seed=42)

    while True:
        image_1 = gen1.next()
        image_2 = gen2.next()

        yield [image_1[0], image_2[0]], image_1[1]
    return [datagen1, datagen2]


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
    # shuffle the data
    data = data.sample(frac=1).reset_index(drop=True)
    return data


def calculate_distance(tensors):
    return abs(tensors[0] - tensors[1])


## Creating the model:
def create_model():
    input1 = Input(IMAGE_SIZE +(3,))
    input2 = Input(IMAGE_SIZE+(3,))

    model = Sequential()
    model.add(Conv2D(64, (10, 10), activation='relu', input_shape=IMAGE_SIZE+(3,)))
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


# Data Exploration
train_data = read_data(TRAIN_PATH)
test_data = read_data(TEST_PATH, 500)

train_gen = create_gen(train_data)
test_gen = create_gen(test_data)

siamese_network = create_model()
siamese_network.summary()
siamese_network.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy')

# def n_way_one_shot(n, data):
#     sample_false = data[data['label'==0]]['image2'].sample(n=n-1)
#     sample_true = data[data['label'==0]].sample(n=1)
#     image_to_compare = sample_true['image1']
#     samples = sample_false.append(sample_true['image2'])
#     for image in samples:
#         pair = (image_to_compare, image)
#     return None


# Training the model:
STEP_SIZE_TRAIN=TRAIN_SIZE//BATCH_SIZE

siamese_network.fit_generator(generator=train_gen,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    epochs=5,shuffle=False)


siamese_network.save_weights('.my_checkpoint')
# siamese_network.fit_generator(train_gen, epochs=5)
# siamese_network.evaluate_generator(test_gen)

# def read_images():
#     for name in os.listdir(FILES_PATH):
#         path = os.path.join(FILES_PATH, name)
#         for filename in os.listdir(path):
#             file_path = os.path.join(path, filename)
#             image = cv2.imread(file_path)
#
# read_images()
