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
from tensorflow.keras.regularizers import l2, l1
from keras import backend as K

FILES_PATH = 'lfw2/lfw2/'
TRAIN_PATH = 'Train.txt'
TEST_PATH = 'Test.txt'

IMAGE_SIZE = (250, 250)
TRAIN_SIZE = 2200
BATCH_SIZE = 32


def create_gen(data, datagen, train=True):
    if train:
        class_mode = 'binary'
        y_col = 'label'
        batch_size = BATCH_SIZE
    else:
        batch_size = 1
        class_mode = 'binary'
        y_col = 'label'

    gen1 = datagen.flow_from_dataframe(dataframe=data, target_size=IMAGE_SIZE, x_col='image1',
                                       y_col=y_col, color_mode="rgb",
                                       class_mode=class_mode, subset='training', shuffle=False,
                                       batch_size=batch_size, seed=42)

    gen2 = datagen.flow_from_dataframe(dataframe=data, target_size=IMAGE_SIZE, x_col='image2',
                                       y_col=y_col, color_mode="rgb",
                                       class_mode=class_mode, subset='training', shuffle=False,
                                       batch_size=batch_size, seed=42)

    while True:
        image_1 = gen1.next()
        image_2 = gen2.next()

        yield [image_1[0], image_2[0]], image_1[1]


def create_path(tup):
    name, num = tup
    padding = 4
    num = f'%0{padding}d' % int(num)
    file_name = name + '_' + num + '.jpg'
    path = os.path.join(FILES_PATH, name, file_name)
    return path


def create_path(tup):
    name, num = tup
    padding = 4
    num = f'%0{padding}d' % int(num)
    file_name = name + '_' + num + '.jpg'
    path = os.path.join(FILES_PATH, name, file_name)
    return path


def read_data(path, n=1100):
    count = 0
    data = pd.DataFrame(columns=['image1', 'image2', 'label', 'name1', 'name2'])
    with open(path, 'r') as f:
        for line in f:
            line = tuple(line.split('\t'))
            if count < n:
                name, image1, image2 = line
                data.loc[count] = [create_path((name, image1)), create_path((name, image2)), '1', name, name]
            else:
                name1, image1, name2, image2 = line
                data.loc[count] = [create_path((name1, image1)), create_path((name2, image2)), '0', name1, name2]
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
    model.add(Conv2D(128, (7, 7), activation='relu', kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (4, 4), activation='relu', kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, (4, 4), activation='relu', kernel_regularizer=l2(2e-4)))
    model.add(Flatten())
    model.add(Dense(4096, activation='sigmoid'))

    encoded1 = model(input1)
    encoded2 = model(input2)

#     distance_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
#     distance = distance_layer([encoded1,encoded2])
    distance_layer = Lambda(calculate_distance)
    distance = distance_layer([encoded1, encoded2])

    output_layer = Dense(1, activation='sigmoid')(distance)
    inputs = [input1, input2]
    network = Model(inputs=inputs, outputs=output_layer)

    return network


def n_way_one_shot(n, data, model, datagen):
    correct_pairs = data[data['label'] == '1'].reset_index(drop=True)
    predictions = list()
    non_pairs = data[data['label'] == '0']
    images = list(non_pairs['image1'].append(non_pairs['image2']))
    names = list(non_pairs['name1'].append(non_pairs['name2']))
    non_pairs = pd.DataFrame(list(zip(images, names)), columns=['image', 'name'])

    for i in range(len(correct_pairs)):
        n_way_samples = correct_pairs.iloc[i:i + 1]
        image_to_compare = n_way_samples['image1'][i]
        name_true = n_way_samples['name1'][i]
        n_way_samples = n_way_samples.iloc[:, :-2]
        # select images that are not the same
        sample_false = select_pairs_to_compare(n, non_pairs, name_true)

        for sample in sample_false:
            n_way_samples = n_way_samples.append({'image1': image_to_compare,
                                                  'image2': sample,
                                                  'label': '0'}, ignore_index=True)
        n_way_gen = create_gen(n_way_samples, datagen, False)
        predict = test_n_way(model, n_way_gen, n)
        predictions.append(predict)

    accuracy = sum(predictions) / len(predictions)
    return accuracy


def test_n_way(model, n_way_gen, n):
    # TODO: check if should be predict proba
    probabilities = model.predict(n_way_gen, steps=n)
    # In our generator the correct pair is in the first row every time,
    # so we would expect it to receive the max probability
    if np.argmax(probabilities) == 0:
        return 1
    else:
        return 0


def select_pairs_to_compare(n, images, name_to_compare):
    sample_false = images[images['name'] != name_to_compare]['image'].sample(n=n - 1)
    return sample_false

# def read_images():
#     max_files = 0
#     folders_with_1 = 0
#     print(f"files in dir:{len(glob.glob(os.path.join(FILES_PATH, '*')))}")
#     for name in glob.glob(os.path.join(FILES_PATH, '*')):
#         # path = os.path.join(FILES_PATH, name)
#         if len(os.listdir(name))>max_files:
#             max_files = len(os.listdir(name))
#         if len(os.listdir(name))==1:
#             folders_with_1+=1
#         for filename in os.listdir(name):
#             file_path = os.path.join(name, filename)
#             image = cv2.imread(file_path)
#     print(f"number of folders with one photo: {folders_with_1}")
#     print(f"maximum photos in file : {max_files}")




# Data Exploration
# read_images()
datagen = generator(rescale=1. / 255)
train_data = read_data(TRAIN_PATH)
test_data = read_data(TEST_PATH, 500)

train_gen = create_gen(train_data, datagen)
test_gen = create_gen(test_data, datagen)

siamese_network = create_model()
siamese_network.summary()
siamese_network.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=[tf.keras.metrics.Accuracy()])

# Training the model:
STEP_SIZE_TRAIN=TRAIN_SIZE//BATCH_SIZE

siamese_network.fit(train_gen,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    epochs=30,shuffle=False)


# siamese_network.save_weights('.my_checkpoint')


acc = n_way_one_shot(3, train_data, siamese_network, datagen)

test_acc = n_way_one_shot(3, test_data, siamese_network, datagen)

print(f"train accuracy: {acc}")
print(f"test accuracy: {acc}")

#
