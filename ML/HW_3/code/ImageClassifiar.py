import itertools
import os
import shutil

import matplotlib.pylab as plt
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub

import PIL.Image as Image
import scipy.io

from tqdm import tqdm

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

import pathlib

##utils
print("TF version:", tf.__version__)
print("Hub version:", hub.__version__)
print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")
AUTOTUNE = tf.data.experimental.AUTOTUNE


#pathGlobalVarbial

data_dir = '../data/images'
data_dir = pathlib.Path(data_dir)


#get labels
label = scipy.io.loadmat('../data/imagelabels.mat')["labels"].T
#plt.hist(label,bins = 102)
CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"])

#chooose model

#MODULE_HANDLE ="https://tfhub.dev/google/imagenet/{}/feature_vector/4".format(handle_base)


module_selection = ("nasnet_large", 331) 
handle_base, pixels = module_selection
MODULE_HANDLE ="https://tfhub.dev/google/imagenet/nasnet_large/classification/4".format(handle_base)
IMAGE_SIZE = (pixels, pixels)
print("Using {} with input size {}".format(MODULE_HANDLE, IMAGE_SIZE))

'''
create dir for all images with the class names

# allImages = []
# mypath = "../data/images"
# for f in tqdm(os.listdir(mypath)):
#     pathToFile = os.path.join(mypath, f)
#     if os.path.isfile(pathToFile):
#         allImages.append(pathToFile)

# print("transferFiles:")
# for imagePath,label in tqdm(zip(allImages,label)):
#     directory  = f"{data_dir}/{label[0]}"
#     if not os.path.isdir(directory):
#         os.makedirs(directory) 
#     shutil.move(imagePath, directory)
'''



def get_label(file_path):
  # convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory
  return parts[-2] == CLASS_NAMES

def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  # resize the image to the desired size.
  return tf.image.resize(img, IMAGE_SIZE)

def process_path(file_path):
  label = get_label(file_path)
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label

def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
  # This is a small dataset, only load it once, and keep it in memory.
  # use `.cache(filename)` to cache preprocessing work for datasets that don't
  # fit in memory.
  if cache:
    if isinstance(cache, str):
      ds = ds.cache(cache)
    else:
      ds = ds.cache()

  ds = ds.shuffle(buffer_size=shuffle_buffer_size)

  # Repeat forever
  ds = ds.repeat()

  ds = ds.batch(BATCH_SIZE)

  # `prefetch` lets the dataset fetch batches in the background while the model
  # is training.
  ds = ds.prefetch(buffer_size=AUTOTUNE)

  return ds




#getData
list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'))

for f in list_ds.take(5):
  print(f.numpy())

# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)


DATASET_SIZE  = 0
print("count dataset size")
for _ in tqdm(labeled_ds.enumerate()):
    DATASET_SIZE+=1

BATCH_SIZE = 32 
labeled_ds.batch(BATCH_SIZE)
train_size = int(0.7 * DATASET_SIZE)
val_size = int(0.15 * DATASET_SIZE)
test_size = int(0.15 * DATASET_SIZE)

train_dataset = labeled_ds.take(train_size)
train_dataset.batch(BATCH_SIZE)
test_dataset = labeled_ds.skip(train_size)
val_dataset = labeled_ds.skip(test_size)
val_dataset.batch(BATCH_SIZE)
test_dataset = labeled_ds.take(test_size)




train_dataset = prepare_for_training(train_dataset)
val_dataset = prepare_for_training(val_dataset)

test_dataset = prepare_for_training(test_dataset)

#model

do_fine_tuning = False 



print("Building model with", MODULE_HANDLE)
model = tf.keras.Sequential([
    # Explicitly define the input shape so the model can be properly
    # loaded by the TFLiteConverter
    tf.keras.layers.InputLayer(input_shape=IMAGE_SIZE + (3,)),
    hub.KerasLayer(MODULE_HANDLE, trainable=do_fine_tuning),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(102,
                          kernel_regularizer=tf.keras.regularizers.l2(0.0001))
])
model.build((None,)+IMAGE_SIZE+(3,))
model.summary()

model.compile(
  optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9), 
  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
  metrics=['accuracy'])


steps_per_epoch = train_size // BATCH_SIZE
validation_steps = val_size // BATCH_SIZE
hist = model.fit(
    train_dataset,
    epochs=5, steps_per_epoch=steps_per_epoch,
    validation_data=val_dataset,
    validation_steps=validation_steps).history


predictions = model.evaluate(test_dataset,use_multiprocessing = True,steps = steps_per_epoch)


plt.figure()
plt.ylabel("Loss (training and validation)")
plt.xlabel("Training Steps")
plt.ylim([0,2])
plt.plot(hist["loss"], label ="train" )
plt.plot(hist["val_loss"], label ="val")
plt.legend()

plt.figure()
plt.ylabel("Accuracy (training and validation)")
plt.xlabel("Training Steps")
plt.ylim([0,1])
plt.plot(hist["accuracy"], label ="train")
plt.plot(hist["val_accuracy"], label ="val")
plt.legend()
























# data_dir = tf.keras.utils.get_file(
#     'flower_photos',
#     'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
#     untar=True)

# datagen_kwargs = dict(rescale=1./255, validation_split=.20)
# dataflow_kwargs = dict(target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
#                    interpolation="bilinear")

# valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
#     **datagen_kwargs)
# valid_generator = valid_datagen.flow_from_directory(
#     data_dir, subset="validation", shuffle=False, **dataflow_kwargs)

# do_data_augmentation = False 
# if do_data_augmentation:
#   train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
#       rotation_range=40,
#       horizontal_flip=True,
#       width_shift_range=0.2, height_shift_range=0.2,
#       shear_range=0.2, zoom_range=0.2,
#       **datagen_kwargs)
# else:
#   train_datagen = valid_datagen
# train_generator = train_datagen.flow_from_directory(
#     data_dir, subset="training", shuffle=True, **dataflow_kwargs)







