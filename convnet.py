#!/usr/bin/env python3

from ikrlib import *
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

def png2fea(dir_name):
    """
    Loads all *.png images from directory dir_name into a dictionary. Keys are the file names
    and values and 2D numpy arrays with corresponding grayscale images
    """
    features = {}
    for f in glob(dir_name + '/*.png'):
        print("Processing file: ", f)
        features[f] = imread(f, True).astype(np.float64).flatten()
    return features


#image dimensions
img_height, img_width=80, 80

# dir = ['data/target_train', 'data/non_target_train', 'data/target_dev', 'data/non_target_dev']

# Possible to add some transformations for bigger variability - smaller chance of over training and specializations
train_data_generator = ImageDataGenerator(horizontal_flip=True, rotation_range=20) # Some rotation & flipping
test_data_generator = ImageDataGenerator()

train_gen = train_data_generator.flow_from_directory('data/train', target_size=(img_width,img_height))
test_gen = test_data_generator.flow_from_directory('data/validation', target_size=(img_width,img_height))


# Set correct input shape
if K.image_data_format() == 'channels_first':
    input_shape=(3,img_width,img_height)
else:
    input_shape=(img_width,img_height,3)

epochs = 10
batch_size = 10
number_of_classes = 2

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
#model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25)) # overtraining protection
model.add(Dense(number_of_classes, activation='softmax')) # The last layer - determining who is who

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])


model.fit_generator(train_gen,
                    steps_per_epoch=128,
                    epochs=epochs,
                    validation_data=test_gen,
                    validation_steps=60)



score = model.evaluate_generator(test_gen)
model.summary()
