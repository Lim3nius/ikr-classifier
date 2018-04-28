#!/usr/bin/env python3

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, SeparableConv2D
from keras import backend as K
import numpy as np
from keras.preprocessing.image import ImageDataGenerator


#image dimensions
img_height, img_width=80, 80


# Possible to add some transformations for bigger variability - smaller chance of over training and specializations
train_data_generator = ImageDataGenerator(rotation_range=3,
                                          horizontal_flip=True,
) # Some rotation & flipping
test_data_generator = ImageDataGenerator()

eval_data_generator = ImageDataGenerator()

train_gen = train_data_generator.flow_from_directory('../data/train',
                                                     target_size=(img_width,img_height),
                                                     batch_size=8,
                                                     # save_to_dir='previews/',
                                                     # save_format='png'
)
test_gen = test_data_generator.flow_from_directory('../data/validation',
                                                   target_size=(img_width,img_height),
                                                   batch_size=8,
)

eval_gen = eval_data_generator.flow_from_directory('../data/eval',
                                                   target_size=(img_width,img_height),
                                                   batch_size=8,
)


# Set correct input shape
if K.image_data_format() == 'channels_first':
    input_shape=(3,img_width,img_height)
else:
    input_shape=(img_width,img_height,3)

epochs = 10
batch_size = 8
number_of_classes = 2

model = Sequential()
model.add(SeparableConv2D(64, kernel_size=(7, 7), activation='relu', input_shape=input_shape))
# model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) # Make average

# change parameters
model.add(SeparableConv2D(64, kernel_size=(5,5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# model.add(Dropout(0.1))

model.add(SeparableConv2D(64, kernel_size=(3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.3))

model.add(Flatten()) # convert 3D to 1D features
model.add(Dense(4096, activation='relu'))
model.add(Dense(1038, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.17)) # overtraining protection
# model.summary()
model.add(Dense(number_of_classes, activation='softmax')) # The last layer - determining who is who

# model.compile(loss=keras.losses.categorical_crossentropy,
#               # optimizer=keras.optimizers.Adadelta(),
#               optimizer=keras.optimizers.SGD,
#               metrics=['accuracy'])

model.compile(loss='categorical_crossentropy', # categorical because of 2 classes
              optimizer='sgd',
              metrics=['accuracy'])


model.fit_generator(train_gen,
                    steps_per_epoch=train_gen.samples // batch_size,
                    epochs=epochs,
                    validation_data=test_gen,
                    validation_steps=train_gen.samples // batch_size,
                    verbose=0
)

score = model.evaluate_generator(test_gen)


# model.predict_generator(test_gen)
results = list(zip(test_gen.filenames, model.predict_generator(test_gen)))
results = [(x,list(y)) for (x,y) in results]


trains = list(zip(train_gen.filenames, model.predict_generator(train_gen)))
trains = [(x,list(y)) for (x,y) in trains]

# print(results)
evals = list(zip(eval_gen.filenames, model.predict_generator(eval_gen)))
evals = [(x,list(y)) for (x,y) in evals]

with open('results_cnn.txt', mode='w') as ff:
    for (x,y) in evals:
        print('{} {} {}'.format(x, max(y), 1 if y[0] > y[1] else 0), file=ff, end='\n')
