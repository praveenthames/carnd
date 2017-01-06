'''
Created on Jan 5, 2017

@author: praveen.subramanian
'''

import csv
#from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import cv2
import time
import json

from keras.models import Sequential
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation, Flatten, Lambda
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

from keras.models import model_from_json
from keras.layers.convolutional import Convolution2D
from keras.layers.advanced_activations import ELU

ifile  = open('/Users/praveen.subramanian/data/carnd/p3/driving_log.csv', "rt")
reader = csv.reader(ifile)

center_image = []
steering_angle = []

for row in reader:
    center_image.append(row[0])
    steering_angle.append(row[3])

ifile.close()

print("No. of center images = ",len(center_image))
print("No. of steering angles = ",len(steering_angle))

num_images = len(center_image)

grey = np.zeros((num_images,160,320,3))
#grey = np.zeros((num_images,51200))
for i in range(num_images):
    im1=cv2.imread(center_image[i])
    #grey[i] = im1
    steering_angle[i] = round(float(steering_angle[i]),2)
    #grey[i] = cv2.cvtColor( im1, cv2.COLOR_RGB2GRAY).flatten()
    
steering_classes = list(set(steering_angle))
for i in range(len(steering_angle)):
    steering_angle[i] = steering_classes.index(steering_angle[i])
    
num_classes = len(set(steering_classes))
    
print("The shape after grey scale conversion and flattening = ",grey.shape)
print("The number of distinct classes = ",num_classes)

X_train, X_val, y_train, y_val = train_test_split(grey,steering_angle , test_size=0.33, random_state=0)

print("Shape of X_train = ",X_train.shape)
print("Shape of y_train = ",len(y_train))
print("Shape of X_val = ",X_val.shape)
print("Shape of y_val = ",len(y_val))


batch_size = 256
nb_epoch = 5

#X_train = X_train.reshape(2676, 51200)
#X_val = X_val.reshape(1319, 51200)
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
#X_train /= 255
#X_val /= 255

# convert class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train, num_classes)
y_val = np_utils.to_categorical(y_val, num_classes)

image_rows = 160
image_columns = 320
image_channels = 3

input_shape = (image_rows, image_columns, image_channels)
model = Sequential()
model.add(Convolution2D(24,5,5, input_shape=input_shape, subsample = (2,2),
                    border_mode='same',
                    name='conv1', init='he_normal'))
model.add(ELU())

model.add(Convolution2D(36,5,5, subsample = (2,2),
                    border_mode='same',
                    name='conv2', init='he_normal'))
model.add(ELU())
model.add(Convolution2D(48,5,5, subsample = (2,2),
                    border_mode='valid',
                    name='conv3', init='he_normal'))
model.add(ELU())
model.add(Convolution2D(64,3,3, subsample = (1,1),
                    border_mode='valid',
                    name='conv4', init='he_normal'))
model.add(ELU())
model.add(Convolution2D(64,3,3, subsample = (1,1),
                    border_mode='valid',
                    name='conv5', init='he_normal'))
model.add(ELU())
model.add(Flatten())
model.add(Dense(100,name='hidden1', init='he_normal'))
model.add(ELU())
model.add(Dense(50,name='hidden2', init='he_normal'))
model.add(ELU())
model.add(Dense(10,name='hidden3',init='he_normal'))
model.add(ELU())
model.add(Dense(1, name='output', init='he_normal'))
adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.summary()
model.compile(optimizer=adam, loss='mse')


history = model.fit(X_train, y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_val, y_val))
score = model.evaluate(X_val, y_val, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

model_json = model.to_json()
with open('model.json', 'w') as f:
    json.dump(model_json, f)
model.save_weights("model.h5")
