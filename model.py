'''
Created on Jan 5, 2017

@author: praveen.subramanian
'''

import csv
import math
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import cv2
import time
import json
import pandas as pd
import random

from keras.models import Sequential
from keras.layers import Input, advanced_activations
from keras.layers.core import Dense, Dropout, Activation, Flatten, Lambda
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

from keras.models import model_from_json
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.layers.advanced_activations import ELU,LeakyReLU
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from keras.backend.tensorflow_backend import batch_normalization
from keras.callbacks import ModelCheckpoint

########################################################################################################
# Data Augmentation
# Reference: https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.oh1e42kmq
########################################################################################################
##############################################
# Helper method for changing image brightness
##############################################
def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    #print(random_bright)
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

###############################################################
# Helper method for shifting the image horizontally/vertically
###############################################################
def trans_image(image,steer,trans_range):
    rows = image.shape[0]
    cols = image.shape[1]
    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer + tr_x/trans_range*2*.2
    tr_y = 40*np.random.uniform()-40/2
    #tr_y = 0
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(cols,rows))
    
    return image_tr,steer_ang

######################################################
# Adding shadow mask to the images at one random side
######################################################
def add_random_shadow(image):
    top_y = 320*np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320*np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    #random_bright = .25+.7*np.random.uniform()
    if np.random.randint(2)==1:
        random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright    
    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
    return image


new_size_col,new_size_row = 64, 64
def preprocessImage(image):
    shape = image.shape
    # note: numpy arrays are (row, col)!
    image = image[math.floor(shape[0]/5):shape[0]-25, 0:shape[1]]
    image = cv2.resize(image,(new_size_col,new_size_row),interpolation=cv2.INTER_AREA)    
    #image = image/255.-.5
    return image

def preprocess_image_file_train(line_data):
    i_lrc = np.random.randint(3)
    if (i_lrc == 0):
        path_file = line_data['left'][0].strip()
        shift_ang = .25
    if (i_lrc == 1):
        path_file = line_data['center'][0].strip()
        shift_ang = 0.
    if (i_lrc == 2):
        path_file = line_data['right'][0].strip()
        shift_ang = -.25
    y_steer = line_data['steer_sm'][0] + shift_ang
    image = cv2.imread(path_file)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image,y_steer = trans_image(image,y_steer,100)
    image = augment_brightness_camera_images(image)
    image = preprocessImage(image)
    image = np.array(image)
    ind_flip = np.random.randint(2)
    if ind_flip==0:
        image = cv2.flip(image,1)
        y_steer = -y_steer
    
    return image,y_steer


pr_threshold = 1

def generate_train_from_PD_batch(data,batch_size = 32):
    
    batch_images = np.zeros((batch_size, new_size_row, new_size_col, 3))
    batch_steering = np.zeros(batch_size)
    while 1:
        for i_batch in range(batch_size):
            i_line = np.random.randint(len(data))
            line_data = data.iloc[[i_line]].reset_index()
            
            keep_pr = 0
            #x,y = preprocess_image_file_train(line_data)
            while keep_pr == 0:
                x,y = preprocess_image_file_train(line_data)
                pr_unif = np.random
                if abs(y)<.1:
                    pr_val = np.random.uniform()
                    if pr_val>pr_threshold:
                        keep_pr = 1
                else:
                    keep_pr = 1
            
            #x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])
            #y = np.array([[y]])
            batch_images[i_batch] = x
            batch_steering[i_batch] = y
        yield batch_images, batch_steering


    
####################################################################
# Reading the training data generated by test drive on the simulator
####################################################################
data_files_s = pd.read_csv('driving_log.csv')
print(data_files_s)

# with open('driving_log.csv') as data:
#     with open('validation_log.csv', 'w') as test:
#         with open('training_log.csv', 'w') as train:
#             header = next(data)
#             test.write(header)
#             train.write(header)
#             for line in data:
#                 if random.random() > 0.66:
#                     test.write(line)
#                 else:
#                     train.write(line)
# 
# data_files_s = pd.read_csv('training_log.csv')
# print(data_files_s)
# print("***********************")
# val_data_files_s = pd.read_csv('validation_log.csv')
# print(val_data_files_s)
###########################################################################
# Testing if the batchgen function defined above is working as expected
# before passing it to the keras.model.fit_generator().
###########################################################################     
i = 1  
batch = generate_train_from_PD_batch(data_files_s,batch_size=256)
for x,y in batch:
    print("$$$$$$$$$$$$$$$$$$")
    print("Dimension 1 of image 1 ",len(x))
    print("************")
    print("Dimension 2 of image 1 ",len(x[0]))
    print("************")
    print("Dimension 3 of image 1 ",len(x[0][0]))
    print("************")
    print("Steering angle ",y[0])
    print("************")
    i = i + 1
    if(i > 1):
        break

################################################################################################################################################################################
# This is a function to create a model with 5 convolutional layers, Flatten, Dropout, RELU, Fully Connected Layer, Dropout, RELU followed by 4 Fully Connected layers.
# As learnt from earlier projects, Droput is to avoid overfitting, RELU is to introduce non-linearity in the model. The last Fully Connected layer
# has only one neuron in this model since this is a regression problem given that the output steering angles as continuous values. This is in contrast to the previous projects
# where the output lables were discrete values and hence in the case of German traffic sign database, the number of neurons in the last Fully Connected layer equals the number
# of output classes (i.e. 42 for German Traffic Data Set problem).
#
# References:
# Nvidia paper: http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
# Confluence link: https://chatbotslife.com/learning-human-driving-behavior-using-nvidias-neural-network-model-and-image-augmentation-80399360efee#.6ptu4rq9t
#
################################################################################################################################################################################
input_shape = (new_size_row, new_size_col, 3)
def creat_model_nvidia():
    
    model = Sequential()

    model.add(Lambda(lambda x: x/127.5 - .5,
                     input_shape=input_shape,
                     output_shape=(new_size_row, new_size_col, 3)))
    model.add(Convolution2D(3, 1, 1, border_mode='same', name='color_conv1'))
    model.add(ELU())
    model.add(Convolution2D(32, 3, 3, border_mode='same', name='conv2'))
    model.add(ELU())
    model.add(Convolution2D(32, 3, 3, border_mode='same', name='conv3'))
    model.add(ELU())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.5))
    
    model.add(Convolution2D(64, 3, 3, border_mode='same', name='conv4'))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, border_mode='same', name='conv5'))
    model.add(ELU())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.5))
    
    model.add(Convolution2D(128, 3, 3, border_mode='same', name='conv6'))
    model.add(ELU())
    model.add(Convolution2D(128, 3, 3, border_mode='same', name='conv7'))
    model.add(ELU())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.5))
              
    model.add(Flatten())
    
    model.add(Dense(512))
    model.add(ELU())
    model.add(Dense(64))
    model.add(ELU())
    model.add(Dense(16))
    model.add(Dense(1))
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='mse')
    
    model.summary()
    
    return model

model = creat_model_nvidia()


val_size = 1
batch_size = 256
# checkpoint
# filepath="model.h5" #"weights.best.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
# callbacks_list = [checkpoint]
# train_r_generator = generate_train_from_PD_batch(data_files_s, batch_size)
# val_r_generator = generate_train_from_PD_batch(val_data_files_s, batch_size)
# model.fit_generator(train_r_generator, samples_per_epoch=20224, nb_epoch=8, verbose=1, callbacks=callbacks_list)

# model.fit_generator(train_r_generator, samples_per_epoch=20224, nb_epoch=8, verbose=1, callbacks=callbacks_list, validation_data=val_r_generator, nb_val_samples=256)

loss = 1
for i_pr in range(8):
      
    train_r_generator = generate_train_from_PD_batch(data_files_s, batch_size)
    nb_vals = np.round(len(data_files_s)/val_size)-1
      
    history = model.fit_generator(train_r_generator, samples_per_epoch=20224, nb_epoch=1, verbose=1)
#     current_loss = history.history["loss"][0]
#     print("current_loss = ",current_loss)
#     if current_loss < loss:
#         loss = current_loss
#         model.save_weights("model.h5")
#         print("Saving new weights")
    pr_threshold = 1/(i_pr+1)
     

#################################################
# Saving the trained model as a json file.
# Also, saving the learned weights as h5 file.
#################################################
model_json = model.to_json()
with open('model.json', 'w') as f:
    json.dump(model_json, f)
model.save_weights("model.h5")
