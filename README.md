## Overview
This project is different in nature compared to the previous ones in that this is a regression problem than a classification problem. The key files for this project are 

* model.py (which contains the model, training logic and the evaluation)
* drive.py
* model.json(which is generated upon running model.py)
* model.h5(which contains the weights and is generated upon running model.py) 

## Introduction
The expectation of this project is to simulate driving a car through a track, gather training data, build a neural network model, fit the training data to this model and finally allow the simulator to self-drive the car by passing in the images from the simulator's track to the final trained model.

## Data Collection
One of the challenges with this project is gathering the training data. My first attempt was to drive the car in the simulator using my keyboard. This way I naturally drove the car in the middle of the road and gathered only the happy path training data. When I trained my model with this data, the model did not know how to recover when a car went outside the track. One of the ways to teach the model to recover was to turn off recording, simulate the car going off the track, then turn the recording on and manuever the car back to center of the track. I found this process a bit tedious given that all these have to be done using my keyboard. That is why I resorted to training data augmentation. 

I followed the below techniques to augment the training data.

1. Instead of relying on only the center camera image, randomly pick left or right camera image and accordingly adjust the steering angle by +/- 0.25.
2. Change the brightness of the image
3. Add shadow to one random portion of the image
4. Shift the camera image horizontally or vertically 

Another important thing is that most of the sample training data contains data corresponding to driving straight, which means the most occurring steering angle is 0. It is natural that there will be a bias towards driving straight with the sample training data. Apart from augmenting data, we also randomly drop out some data with steering angle = 0 using a variable called pr_threshold.

## Pre-processing the image
The images are pre-processed as follows

1. Crop the top 1/5 portion of the image to remove the horizon and the bottom 25 pixels to remove the car's hood.
2. Resize the image to 64x64 square image.


## Model Architecture

Before starting with building the model, I did some research and found the following resources useful and started with the models discussed in these resources.
1. An augmentation based deep neural network approach to learn human driving behavior: https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.oh1e42kmq
2. End-to-End Deep Learning for Self Driving cars : https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
3. Learning human driving behavior using NVIDIA's neural network model and image augmentation: https://chatbotslife.com/learning-human-driving-behavior-using-nvidias-neural-network-model-and-image-augmentation-80399360efee#.6ptu4rq9t
4. Cloning a car to mimic human driving: https://medium.com/@mohankarthik/cloning-a-car-to-mimic-human-driving-5c2f7e8d8aff#.on392afor
5. VGG16 model from Keras: https://github.com/fchollet/keras/blob/master/keras/applications/vgg16.py

With the above references as baseline, I converged at the following model, which is a modified version of the VGG16 model architecture. Below is the summary of the model I implemented to train the data. I used the Lambda layer to normalize the intensities between -0.5 and 0.5. We did not use validation to choose the final model, but to verify if the training scheme was implemented correctly. The model generated after 8 epochs was chosen for further testing. This model drove for more than 2 laps on track 1. 

Also, I am using the fit_generator method of the keras model so that the input training set is fed in as batches thereby enabling it to run on my local machine.

		____________________________________________________________________________________________________
		Layer (type)                     Output Shape          Param #     Connected to                     
		====================================================================================================
		lambda_1 (Lambda)                (None, 64, 64, 3)     0           lambda_input_1[0][0]             
		____________________________________________________________________________________________________
		color_conv1 (Convolution2D)      (None, 64, 64, 3)     12          lambda_1[0][0]                   
		____________________________________________________________________________________________________
		elu_1 (ELU)                      (None, 64, 64, 3)     0           color_conv1[0][0]                
		____________________________________________________________________________________________________
		conv2 (Convolution2D)            (None, 64, 64, 32)    896         elu_1[0][0]                      
		____________________________________________________________________________________________________
		elu_2 (ELU)                      (None, 64, 64, 32)    0           conv2[0][0]                      
		____________________________________________________________________________________________________
		conv3 (Convolution2D)            (None, 64, 64, 32)    9248        elu_2[0][0]                      
		____________________________________________________________________________________________________
		elu_3 (ELU)                      (None, 64, 64, 32)    0           conv3[0][0]                      
		____________________________________________________________________________________________________
		maxpooling2d_1 (MaxPooling2D)    (None, 32, 32, 32)    0           elu_3[0][0]                      
		____________________________________________________________________________________________________
		dropout_1 (Dropout)              (None, 32, 32, 32)    0           maxpooling2d_1[0][0]             
		____________________________________________________________________________________________________
		conv4 (Convolution2D)            (None, 32, 32, 64)    18496       dropout_1[0][0]                  
		____________________________________________________________________________________________________
		elu_4 (ELU)                      (None, 32, 32, 64)    0           conv4[0][0]                      
		____________________________________________________________________________________________________
		conv5 (Convolution2D)            (None, 32, 32, 64)    36928       elu_4[0][0]                      
		____________________________________________________________________________________________________
		elu_5 (ELU)                      (None, 32, 32, 64)    0           conv5[0][0]                      
		____________________________________________________________________________________________________
		maxpooling2d_2 (MaxPooling2D)    (None, 16, 16, 64)    0           elu_5[0][0]                      
		____________________________________________________________________________________________________
		dropout_2 (Dropout)              (None, 16, 16, 64)    0           maxpooling2d_2[0][0]             
		____________________________________________________________________________________________________
		conv6 (Convolution2D)            (None, 16, 16, 128)   73856       dropout_2[0][0]                  
		____________________________________________________________________________________________________
		elu_6 (ELU)                      (None, 16, 16, 128)   0           conv6[0][0]                      
		____________________________________________________________________________________________________
		conv7 (Convolution2D)            (None, 16, 16, 128)   147584      elu_6[0][0]                      
		____________________________________________________________________________________________________
		elu_7 (ELU)                      (None, 16, 16, 128)   0           conv7[0][0]                      
		____________________________________________________________________________________________________
		maxpooling2d_3 (MaxPooling2D)    (None, 8, 8, 128)     0           elu_7[0][0]                      
		____________________________________________________________________________________________________
		dropout_3 (Dropout)              (None, 8, 8, 128)     0           maxpooling2d_3[0][0]             
		____________________________________________________________________________________________________
		flatten_1 (Flatten)              (None, 8192)          0           dropout_3[0][0]                  
		____________________________________________________________________________________________________
		dense_1 (Dense)                  (None, 512)           4194816     flatten_1[0][0]                  
		____________________________________________________________________________________________________
		elu_8 (ELU)                      (None, 512)           0           dense_1[0][0]                    
		____________________________________________________________________________________________________
		dense_2 (Dense)                  (None, 64)            32832       elu_8[0][0]                      
		____________________________________________________________________________________________________
		elu_9 (ELU)                      (None, 64)            0           dense_2[0][0]                    
		____________________________________________________________________________________________________
		dense_3 (Dense)                  (None, 16)            1040        elu_9[0][0]                      
		____________________________________________________________________________________________________
		dense_4 (Dense)                  (None, 1)             17          dense_3[0][0]                    
		====================================================================================================
		Total params: 4515725
		____________________________________________________________________________________________________
        

---

# Conclusion
This model took sometime to converge on the best weights. If I had used a GPU based machine, it might have saved sometime.But I got some decently trained model which is able to recover itself when the car moves to the edge of the track. 
