## Overview
This project is different in nature compared to the previous ones in that this is a regression problem than a classification problem. The key files for this project are 

* model.py (which contains the model, training logic and the evaluation)
* drive.py
* model.json(which is generated upon running model.py)
* model.h5(which contains the weights and is generated upon running model.py) 


## Model Architecture

Below is the summary of the model I implemented to train the data.

        ____________________________________________________________________________________________________
		Layer (type)                     Output Shape          Param #     Connected to                     
		====================================================================================================
		convolution2d_1 (Convolution2D)  (None, 78, 158, 24)   1824        convolution2d_input_1[0][0]      
		____________________________________________________________________________________________________
		convolution2d_2 (Convolution2D)  (None, 37, 77, 36)    21636       convolution2d_1[0][0]            
		____________________________________________________________________________________________________
		convolution2d_3 (Convolution2D)  (None, 17, 37, 48)    43248       convolution2d_2[0][0]            
		____________________________________________________________________________________________________
		convolution2d_4 (Convolution2D)  (None, 15, 35, 64)    27712       convolution2d_3[0][0]            
		____________________________________________________________________________________________________
		convolution2d_5 (Convolution2D)  (None, 13, 33, 64)    36928       convolution2d_4[0][0]            
		____________________________________________________________________________________________________
		flatten_1 (Flatten)              (None, 27456)         0           convolution2d_5[0][0]            
		____________________________________________________________________________________________________
		dropout_1 (Dropout)              (None, 27456)         0           flatten_1[0][0]                  
		____________________________________________________________________________________________________
		activation_1 (Activation)        (None, 27456)         0           dropout_1[0][0]                  
		____________________________________________________________________________________________________
		dense_1 (Dense)                  (None, 1164)          31959948    activation_1[0][0]               
		____________________________________________________________________________________________________
		dropout_2 (Dropout)              (None, 1164)          0           dense_1[0][0]                    
		____________________________________________________________________________________________________
		activation_2 (Activation)        (None, 1164)          0           dropout_2[0][0]                  
		____________________________________________________________________________________________________
		dense_2 (Dense)                  (None, 100)           116500      activation_2[0][0]               
		____________________________________________________________________________________________________
		dense_3 (Dense)                  (None, 50)            5050        dense_2[0][0]                    
		____________________________________________________________________________________________________
		dense_4 (Dense)                  (None, 10)            510         dense_3[0][0]                    
		____________________________________________________________________________________________________
		dense_5 (Dense)                  (None, 1)             11          dense_4[0][0]                    
		====================================================================================================
		Total params: 32213367
		____________________________________________________________________________________________________

        

---

# Conclusion
The performance and accuracy of this model can be improved if we take into account the images from the left and right camera as well. A powerful GPU machine to train the model against a larger dataset would help. You could also collect training data from the other track as well to add variety to the input feature set. Right now, the model is built based on the reference network architecture that nvidia has published. However, if we can use VGG or GoogLeNet to transfer learning, we could improve performance and get more accurate results.
