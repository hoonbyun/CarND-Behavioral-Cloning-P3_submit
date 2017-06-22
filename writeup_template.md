

#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./sample/center.jpg "Recovery Image"
[image4]: ./sample/recover_01.jpg "Recovery Image"
[image5]: ./sample/recover_02.jpg "Recovery Image"
[image6]: ./sample/recover_03.jpg "Recovery Image"
[image7]: ./sample/image_flip.jpg "Flip Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* video.mp4 recorded autonomous mode driving


####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed
My model is based on the NVDIA neural network which works very well for the autonomous driving learning
- Cropping image to (65, 320, 3) from (160, 320, 3) to leave the view onto the road
- Lamde layer to normalize each image
- Set of convolution layers with 5x5 kernel size with stream of filter size 24 -> 36 -> 48 -> 64 with relu activation
- Flatten all output to be prepared for the fully connected layer
- Fully connected layers from 100 input to 50 and finally one single output

####2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting and I added two different dropout layers with 50% threshold.
I picked up 80% sampels for the training data and rest of them are used for the validation set.
I also tried to add more training samples as much as i can to avoid overfitting issue with few training data.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.
mse is used for the loss function to optimize weights and biases since the output is a single number.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road center.
Recovery run was also added and left and right cameras images were used as well.
The inversion images were also generated and added.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to mimic the NVDIA network model which is proven solution for the autonomous driving learning.
The model starts with several convolution layers to build up more filters as it is passed by layers, and each convolution layer is followed by the relu activation then the network goes with the fully connected layers until the output is finallized with a single number.
Adam optimizer and mse loss optimizer are used to optimize the models since the output is a single number not like a classifier, so there is no learning rate parameter, the number of epochs is 5 which seems enough to converge into the relatively small loss factor.

####2. Final Model Architecture
Following is the code of my final model architecture, it starts with the cropping the images to cut the view only for the road then Lamda method used to nomalize images all together.
Several convolution layers used by increasing the number of the filters with the fixed patch size 5x5, and I placed two dropout layers to avoid overfiltting.
The last layers consist of the fully connected models to finalize the single output value.

```
    model.add(Cropping2D(cropping=((75,20), (0,0)), input_shape=preProcess_img_shape))
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape= postProcess_img_shape))
    model.add(Convolution2D(24,5,5, subsample=(2,2), input_shape= postProcess_img_shape, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Convolution2D(36,5,5 , subsample=(2,2), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(64,3,3, activation="relu"))
    model.add(Convolution2D(64,3,3, activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(1))
```


####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded a lap on track along the center lane driving. Here is an example image of center lane driving:

![alt text][image3]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover and drive back to the center of the road from the side, it is very essential traning data to get the model to learn how to drive and make sure being at the center of the road at any situdation.

![alt text][image4]  
 
![alt text][image6]

Then I added flipped images so that the model learns or used to the right turn since the track consists of mostly left turns.

![alt text][image7]

For the augmentation, the images from the left and right cameras were uesd, but the simulator only gives the steer angle for the center of the vehicle so I need to compensate or correct the angles for the left and right cameras's images, for example the angler gets smaller for the left camera when the vehicle turns to the left while the anger in point of the right camera looks larger, the corresponding code is below.  
`
left steer = steer_cent*(1.0 + np.sign(steer_cent)*STEER_SCALE_FACTOR_SIDE_IMAGE)  
`
`
right steer = steer_cent*(1.0 - np.sign(steer_cent)*STEER_SCALE_FACTOR_SIDE_IMAGE)
`


After the collection process, I had 38428 number of data points. I then preprocessed this data by normalization and cropping the image have the input image with the road view only.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the loss getting converged and stable after the interation I used an adam optimizer so that manually training the learning rate wasn't necessary.
