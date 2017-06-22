import pickle
import tensorflow as tf
import numpy as np
# TODO: import Keras layers you need here
from keras.layers import Input, Flatten, Dense, Activation, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model, Sequential

import csv
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import gc
from keras import backend as K
#from keras.utils import plot_model


flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('sim_test_csv_file', '', "csv test file from sim .csv")

STEER_SCALE_FACTOR_SIDE_IMAGE = 0.1

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                img_path = batch_sample[0]
                steer = batch_sample[1]
                bIsFlipImg = batch_sample[2]
                img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
                if img != None:
                    #To flip a image and inverse the steer value
                    if bIsFlipImg:
                        img = np.fliplr(img)
                        steer = -1*steer
                        #print (steer)
                        #print (img.shape)
                        #cv2.imshow('img_path', img)
                        #cv2.waitKey(10000)
                    images.append(img)
                    angles.append(steer)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield X_train, y_train
    
def getSamples(sCsvFilePath, sampleList, bAugFlipImg = False):
    rows = []
    with open(sCsvFilePath) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            rows.append(row)
            
        curRelPath = os.path.dirname(sCsvFilePath)

    for row in rows[1:]:
        #get image path for each image fro each camera center, left, right
        path_cent_image = curRelPath+os.sep+'IMG'+os.sep+row[0].split('/')[-1]
        path_left_image = curRelPath+os.sep+'IMG'+os.sep+row[1].split('/')[-1]
        path_right_image = curRelPath+os.sep+'IMG'+os.sep+row[2].split('/')[-1]
        steer_cent = float(row[3])
        sampleList.append([path_cent_image, steer_cent, False])
        sampleList.append([path_left_image, steer_cent*(1.0 + np.sign(steer_cent)*STEER_SCALE_FACTOR_SIDE_IMAGE), False])
        sampleList.append([path_right_image, steer_cent*(1.0 - np.sign(steer_cent)*STEER_SCALE_FACTOR_SIDE_IMAGE), False])
        if bAugFlipImg:
            #apply angle correction to the left and right camera image
            sampleList.append([path_cent_image, steer_cent, True])
            sampleList.append([path_left_image, steer_cent*(1.0 + np.sign(steer_cent)*STEER_SCALE_FACTOR_SIDE_IMAGE), True])
            sampleList.append([path_right_image, steer_cent*(1.0 - np.sign(steer_cent)*STEER_SCALE_FACTOR_SIDE_IMAGE), True])



def main(_):

    #image samples, [image, steer]
    samples = []
    #get image path and steer for each image
    getSamples(FLAGS.sim_test_csv_file, samples, True)
    #suffle samples
    samples = shuffle(samples)
    #split data set into train and validation data
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    print (len(train_samples))
    print (len(validation_samples))
    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)
    
    preProcess_img_shape = (160,320,3)
    postProcess_img_shape = (65, 320, 3)
    model = Sequential()
    #Cropping 75 rows pixels from the top, 20 rows pixels from the bottom
    model.add(Cropping2D(cropping=((75,20), (0,0)), input_shape=preProcess_img_shape))
    
    # Preprocess incoming data, centered around zero with small standard deviation 
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape= postProcess_img_shape))

    # LeNet model
    # model.add(Convolution2D(32, 5, 5, subsample=(2,2), activation='relu', input_shape = postProcess_img_shape))
    # model.add(MaxPooling2D((2, 2)))
    # model.add(Convolution2D(16, 5, 5, subsample=(2,2), activation='relu'))
    # model.add(MaxPooling2D((2, 2)))
    # model.add(Flatten())
    # model.add(Dense(128))
    # model.add(Dropout(0.5))
    # model.add(Dense(84))
    # model.add(Dense(1))

    # NVDIA model
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

    model.compile(loss='mse', optimizer='adam')
    model.fit_generator(train_generator, samples_per_epoch= \
                len(train_samples), validation_data=validation_generator, \
                nb_val_samples=len(validation_samples), nb_epoch=5)


    model.save('model.h5')
    #plot_model(model, to_file='model.png')
    exit()
    gc.collect()
    K.clear_session()
# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
