import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from os import walk
import h5py
import numpy as np
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
import math
from adabelief_tf import AdaBeliefOptimizer
from os import walk
from skimage.transform import resize
import pickle
import csv
import pandas as pd

TRAIN_PATH = 'ISTANBUL/training/' #might need to change here
MODEL_SAVE_PATH = 'ISTANBUL.h5'
IMAGE_WIDTH = 436
IMAGE_HEIGHT = 495

NUMBER_OF_CHANNELS = 8
TIMESTEPS_X = 12
TIMESTEPS_Y = 6

BATCH_SIZE = 2

X_Seq = TIMESTEPS_X * NUMBER_OF_CHANNELS
Y_Seq = TIMESTEPS_Y * NUMBER_OF_CHANNELS


def get_filenames(path):
    for (dirpath, dirnames, filenames) in walk(path):
        return filenames
    
class generator:
    def __call__(self, file):
      while True:
          with h5py.File(file, 'r') as hf:
            #get the data
            		a_group_key = list(hf.keys())[0]
            		data = list(hf[a_group_key])

            		# transform to appropriate numpy array 
            		data = data[0:]
            		data = np.stack(data, axis=0)

            		yield data_preprocessing(data)

def data_preprocessing(data):

    #combine dimensions of 288 and 8 as in the CNN paper
    import functools
    def combine_dims(a, i=0, n=1):
      """
      Combines dimensions of numpy array `a`, 
      starting at index `i`,
      and combining `n` dimensions
      """
      s = list(a.shape)
      combined = functools.reduce(lambda x,y: x*y, s[i:i+n+1])
      return np.reshape(a, s[:i] + [combined] + s[i+n+1:])

    

    x = []
    y = []
    
    #load the data and split it into 13:3 chunks
    data = np.moveaxis(data, -1, 1)
    data = combine_dims(data, 0) # combines dimension 0 and 1

    for i in range(10):
        start = np.random.randint(0, len(data)-(X_Seq+Y_Seq))
        x.append(np.asarray(data[start:start+X_Seq]))
        y.append(np.pad(np.asarray(data[start+X_Seq:start+(X_Seq+Y_Seq)]), pad_width=((0,0),(1,0),(6,6)), mode='constant'))
    
    #divide the data by 255 and take square root
    x = np.array(x)
    y = np.array(y)
    x = np.moveaxis(x, 1, -1)
    y = np.moveaxis(y, 1, -1)
    x = x / 255.
    y = y / 255.
    
    return x, y

filenames = get_filenames(TRAIN_PATH)
ds = tf.data.Dataset.from_tensor_slices(filenames)

ds = ds.interleave(lambda filename: tf.data.Dataset.from_generator(
        generator(),
    (np.float32, np.float32),       
    args=(TRAIN_PATH + filename,)))



"""Deep Learning"""

import tensorflow
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import ZeroPadding2D, Input, Conv2D, MaxPool2D, Dropout, Flatten, Dense, MaxPooling2D, Conv2DTranspose, concatenate, UpSampling2D, ZeroPadding2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau

 

def unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
#Build the model
    KERNELS_INPUT = X_Seq // 2   
    KERNELS_OUT = Y_Seq

    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = ZeroPadding2D(((1,0),(6,6)))(inputs)
 

     #Contraction path
    c1 = Conv2D(KERNELS_INPUT, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(KERNELS_INPUT, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2,2))(c1)
    
    c2 = Conv2D(KERNELS_INPUT*2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(KERNELS_INPUT*2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2,2))(c2)
     
    c3 = Conv2D(KERNELS_INPUT*4, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(KERNELS_INPUT*4, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2,2))(c3)
     
    c4 = Conv2D(KERNELS_INPUT*4, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(KERNELS_INPUT*4, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2,2))(c4)
     
    c5 = Conv2D(KERNELS_INPUT*8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(KERNELS_INPUT*8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    #Expansive path 
    u6 = Conv2DTranspose(KERNELS_INPUT*8, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(KERNELS_INPUT*8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(KERNELS_INPUT*8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
     
    u7 = Conv2DTranspose(KERNELS_INPUT*4, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(KERNELS_INPUT*4, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(KERNELS_INPUT*4, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
     
    u8 = Conv2DTranspose(KERNELS_INPUT*4, (2, 2), strides=(2, 2), padding='same')(c7)

    u8 = concatenate([u8, c2])
    c8 = Conv2D(KERNELS_INPUT*2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(KERNELS_INPUT*2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
     
    u9 = Conv2DTranspose(KERNELS_INPUT, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(KERNELS_OUT, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(KERNELS_OUT, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
     
    outputs = Conv2D(KERNELS_OUT, (3, 3), activation='relu')(c9)
     
    model = Model(inputs=[inputs], outputs=[c9])
    model.compile(loss='mse',
                  optimizer = AdaBeliefOptimizer(learning_rate = 1e-2, 
                                                 print_change_log = False),
                  metrics=[tf.keras.metrics.MeanSquaredError()])
    model.summary()    
    return model

model = unet_model(495,436, 96)

from tensorflow.keras.callbacks import ReduceLROnPlateau
steps = np.int(np.floor(len(filenames)/BATCH_SIZE))
history = model.fit(ds, epochs=5, batch_size = BATCH_SIZE, steps_per_epoch = steps )
model.save('ISTANBUL.h5')
hist_df = pd.DataFrame(history.history) 
# or save to csv: 
hist_csv_file = 'ISTANBULhistory.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)