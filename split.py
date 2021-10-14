
import os
from os import walk
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
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
import functools 
""""
Loads .h5 data files and splits them into smaller in order to fix OOM error.
"""

def data_preprocessing(data):

    #combine dimensions of 288 and 8 as in the CNN paper
    def combine_dims(a, i=0, n=1):
      """
      Combines dimensions of numpy array `a`, 
      starting at index `i`,
      and combining `n` dimensions
      """
      s = list(a.shape)
      combined = functools.reduce(lambda x,y: x*y, s[i:i+n+1])
      return np.reshape(a, s[:i] + [combined] + s[i+n+1:])

    
    #load the data and split it into 13:3 chunks
    data = np.moveaxis(data, -1, 1)
    data = combine_dims(data, 0) # combines dimension 0 and 1
    
    #divide the data by 255 and take square root
    data = np.array(data)
    data = np.moveaxis(data, 1, -1)
    data = data / 255.
    return data

def open_file(file):
    with h5py.File(file, 'r') as hf:
                #get the data
                a_group_key = list(hf.keys())[0]
                data = list(hf[a_group_key])

                # transform to appropriate numpy array 
                data = data[0:]
                data = np.stack(data, axis=0)
                data = data / 255.

                return data


def get_filenames(path):
    for (dirpath, dirnames, filenames) in walk(path):
        return filenames
    

files = get_filenames('MELBOURNE/training/')
files

for file in files:
    f = open_file('MELBOURNE/training/'+file)
    f_1 = f[:144]
    f_2 = f[144:]
    f_1 = data_preprocessing(f_1)
    f_2 = data_preprocessing(f_2)
    hf = h5py.File('split_MELBOURNE/'+file+'_1', 'w')
    hf.create_dataset('dataset', data=f_1, compression=9)
    hf.close()
    hf = h5py.File('split_MELBOURNE/'+file+'_2', 'w')
    hf.create_dataset('dataset', data=f_2, compression=9)
    hf.close()




