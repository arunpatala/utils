import keras
import os
from scipy.misc import imread
from keras.models import load_model
import pickle

import pandas as pd
import numpy as np
from keras.utils.np_utils import to_categorical


def irctc_data(dirpath, N=100, max_size = 8):

    l = [y[0].upper().replace('2','@').replace('O','Q').replace('C','G') for y in pd.read_csv('irctc.csv', header=None).as_matrix().tolist()]


    chars = ['3', '4', '6', '7', '9', '=', '@', 'A', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'T', 'U',
     'V', 'W', 'X', 'Y', 'Z']

    char_indices = dict((c, i+1) for i, c in enumerate(chars))
    indices_char = dict((i+1, c) for i, c in enumerate(chars))
    
    num_classes = len(chars) + 1

    def load(cnt=0, ffmt=dirpath+'/{}.png', ln=8):
        X = imread(ffmt.format(cnt))
        Y = l[cnt]
        y = [char_indices.get(y,0) for y in Y]
        return X, y + [0 for i in range(ln-len(y))]

    def loadN(frm=0, to=10):
        X,Y = [], []
        for i in range(frm, to):
            x, y = load(i) 
            X.append(x)
            Y.append(y)
        return np.array(X), np.array(Y)


    
    x_train, y_train = loadN(1,N)
    
    x_train = x_train.astype('float32')
    x_train /= 255
    print('x_train shape:', x_train.shape)

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train[:,0], num_classes)
    x_train = x_train[:,:,:50,:1]
    input_shape = x_train[0].shape
    return (x_train, y_train), (x_train, y_train), input_shape