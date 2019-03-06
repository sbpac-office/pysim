# -*- coding: utf-8 -*-
"""
Application: Smart regenerative braking based on reinforment learning
======================================================================

Author
~~~~~~~~~~~~~
* kyunghan <kyunghah.min@gmail.com>

Description
~~~~~~~~~~~~~
* Network set for reinforcement learning

Update
~~~~~~~~~~~~~
* [19/03/05] - Initial draft design
"""
from keras.models import model_from_json
from keras.models import Sequential, load_model, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.wrappers import TimeDistributed
from keras.layers import Convolution2D, Dense, Flatten, merge, MaxPooling2D, Input, AveragePooling2D, Lambda, Merge, Activation, Embedding
from keras.optimizers import SGD, Adam, rmsprop
from keras.layers.recurrent import LSTM, GRU
from keras.layers.normalization import BatchNormalization
from keras import backend as K

import tensorflow as tf
import numpy as np
#%%


if __name__ == "__main__":
    input_shape = (10, 8)
    output_dim = 7
    learning_rate =  0.00005
    
    networks = Networks()
    model_agent = networks.drqn(input_shape, output_dim, learning_rate)