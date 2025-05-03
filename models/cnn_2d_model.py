#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   cnn_2d_model.py
@Time    :   2024/07/05 02:41:42
@Author  :   CrissChan 
@Email    :   zhiying.huang.4g@stu.hosei.ac.jp
@Description    :   Only one emotion can be dichotomized in this model
'''

import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.layers import Flatten, Dense, Reshape, Average
from tensorflow.keras.models import Sequential, Model

import config

class CnnTwoDimensionModel:
    '''
    filters - list of the number of Convolution kernel of the model
    kernel_size_list - List of convolution kernel sizes
    dropout_rate - from config
    learning_rate - from config
    '''
    def __init__(self, filters, kernel_size_list, dropout_rate, learning_rate):
        self.filters = filters
        self.kernel_size_list = kernel_size_list
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate

    def create_2d_cnn_model(self):
        # create base network
        seq = Sequential()
        seq.add(Conv2D(self.filters[0], self.kernel_size_list[0], activation='relu', padding='same', name='conv1', input_shape=(8, 9, 8)))
        seq.add(BatchNormalization())
        seq.add(Dropout(self.dropout_rate))
        seq.add(Conv2D(self.filters[1], self.kernel_size_list[1], activation='relu', padding='same', name='conv2'))
        seq.add(BatchNormalization())
        seq.add(Dropout(self.dropout_rate))
        seq.add(Conv2D(self.filters[2], self.kernel_size_list[2], activation='relu', padding='same', name='conv3'))
        seq.add(BatchNormalization())
        seq.add(Dropout(self.dropout_rate))
        seq.add(Conv2D(self.filters[0], self.kernel_size_list[3], activation='relu', padding='same', name='conv4'))
        seq.add(MaxPooling2D(2, 2, name='pool1'))
        seq.add(BatchNormalization())
        seq.add(Dropout(self.dropout_rate))
        seq.add(Flatten(name='fla1'))
        seq.add(Dense(512, activation='relu', name='dense1'))
        seq.add(Reshape((1, 512), name='reshape'))
        seq.add(BatchNormalization())
        seq.add(Dropout(self.dropout_rate))

        input_layer = Input(shape=(8, 9, 8))
        x = seq(input_layer)
        x = Flatten(name='flat')(x)
        out = Dense(2, activation='softmax', name='out')(x)
        model = Model([input_layer], out)

        model.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
            loss_weights=[1,1])

        return model