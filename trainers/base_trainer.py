#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   base_trainer.py
@Time    :   2024/07/05 03:07:31
@Author  :   CrissChan 
@Email    :   zhiying.huang.4g@stu.hosei.ac.jp
@Description    :   Defined the parent class of all trainers
                    For each different model, there will be a subclass that inherits BaseTrainer
'''

import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import tensorflow as tf

class BaseTrainer:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        # self.val_data = val_data
        self.test_data = test_data
        self.callbacks = self.lr_es_scheduler()

    # Learning rate decay function
    @staticmethod
    def lr_scheduler(epoch, lr):
        decay_rate = 0.5  # Adjust the decay rate as needed
        decay_step = 5   # Decayed every 5 epochs
        if epoch % decay_step == 0 and epoch:
            return lr * decay_rate
        return lr
    
    def lr_es_scheduler(self):
        callbacks = [
                tf.keras.callbacks.LearningRateScheduler(self.lr_scheduler, verbose=1),
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=16, verbose=1)
            ]
        return callbacks