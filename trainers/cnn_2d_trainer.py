#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   cnn_2d_trainer.py
@Time    :   2024/07/07 05:34:47
@Author  :   CrissChan 
@Email    :   zhiying.huang.4g@stu.hosei.ac.jp
@Description    :   The trainer of 2D cnn model
'''

import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

from trainers.base_trainer import BaseTrainer
import config

class CnnTwoDimensionTrainer(BaseTrainer):
    def __init__(self, train_data, test_data, model, emotion_type):
        '''
        model - One trainer class only train one model in once time
        '''
        super().__init__(train_data, test_data)
        self.model = model

        # Check the value of emotion_type
        if emotion_type not in ['arousal', 'valence']:
            raise ValueError("emotion_type must be either 'arousal' or 'valence'")
        self.emotion_type = emotion_type

    def train_model(self):
        X_train = self.train_data['x'] # Total training data
        y_a_train = self.train_data['y_a'] # Total training arousal label
        y_v_train = self.train_data['y_v'] # Total training valence label

        # Divide total training data into training data and test data according to 8:2
        X_train_length = len(X_train)
        split_index = int(X_train_length * 0.8)
        x_train = X_train[:split_index] # training data
        x_val = X_train[split_index:] # validation data

        y_a_train_ = y_a_train[:split_index] # train arousal label
        y_a_val = y_a_train[split_index:] # validation arousal label

        y_v_train_ = y_v_train[:split_index] # train valence label
        y_v_val = y_v_train[split_index:] # validation valence label

        x_test = self.test_data['x'] # test data
        y_a_test = self.test_data['y_a'] # test arousal label
        y_v_test = self.test_data['y_v'] # test valence label

        # select which kind of label by emotion type
        y_train = y_val = y_test = None # train, validation, test type
        if self.emotion_type == "valence":
            y_train = y_v_train_
            y_val = y_v_val
            y_test = y_v_test
        elif self.emotion_type == "arousal":
            y_train = y_a_train_
            y_val = y_a_val
            y_test = y_a_test

        self.model.fit([x_train[:, i] for i in range(x_train.shape[1])], # train data, Generates a list where each element is a column of x_train (feature)
            y_train,
            validation_data=([x_val[:, i] for i in range(x_val.shape[1])],
                        y_val),
            # sample_weight=sample_weights,
            epochs=config.epoch,
            batch_size=config.batch_size,
            callbacks=self.callbacks)
        
        # get result
        result = self.model.evaluate([x_test[:, i] for i in range(x_test.shape[1])], y_test)
        result_output = f"{self.emotion_type} accuracy: {result[1]}"
        acc_output = result[1]
        print(result_output)

        # compute confusion matrix and f1 score
        predictions = self.model.predict([x_test[:, i] for i in range(x_test.shape[1])])
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(y_test, axis=1)
        conf_matrix = confusion_matrix(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')

        return acc_output, conf_matrix, f1
