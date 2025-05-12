import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import os
import random

import numpy as np
import scipy.io as sio
from tensorflow.keras.utils import to_categorical

from dataloader.base_data_loader_deap import BaseDataLoader
import config

class LeaveOneVideoOutLoader(BaseDataLoader):
    def __init__(self, dataset_dir, random_seed, train_content_list, test_content_list):
        '''
        train_content_list - Content numbers used to train
        test_content_list - Content numbers used to test
        '''
        super().__init__(dataset_dir, random_seed)
        self.train_content_list = train_content_list
        self.test_content_list = test_content_list

    def load_data(self):
        random.seed(self.random_seed)

        # Initialize the dicts used to store train, val, and test data
        train_data = {'y_a_train': [], 'y_v_train': [], 'x_train': []}
        val_data = {'y_a_val': [], 'y_v_val': [], 'x_val': []}
        test_data = {'y_a_test': [], 'y_v_test': [], 'x_test': []}

        y_a_test, y_v_test, x_test = super().processing_data_in_content_list(self.test_content_list)
        y_a_, y_v_, x_ = super().processing_data_in_content_list(self.train_content_list) # y_a_, y_v_, x_ include train and val

        # Divide training set and verification set in 8:2
        all_data_size = len(x_)
        train_size = int(0.8 * all_data_size)
        val_size = int(0.2 * all_data_size)

        train_data['x_train'] = x_[:train_size]
        train_data['y_a_train'] = y_a_[:train_size]
        train_data['y_v_train'] = y_v_[:train_size]

        val_data['x_val'] = x_[train_size:]
        val_data['y_a_val'] = y_a_[train_size:]
        val_data['y_v_val'] = y_v_[train_size:]

        test_data['x_test'] = x_test
        test_data['y_a_test'] = y_a_test
        test_data['y_v_test'] = y_v_test

        return train_data, val_data, test_data
    
if __name__ == '__main__':
    leave_one_video_out_loader = LeaveOneVideoOutLoader(config.DEAP_dataset_path, 42, config.DEAP_all_videos_list, [1])
    train_data, val_data, test_data = leave_one_video_out_loader.load_data()