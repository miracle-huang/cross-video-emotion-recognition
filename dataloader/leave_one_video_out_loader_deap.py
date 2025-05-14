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

    def normalize_non_zero_values_with_stats(self, data_tensor, min_val=None, max_val=None):
        """
        使用给定的最小值和最大值对张量中的非零值进行归一化，保持零值不变
        如果未提供min_val和max_val，则从数据本身计算
        
        参数:
        data_tensor: 需要归一化的张量
        min_val: 用于归一化的最小值（可选）
        max_val: 用于归一化的最大值（可选）
        
        返回:
        归一化后的张量和使用的min_val、max_val
        """
        # 创建一个掩码，标记所有非零元素
        mask = (data_tensor != 0)
        
        # 如果没有非零元素，直接返回原始张量
        if not np.any(mask):
            return data_tensor, min_val, max_val
        
        # 如果未提供min_val和max_val，从数据中计算
        if min_val is None or max_val is None:
            # 获取所有非零元素
            non_zero_values = data_tensor[mask]
            
            # 计算非零元素的最小值和最大值
            min_val = np.min(non_zero_values)
            max_val = np.max(non_zero_values)
        
        # 如果最大值和最小值相同，直接将所有非零值设为1（或其他合适的值）
        if max_val == min_val:
            normalized_tensor = data_tensor.copy()
            normalized_tensor[mask] = 1.0
            return normalized_tensor, min_val, max_val
        
        # 创建一个新的张量用于存储结果
        normalized_tensor = data_tensor.copy()
        
        # 只归一化非零元素
        normalized_tensor[mask] = (data_tensor[mask] - min_val) / (max_val - min_val)
        
        return normalized_tensor, min_val, max_val

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

        # 对训练数据进行非零值归一化，并获取归一化参数
        train_data['x_train'], min_val, max_val = self.normalize_non_zero_values_with_stats(train_data['x_train'])

        # 使用训练集的min_val和max_val对验证集和测试集进行归一化
        val_data['x_val'], _, _ = self.normalize_non_zero_values_with_stats(val_data['x_val'], min_val, max_val)
        test_data['x_test'], _, _ = self.normalize_non_zero_values_with_stats(test_data['x_test'], min_val, max_val)

        return train_data, val_data, test_data
    
if __name__ == '__main__':
    leave_one_video_out_loader = LeaveOneVideoOutLoader(config.DEAP_dataset_path, 42, config.DEAP_all_videos_list, [1])
    train_data, val_data, test_data = leave_one_video_out_loader.load_data()