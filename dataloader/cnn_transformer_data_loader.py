import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import os
import numpy as np
import scipy.io as sio

import config

class CNNTransformerDataLoader:
    def __init__(self, dataset_dir, random_seed):
        self.dataset_dir = dataset_dir
        self.random_seed = random_seed
    
    def load_data(self, video_list):
        all_data = {'y_a_': [], 'y_v_': [], 'x_': []}

        for i in video_list:
            # Print the file name which in processing
            short_name = f'{i:02}'
            print("\nprocessing: ", short_name, "......") 

            file_path = os.path.join(self.dataset_dir, 'EEG_video' + short_name)
            file = sio.loadmat(file_path)

            eeg_data = file['eeg_data']
            y_v = file['valence_labels'].flatten()  # 确保 y_v 是一维数组
            y_a = file['arousal_labels'].flatten()  # 确保 y_a 是一维数组

            # 反转标签，验证准确率极低的原因
            y_v ^=1
            y_a ^=1

            all_data['y_a_'].append(y_a)
            all_data['y_v_'].append(y_v)
            all_data['x_'].append(eeg_data)

        y_a_ = np.concatenate(all_data['y_a_'])
        y_v_ = np.concatenate(all_data['y_v_'])
        x_ = np.concatenate(all_data['x_'], axis=1).swapaxes(0, 1)

        # 检查是否还有 NaN 值
        nan_count = np.isnan(x_).sum()
        print(f"NaN count in x_: {nan_count}")

        return y_a_, y_v_, x_
    
    def set_train_val_test(self, train_video_list, test_video_list):
        train_data = {'y_a_train': [], 'y_v_train': [], 'x_train': []}
        val_data = {'y_a_val': [], 'y_v_val': [], 'x_val': []}
        test_data = {'y_a_test': [], 'y_v_test': [], 'x_test': []}

        y_a_, y_v_, x_ = self.load_data(train_video_list)
        y_a_test, y_v_test, x_test = self.load_data(test_video_list)

        # Shuffling the data to introduce more randomness
        random_index = np.arange(len(x_))
        np.random.shuffle(random_index)
        x_ = x_[random_index]
        y_v_ = y_v_[random_index]
        y_a_ = y_a_[random_index]

        unique_v, counts_v = np.unique(y_v_, return_counts=True)
        count_dict = dict(zip(unique_v, counts_v))
        print('y_v_当中0和1的个数', count_dict)  # 输出：{0: count0, 1: count1}
        unique_a, counts_a = np.unique(y_a_, return_counts=True)
        count_dict = dict(zip(unique_a, counts_a))
        print('y_a_当中0和1的个数', count_dict)  # 输出：{0: count0, 1: count1}

        # Divide training set and verification set in 8:2
        all_data_size = len(x_)
        train_size = int(0.8 * all_data_size)
        val_size = int(0.2 * all_data_size)

        # 选择用哪些channel进行实验
        # select_channels = [2, 11]  # FC6, F4 [10, 11]; P8, AF4 [8, 13]; F3, F4 [2, 11]

        # train_data['x_train'] = x_[:train_size][:, select_channels, :]
        train_data['x_train'] = x_[:train_size]
        train_data['y_a_train'] = y_a_[:train_size]
        train_data['y_v_train'] = y_v_[:train_size]

        val_data['x_val'] = x_[train_size:]
        val_data['y_a_val'] = y_a_[train_size:]
        val_data['y_v_val'] = y_v_[train_size:]

        test_data['x_test'] = x_test
        test_data['y_a_test'] = y_a_test
        test_data['y_v_test'] = y_v_test

        unique_v_train, counts_v_train = np.unique(train_data['y_v_train'], return_counts=True)
        count_dict = dict(zip(unique_v_train, counts_v_train))
        print('y_v_train当中0和1的个数', count_dict)  # 输出：{0: count0, 1: count1}
        unique_a_train, counts_a_train = np.unique(train_data['y_a_train'], return_counts=True)
        count_dict = dict(zip(unique_a_train, counts_a_train))
        print('y_a_train当中0和1的个数', count_dict)  # 输出：{0: count0, 1: count1}

        return train_data, val_data, test_data


if __name__ == '__main__':
    dataloader = CNNTransformerDataLoader(dataset_dir=config.AMIGO_cnn_transformer_dataset_path, random_seed=42)
    # dataloader.load_data(config.AMIGO_all_videos_list)
    dataloader.set_train_val_test(config.AMIGO_all_videos_list, [1])