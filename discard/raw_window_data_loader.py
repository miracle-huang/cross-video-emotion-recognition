import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import os
import numpy as np
import scipy.io as sio
from tensorflow.keras.utils import to_categorical

import config

class RawWindowDataLoader:
    def __init__(self, dataset_dir, random_seed):
        self.dataset_dir = dataset_dir
        self.random_seed = random_seed

    def load_data(self, video_list):
        all_data = {'y_a_': [], 'y_v_': [], 'x_': []}

        for i in video_list:
            # Print the file name which in processing
            short_name = f'{i:02}'
            print("\nprocessing: ", short_name, "......") 

            file_path = os.path.join(self.dataset_dir, 'ECG_video' + short_name)
            file = sio.loadmat(file_path)
            # eeg_data = file['eeg_data']  # EEG data
            ecg_data = file['ecg_data']  # ECG data
            y_v = file['valence_labels']
            y_a = file['arousal_labels']

            # One-Hot Encoding num_classes=2
            one_video_y_v = to_categorical(y_v, 2)[0]
            one_video_y_a = to_categorical(y_a, 2)[0]
            one_video_x = ecg_data
            
            # Shuffling the data to introduce more randomness
            random_index = np.arange(len(one_video_x))
            np.random.shuffle(random_index)
            one_video_x = one_video_x[random_index]
            one_video_y_v = one_video_y_v[random_index]
            one_video_y_a = one_video_y_a[random_index]

            all_data['y_a_'].append(one_video_y_a)
            all_data['y_v_'].append(one_video_y_v)
            all_data['x_'].append(one_video_x)

        y_a_ = np.concatenate(all_data['y_a_'])
        y_v_ = np.concatenate(all_data['y_v_'])
        x_ = np.concatenate(all_data['x_'])

        nan_count = np.isnan(x_).sum()
        print(f"NaN count in x_: {nan_count}")

        return y_a_, y_v_, x_

    def set_train_val_test(self, train_video_list, test_video_list):
        train_data = {'y_a_train': [], 'y_v_train': [], 'x_train': []}
        val_data = {'y_a_val': [], 'y_v_val': [], 'x_val': []}
        test_data = {'y_a_test': [], 'y_v_test': [], 'x_test': []}

        y_a_, y_v_, x_ = self.load_data(train_video_list)
        y_a_test, y_v_test, x_test = self.load_data(test_video_list)

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
    dataloader = RawWindowDataLoader(dataset_dir=config.AMIGO_raw_window_path, random_seed=42)
    dataloader.set_train_val_test(config.AMIGO_all_videos_list, [1])
        