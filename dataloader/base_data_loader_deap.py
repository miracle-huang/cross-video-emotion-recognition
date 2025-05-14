import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import os
import random
import numpy as np
import scipy.io as sio
from tensorflow.keras.utils import to_categorical
from collections import Counter

import config

class BaseDataLoader:
    def __init__(self, dataset_dir, random_seed):
        '''
        dataset_dir - Address of the file to be loaded
        random_seed - Random seed
        random_label - Whether to set labe randomly. This variable is of Boolean type
        '''
        self.dataset_dir = dataset_dir
        self.random_seed = random_seed

    def processing_data_in_content_list(self, content_list):
        all_data = {'y_a_': [], 'y_v_': [], 'x_': []}

        for i in content_list:
            # Print the file name which in processing
            short_name = f'{i:02}'
            print("\nprocessing: ", short_name, "......") 

            # file_path = os.path.join(self.dataset_dir, 'DE_video' + short_name) # The file path of specific content
            file_path = os.path.join(self.dataset_dir, 'PSD_video' + short_name) 
            file = sio.loadmat(file_path)
            data = file['data'] # All data
            y_v = file['valence_labels'][0] # valence label
            y_a = file['arousal_labels'][0] # arousal label

            # Set valance and arousal labels based on participant_ratings
            if i in config.DEAP_half_valence_low:
                y_v[:] = 0
            else:
                y_v[:] = 1
            if i in config.DEAP_half_arousal_low:
                y_a[:] = 0
            else:
                y_a[:] = 1
        
            # One-Hot Encoding num_classes=2
            y_v = to_categorical(y_v, 2)
            y_a = to_categorical(y_a, 2)

            # Sort the loaded data into a form suitable for neural network training
            one_content_x = data.transpose([0, 2, 3, 1]) # data in one content
            one_content_y_v = np.empty([0, 2]) # valence label in one content
            one_content_y_a = np.empty([0, 2]) # arousal label in one content

            # Set lable in one content
            for j in range(len(y_a)):
                one_content_y_v = np.vstack((one_content_y_v, y_v[j]))
                one_content_y_a = np.vstack((one_content_y_a, y_a[j]))

            # Shuffling the data to introduce more randomness
            random_index = np.arange(len(one_content_x))
            np.random.shuffle(random_index)
            one_content_x = one_content_x[random_index]
            one_content_y_v = one_content_y_v[random_index]
            one_content_y_a = one_content_y_a[random_index]

            all_data['y_a_'].append(one_content_y_a)
            all_data['y_v_'].append(one_content_y_v)
            all_data['x_'].append(one_content_x)
        
        y_a_ = np.concatenate(all_data['y_a_'])
        y_v_ = np.concatenate(all_data['y_v_'])
        x_ = np.concatenate(all_data['x_'])

        return y_a_, y_v_, x_
    
if __name__ == '__main__':
    base_data_loader = BaseDataLoader(config.DEAP_dataset_path, 42)
    base_data_loader.processing_data_in_content_list(config.DEAP_valence_rating)