import numpy as np
import os
import scipy.io as sio

import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from process_util import segment_signal

dataset_dir = "D:/huangzhiying/cross-video-emotion-recognition/cross-video-emotion-recognition/dataset/DEAP/raw_data"

def read_file(file):
    data = sio.loadmat(file)
    data = data['data']
    return data

def process_data(file_path):
    data = read_file(file_path)
    start_index = 384
    for video_index in range(40):
        video_data = data[video_index]
        channel_f3_data = video_data[2]
        channel_f4_data = video_data[19]

        print()


def data_process_raw_deap():
    for file in os.listdir(dataset_dir):
        print("processing: ", file, "......")
        file_path = os.path.join(dataset_dir, file)
        file_path = os.path.join(dataset_dir, file).replace('\\', '/')
        process_data(file_path)
        

if __name__ == "__main__":
    data_process_raw_deap()
