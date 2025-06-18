import numpy as np
import os
import scipy.io as sio
from scipy.io import loadmat

import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from process_util import segment_signal, calculate_0_and_nan
import config

dataset_dir = "dataset/DEAP/raw_data"
frequency = 128  # 采样频率
video_count = 40  # 视频数量
channel_count = 32  # 通道数量

def read_file(file):
    data = sio.loadmat(file)
    data = data['data']
    return data

def data_processing_deap():
    all_video_list_eeg = [
        [ [] for _ in range(channel_count) ]
        for _ in range(video_count)
    ]
    for file in os.listdir(dataset_dir):
        print("processing: ", file, "......")
        file_path = os.path.join(dataset_dir, file)
        data = read_file(file_path)
        for video_index in range(video_count):
            for channel_index in range(channel_count):
                channel_data = data[video_index][channel_index]
                channel_data = channel_data[384:]
                nan_count, zero_count = calculate_0_and_nan(channel_data)

                if nan_count > 0 or zero_count > 0:
                    print(f"视频 {video_index + 1} 通道 {channel_index + 1} 存在 {nan_count} 个 NaN 和 {zero_count} 个零值")
                    all_video_list_eeg[video_index][channel_index].extend([])
                else:
                    channel_windows = segment_signal(channel_data, config.window_size_10, config.overlap)
                    # print(f"视频 {video_index + 1} 通道 {channel_index + 1} 窗口数量: {len(channel_windows)}")
                    all_video_list_eeg[video_index][channel_index].extend(channel_windows)

    print("所有视频处理完成，开始保存数据...")
    for video_index in range(video_count):
        video_eeg_array = np.array(all_video_list_eeg[video_index]) # (32, 192, 1280)
        
        if video_index + 1 in config.DEAP_half_valence_low:
            valence_labels_eeg = np.repeat(0, video_eeg_array.shape[1])
        else:
            valence_labels_eeg = np.repeat(1, video_eeg_array.shape[1])
        if video_index + 1 in config.DEAP_half_arousal_low:
            arousal_labels_eeg = np.repeat(0, video_eeg_array.shape[1])
        else:
            arousal_labels_eeg = np.repeat(1, video_eeg_array.shape[1])
        
        processed_data_eeg = {
            'eeg_data': video_eeg_array,
            'arousal_labels': arousal_labels_eeg,
            'valence_labels': valence_labels_eeg
        }
        result_dir = 'dataset/DEAP/cnn_transformer/window_10s/'
        sio.savemat(result_dir + "EEG_video" + str(video_index + 1).zfill(2) + ".mat", processed_data_eeg)
        print("Saved video {} processed video data ".format(video_index + 1))

if __name__ == "__main__":
    data_processing_deap()