import numpy as np
import os
import scipy.io as sio

import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from process_util import segment_signal, calculate_0_and_nan
import config

dataset_dir = "dataset/amigo/raw_data/"
frequency = 128  # 采样频率
video_count = 16  # 视频数量
channel_count = 14  # 通道数量

def process_data_in_one_subject(signal_data):
    data_in_one_subject = []
    
    for video_index in range(video_count):
        video_data = [[] for _ in range(channel_count)]

        for channel_index in range(channel_count):
            channel_data = signal_data[video_index][:, channel_index]

            nan_count, zero_count = calculate_0_and_nan(channel_data)
            if nan_count > 0 or zero_count > 0:
                print(f"视频 {video_index + 1} 通道 {config.amigo_channel_mapping[channel_index]} 存在 {nan_count} 个 NaN 和 {zero_count} 个零值")
                video_data[channel_index].extend([])
            else:
                channel_windows = segment_signal(channel_data, config.window_size_10, config.overlap)
                print(f"视频 {video_index + 1} 通道 {config.amigo_channel_mapping[channel_index]} 窗口数量: {len(channel_windows)}")
                video_data[channel_index].extend(channel_windows)

        data_in_one_subject.append(video_data)

    return data_in_one_subject

def data_processing_amigo():
    all_video_list_eeg = [[[] for _ in range(channel_count)] for _ in range(video_count)]
    for dirpath, dirnames, filenames in os.walk(dataset_dir):
        for filename in filenames:
            if filename.endswith('.mat'):
                mat_file_path = os.path.join(dirpath, filename)
                print(f"找到 .mat 文件: {mat_file_path}")

                all_data = sio.loadmat(mat_file_path)
                signal_data = all_data['joined_data'][0]
                data_in_one_subject = process_data_in_one_subject(signal_data)
                for video_index in range(video_count):
                    for channel_index in range(channel_count):
                        all_video_list_eeg[video_index][channel_index].extend(data_in_one_subject[video_index][channel_index])
    
    print("所有视频处理完成，开始保存数据...")
    for video_index in range(video_count):
        video_eeg_array = np.array(all_video_list_eeg[video_index])
        arousal_labels_eeg = np.repeat(config.AMIGO_video_arousal_labels[int(video_index)], video_eeg_array.shape[1])
        valence_labels_eeg = np.repeat(config.AMIGO_video_valence_labels[int(video_index)], video_eeg_array.shape[1])
        
        processed_data_eeg = {
            'eeg_data': video_eeg_array,
            'arousal_labels': arousal_labels_eeg,
            'valence_labels': valence_labels_eeg
        }
        result_dir = 'dataset/amigo/cnn_transformer/window_10s/'
        sio.savemat(result_dir + "EEG_video" + str(video_index + 1).zfill(2) + ".mat", processed_data_eeg)
        print("Saved video {} processed video data ".format(video_index + 1))

if __name__ == "__main__":
    data_processing_amigo()