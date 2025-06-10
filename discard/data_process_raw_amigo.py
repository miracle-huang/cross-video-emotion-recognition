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

def process_data_in_one_video(signal_data, video_index):
    channel_f3_data = signal_data[video_index][:, 2]
    channel_f4_data = signal_data[video_index][:, 11]
    channel_ECG_right_data = signal_data[video_index][:, 14]
    channel_ECG_left_data = signal_data[video_index][:, 15]

    channel_EEG_dict = {
            'F3': channel_f3_data,
            'F4': channel_f4_data,
        }
    channel_ECG_dict = {
            'ECG_right': channel_ECG_right_data,
            'ECG_left': channel_ECG_left_data
        }
    
    channel_windows_EEG = []
    channel_windows_ECG = []

    for channel_name, channel_data in channel_EEG_dict.items():
        nan_count, zero_count = calculate_0_and_nan(channel_data)
        if nan_count > 0 or zero_count > 0:
            print(f"视频 {video_index + 1} 通道 {channel_name} 存在 {nan_count} 个 NaN 和 {zero_count} 个零值")
            channel_windows_EEG.extend([])
        else:
            channel_windows = segment_signal(channel_data, config.window_size_10, config.overlap)
            print(f"视频 {video_index + 1} 通道 {channel_name} 窗口数量: {len(channel_windows)}")
            channel_windows_EEG.extend(channel_windows)

    for channel_name, channel_data in channel_ECG_dict.items():
        nan_count, zero_count = calculate_0_and_nan(channel_data)
        if nan_count > 0 or zero_count > 0:
            print(f"视频 {video_index + 1} 通道 {channel_name} 存在 {nan_count} 个 NaN 和 {zero_count} 个零值")
            channel_windows_ECG.extend([])
        else:
            channel_windows = segment_signal(channel_data, config.window_size_10, config.overlap)
            print(f"视频 {video_index + 1} 通道 {channel_name} 窗口数量: {len(channel_windows)}")
            channel_windows_ECG.extend(channel_windows)

    return channel_windows_EEG, channel_windows_ECG

def data_processing_amigo():
    all_video_list_eeg = [[] for _ in range(video_count)]
    all_video_list_ecg = [[] for _ in range(video_count)]
    for dirpath, dirnames, filenames in os.walk(dataset_dir):
        for filename in filenames:
            if filename.endswith('.mat'):
                mat_file_path = os.path.join(dirpath, filename)
            
                print(f"找到 .mat 文件: {mat_file_path}")
                
                all_data = sio.loadmat(mat_file_path)
                signal_data = all_data['joined_data'][0]
                for video_index in range(video_count):
                    channel_windows_EEG, channel_windows_ECG = process_data_in_one_video(signal_data, video_index)
                    all_video_list_eeg[video_index].extend(channel_windows_EEG)
                    all_video_list_ecg[video_index].extend(channel_windows_ECG)

    print("所有视频处理完成，开始保存数据...")
    for video_index in range(video_count):
        
        video_eeg_array = np.array(all_video_list_eeg[video_index])
        arousal_labels_eeg = np.repeat(config.AMIGO_video_arousal_labels[int(video_index)], video_eeg_array.shape[0])
        valence_labels_eeg = np.repeat(config.AMIGO_video_valence_labels[int(video_index)], video_eeg_array.shape[0])

        video_ecg_array = np.array(all_video_list_ecg[video_index])
        arousal_labels_ecg = np.repeat(config.AMIGO_video_arousal_labels[int(video_index)], video_ecg_array.shape[0])
        valence_labels_ecg = np.repeat(config.AMIGO_video_valence_labels[int(video_index)], video_ecg_array.shape[0])

        processed_data_eeg = {
            'eeg_data': video_eeg_array,
            'arousal_labels': arousal_labels_eeg,
            'valence_labels': valence_labels_eeg
        }
        processed_data_ecg = {
            'ecg_data': video_ecg_array,
            'arousal_labels': arousal_labels_ecg,
            'valence_labels': valence_labels_ecg
        }

        result_dir = 'dataset/amigo/raw_window_data_10s/'
        sio.savemat(result_dir + "EEG_video" + str(video_index + 1).zfill(2) + ".mat", processed_data_eeg)
        sio.savemat(result_dir + "ECG_video" + str(video_index + 1).zfill(2) + ".mat", processed_data_ecg)
        print("Saved video {} processed video data ".format(video_index + 1))

if __name__ == "__main__":
    data_processing_amigo()