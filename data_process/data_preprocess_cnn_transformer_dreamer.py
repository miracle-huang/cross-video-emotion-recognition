import numpy as np
import os
import scipy.io as sio
from scipy.io import loadmat

import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from process_util import segment_signal, calculate_0_and_nan, butter_bandpass_filter
import config

frequency = 128  # 采样频率
video_count = 18  # 视频数量
channel_count = 14  # 通道数量
subject_count = 23  # 受试者数量

def average_reference(video_eeg_stimuli):
    """
    对EEG数据进行平均参考处理
    """
    mean_ref = np.mean(video_eeg_stimuli, axis=1, keepdims=True)
    video_eeg_stimuli_ref = video_eeg_stimuli - mean_ref
    return video_eeg_stimuli_ref

def data_processing_dreamer():
    all_video_list_eeg = [
        [ [] for _ in range(channel_count) ]
        for _ in range(video_count)
    ]
    dreamer_dataset = sio.loadmat('dataset/Dreamer/DREAMER.mat')    
    dreamer_struct = dreamer_dataset['DREAMER']
    dreamer_data = dreamer_struct[0, 0]['Data']
    for subject_index in range(subject_count):
        print(f"Processing subject {subject_index + 1}/{subject_count}")
        subject_data = dreamer_data[0, subject_index]

        # 获取EEG数据
        eeg_data = subject_data['EEG'][0, 0]
        stimuli = eeg_data['stimuli'][0, 0]

        # 遍历每个视频
        for video_index in range(video_count):
            video_eeg_stimuli = stimuli[video_index][0]
            video_eeg_stimuli_ref = average_reference(video_eeg_stimuli)
            video_eeg = video_eeg_stimuli_ref.transpose((1, 0))  # 转置为 (channel, time)
            for channel_index in range(channel_count):
                channel_data_raw = video_eeg[channel_index]
                channel_data = butter_bandpass_filter(channel_data_raw, 0.5, 45, frequency, order=5)

                nan_count, zero_count = calculate_0_and_nan(channel_data)
                if nan_count > 0:
                    print(f"视频 {video_index + 1} 通道 {config.dreamer_channel_mapping[channel_index]} 存在 {nan_count} 个 NaN 值")
                    all_video_list_eeg[video_index][channel_index].extend([])
                else:
                    channel_windows = segment_signal(channel_data, config.window_size_10, config.overlap)
                    # print(f"视频 {video_index + 1} 通道 {config.dreamer_channel_mapping[channel_index]} 窗口数量: {len(channel_windows)}")
                    all_video_list_eeg[video_index][channel_index].extend(channel_windows)
        
    print("所有视频处理完成，开始保存数据...")
    for video_index in range(video_count):
        video_eeg_array = np.array(all_video_list_eeg[video_index]) 
        arousal_labels_eeg = np.repeat(config.DREAMER_video_arousal_labels[int(video_index)], video_eeg_array.shape[1])
        valence_labels_eeg = np.repeat(config.DREAMER_video_valence_labels[int(video_index)], video_eeg_array.shape[1])

        print('video_index:', video_index)
        print('arousal label:', config.DREAMER_video_arousal_labels[int(video_index)])
        print('valence label:', config.DREAMER_video_valence_labels[int(video_index)])

        # processed_data_eeg = {
        #     'eeg_data': video_eeg_array,
        #     'arousal_labels': arousal_labels_eeg,
        #     'valence_labels': valence_labels_eeg
        # }
        # result_dir = 'dataset/dreamer/cnn_transformer/window_10s/'
        # sio.savemat(result_dir + "EEG_video" + str(video_index + 1).zfill(2) + ".mat", processed_data_eeg)
        # print("Saved video {} processed video data ".format(video_index + 1))

if __name__ == "__main__":
    data_processing_dreamer()