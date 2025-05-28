import math
import re
import os
import glob

import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from scipy.io import loadmat

import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import config
from process_util import butter_bandpass_filter, compute_DE, compute_PSD, segment_signal, array_1Dto2D, calculate_0_and_nan, is_valid_for_stacking


dreamer_dataset = sio.loadmat('dataset/Dreamer/DREAMER.mat')    

dreamer_struct = dreamer_dataset['DREAMER']
dreamer_data = dreamer_struct[0, 0]['Data'] 

# 获取受试者数量和视频数量
n_subjects = dreamer_data.shape[1]
n_videos = 18  # 有18个视频

frequency = 128  # 采样频率

def average_reference(video_eeg_stimuli):
    """
    对EEG数据进行平均参考处理
    """
    mean_ref = np.mean(video_eeg_stimuli, axis=1, keepdims=True)
    video_eeg_stimuli_ref = video_eeg_stimuli - mean_ref
    return video_eeg_stimuli_ref

def remove_outliers_rolling(eeg_data, window_size=config.window_size, threshold=3):
    """
    eeg_data: np.ndarray, shape = (time, channel)
    window_size: 滑动窗口大小
    threshold: 判定异常点的标准差倍数
    """
    eeg_cleaned = np.copy(eeg_data)

    for ch in range(eeg_data.shape[1]):  # 遍历每个通道
        series = pd.Series(eeg_data[:, ch])
        
        # 计算滚动均值和标准差（center=True 表示当前点在窗口中间）
        rolling_mean = series.rolling(window=window_size, center=True).mean()
        rolling_std = series.rolling(window=window_size, center=True).std()
        
        # 判定异常点：偏离超过 threshold × std
        outliers = np.abs(series - rolling_mean) > (threshold * rolling_std)
        
        # 用 NaN 替换异常值
        series[outliers] = np.nan
        
        # 插值填补 NaN（默认线性插值）
        series = series.interpolate(method='linear', limit_direction='both')

        # 回写清洗后的通道数据
        eeg_cleaned[:, ch] = series.to_numpy()

        plt.plot(eeg_data[:, 0], label='Original')
        plt.plot(eeg_cleaned[:, 0], label='After Cleaning')
        plt.legend()
        plt.title("comparition of original and cleaned data")
        plt.show()
        
        print(f"通道 {ch} 完成，去除异常点数量: {np.sum(outliers)}")
    
    return eeg_cleaned

def process_data_in_one_video(signal_data, video_index):
    video_de_theta = []
    video_de_alpha = []
    video_de_beta = []
    video_de_gamma = []
    video_psd_theta = []
    video_psd_alpha = []
    video_psd_beta = []
    video_psd_gamma = []

    for channel_index in range(14):
        one_channel_data = signal_data[:, channel_index]

        nan_count, zero_count = calculate_0_and_nan(one_channel_data)
        if nan_count > 0:
            print(f"视频 {video_index + 1} 通道 {channel_index + 1} 存在 {nan_count} 个 NaN")
            continue
        # if zero_count > 0:
        #     # 如果存在零值，打印警告信息,但不跳过
        #     print(f"视频 {video_index + 1} 通道 {channel_index + 1} 存在  {zero_count} 个零值")   
        de_theta_channel = np.zeros(shape=[0], dtype=float)
        de_alpha_channel = np.zeros(shape=[0], dtype=float)
        de_beta_channel = np.zeros(shape=[0], dtype=float)
        de_gamma_channel = np.zeros(shape=[0], dtype=float)
        psd_theta_channel = np.zeros(shape=[0], dtype=float)
        psd_alpha_channel = np.zeros(shape=[0], dtype=float)
        psd_beta_channel = np.zeros(shape=[0], dtype=float)
        psd_gamma_channel = np.zeros(shape=[0], dtype=float)
        channel_windows = segment_signal(one_channel_data, config.window_size, config.overlap)

        for window in channel_windows:
            theta_eeg = butter_bandpass_filter(window, 4, 8, frequency)
            alpha_eeg = butter_bandpass_filter(window, 8, 14, frequency)
            beta_eeg = butter_bandpass_filter(window, 14, 31, frequency)
            gamma_eeg = butter_bandpass_filter(window, 31, 45, frequency)

            # 计算DE和PSD
            de_theta_channel = np.append(de_theta_channel, compute_DE(theta_eeg))
            de_alpha_channel = np.append(de_alpha_channel, compute_DE(alpha_eeg))
            de_beta_channel = np.append(de_beta_channel, compute_DE(beta_eeg))
            de_gamma_channel = np.append(de_gamma_channel, compute_DE(gamma_eeg))
            psd_theta_channel = np.append(psd_theta_channel, compute_PSD(theta_eeg, 4, 8))
            psd_alpha_channel = np.append(psd_alpha_channel, compute_PSD(alpha_eeg, 8, 14))
            psd_beta_channel = np.append(psd_beta_channel, compute_PSD(beta_eeg, 14, 31))
            psd_gamma_channel = np.append(psd_gamma_channel, compute_PSD(gamma_eeg, 31, 45))

        video_de_theta.append(de_theta_channel)
        video_de_alpha.append(de_alpha_channel)
        video_de_beta.append(de_beta_channel)
        video_de_gamma.append(de_gamma_channel)
        video_psd_theta.append(psd_theta_channel)
        video_psd_alpha.append(psd_alpha_channel)
        video_psd_beta.append(psd_beta_channel)
        video_psd_gamma.append(psd_gamma_channel)

    array_video_de_theta = np.array(video_de_theta).T
    array_video_de_alpha = np.array(video_de_alpha).T
    array_video_de_beta = np.array(video_de_beta).T
    array_video_de_gamma = np.array(video_de_gamma).T
    array_video_psd_theta = np.array(video_psd_theta).T
    array_video_psd_alpha = np.array(video_psd_alpha).T
    array_video_psd_beta = np.array(video_psd_beta).T
    array_video_psd_gamma = np.array(video_psd_gamma).T

    # 将1D数据转换为2D数据
    array_video_de_theta_2D = array_1Dto2D(array_video_de_theta)
    array_video_de_alpha_2D = array_1Dto2D(array_video_de_alpha)
    array_video_de_beta_2D = array_1Dto2D(array_video_de_beta)
    array_video_de_gamma_2D = array_1Dto2D(array_video_de_gamma)
    array_video_psd_theta_2D = array_1Dto2D(array_video_psd_theta)
    array_video_psd_alpha_2D = array_1Dto2D(array_video_psd_alpha)
    array_video_psd_beta_2D = array_1Dto2D(array_video_psd_beta)
    array_video_psd_gamma_2D = array_1Dto2D(array_video_psd_gamma)

    arrays_to_stack_de = [
        array_video_de_theta_2D,
        array_video_de_alpha_2D,
        array_video_de_beta_2D,
        array_video_de_gamma_2D,
    ]
    arrays_to_stack_psd = [
        array_video_psd_theta_2D,
        array_video_psd_alpha_2D,
        array_video_psd_beta_2D,
        array_video_psd_gamma_2D
    ]
    return arrays_to_stack_de, arrays_to_stack_psd

def process_video_data():
    all_video_list_de = [[] for _ in range(n_videos)]
    all_video_list_psd = [[] for _ in range(n_videos)]
    # 遍历所有受试者
    for subject_idx in range(n_subjects):
        print(f"Processing subject {subject_idx + 1}/{n_subjects}")
        subject_data = dreamer_data[0, subject_idx]

        # 获取EEG数据
        eeg_data = subject_data['EEG'][0, 0]
        stimuli = eeg_data['stimuli'][0, 0]

        # 遍历每个视频
        for video_index in range(n_videos):
            # 提取当前视频的EEG数据
            video_eeg_stimuli = stimuli[video_index][0]
            video_eeg_stimuli_ref = average_reference(video_eeg_stimuli)
            # video_eeg_stimuli_cleaned = remove_outliers_rolling(video_eeg_stimuli_ref)
            arrays_to_stack_de, arrays_to_stack_psd = process_data_in_one_video(video_eeg_stimuli_ref, video_index)
            if is_valid_for_stacking(arrays_to_stack_de) and is_valid_for_stacking(arrays_to_stack_psd):
                de_video_data_2d = np.stack(arrays_to_stack_de, axis=3)
                psd_video_data_2d = np.stack(arrays_to_stack_psd, axis=3)
                all_video_list_de[video_index].append(de_video_data_2d)
                all_video_list_psd[video_index].append(psd_video_data_2d)

        print()

    for video_index in range(n_videos):
        video_array_de = np.concatenate(all_video_list_de[video_index], axis=0)
        video_array_psd = np.concatenate(all_video_list_psd[video_index], axis=0)
        arousal_labels = np.repeat(config.DREAMER_video_arousal_labels[int(video_index)], video_array_de.shape[0])
        valence_labels = np.repeat(config.DREAMER_video_valence_labels[int(video_index)], video_array_psd.shape[0])
        processed_data_de = {
            'data': video_array_de,
            'arousal_labels': arousal_labels,
            'valence_labels': valence_labels
        }
        processed_data_psd = {
            'data': video_array_psd,
            'arousal_labels': arousal_labels,
            'valence_labels': valence_labels
        }
        result_dir = 'dataset/Dreamer/processed_data/'
        sio.savemat(result_dir + "DE_video" + str(video_index + 1).zfill(2), processed_data_de)
        print("Saved video {} processed video data DE.".format(video_index + 1))
        sio.savemat(result_dir + "PSD_video" + str(video_index + 1).zfill(2), processed_data_psd)
        print("Saved video {} processed video data PSD.".format(video_index + 1))

if __name__ == "__main__":
    process_video_data()