import math
import re
import os
import glob
from unittest import result

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
from process_util import butter_bandpass_filter, compute_DE, compute_PSD, segment_signal, \
    array_1Dto2D, calculate_0_and_nan, is_valid_for_stacking

root_dir = "dataset/DEAP/raw_data/"
frequency = 128  # 采样频率

def process_data_in_one_video(signal_data, video_index):
    video_de_theta = []
    video_de_alpha = []
    video_de_beta = []
    video_de_gamma = []
    video_psd_theta = []
    video_psd_alpha = []
    video_psd_beta = []
    video_psd_gamma = []

    for channel_index in range(32):
        one_channel_data = signal_data[video_index][channel_index]
        
        nan_count, zero_count = calculate_0_and_nan(one_channel_data)
        if nan_count > 0 or zero_count > 0:
            print(f"视频 {video_index + 1} 通道 {channel_index + 1} 存在 {nan_count} 个 NaN 和 {zero_count} 个零值")
            continue    

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

    # arrays_to_stack_de = [
    #     array_video_de_theta_2D,
    #     array_video_de_alpha_2D,
    #     array_video_de_beta_2D,
    #     array_video_de_gamma_2D,
    # ]
    # arrays_to_stack_psd = [
    #     array_video_psd_theta_2D,
    #     array_video_psd_alpha_2D,
    #     array_video_psd_beta_2D,
    #     array_video_psd_gamma_2D
    # ]
    arrays_to_stack_de_theta = [
        array_video_de_theta_2D,
    ]
    arrays_to_stack_psd_theta = [
        array_video_psd_theta_2D,
    ]
    arrays_to_stack_de_alpha = [
        array_video_de_alpha_2D,
    ]
    arrays_to_stack_psd_alpha = [
        array_video_psd_alpha_2D,
    ]
    arrays_to_stack_de_beta = [
        array_video_de_beta_2D,
    ]
    arrays_to_stack_psd_beta = [
        array_video_psd_beta_2D,
    ]
    arrays_to_stack_de_gamma = [
        array_video_de_gamma_2D,
    ]
    arrays_to_stack_psd_gamma = [
        array_video_psd_gamma_2D,
    ]

    return arrays_to_stack_de_theta, arrays_to_stack_psd_theta, \
           arrays_to_stack_de_alpha, arrays_to_stack_psd_alpha, \
           arrays_to_stack_de_beta, arrays_to_stack_psd_beta, \
           arrays_to_stack_de_gamma, arrays_to_stack_psd_gamma
    # return arrays_to_stack_de, arrays_to_stack_psd

def data_processing_deap():
    all_video_list_de_theta = [[] for _ in range(40)]
    all_video_list_psd_theta = [[] for _ in range(40)]
    all_video_list_de_alpha = [[] for _ in range(40)]
    all_video_list_psd_alpha = [[] for _ in range(40)]
    all_video_list_de_beta = [[] for _ in range(40)]
    all_video_list_psd_beta = [[] for _ in range(40)]
    all_video_list_de_gamma = [[] for _ in range(40)]
    all_video_list_psd_gamma = [[] for _ in range(40)]

    for file in os.listdir(root_dir):
        print("processing: ", file, "......")
        file_path = os.path.join(root_dir, file)
        signal_data = sio.loadmat(file_path)['data'][:, :32, 384:] # 去掉前 3 秒基线数据，维度为 (40, 32, 7680)
        for video_index in range(40):
            arrays_to_stack_de_theta, arrays_to_stack_psd_theta, \
            arrays_to_stack_de_alpha, arrays_to_stack_psd_alpha, \
            arrays_to_stack_de_beta, arrays_to_stack_psd_beta, \
            arrays_to_stack_de_gamma, arrays_to_stack_psd_gamma = process_data_in_one_video(signal_data, video_index)

            all_video_list_de_theta[video_index].append(np.stack(arrays_to_stack_de_theta, axis=3))
            all_video_list_psd_theta[video_index].append(np.stack(arrays_to_stack_psd_theta, axis=3))
            all_video_list_de_alpha[video_index].append(np.stack(arrays_to_stack_de_alpha, axis=3))
            all_video_list_psd_alpha[video_index].append(np.stack(arrays_to_stack_psd_alpha, axis=3))
            all_video_list_de_beta[video_index].append(np.stack(arrays_to_stack_de_beta, axis=3))
            all_video_list_psd_beta[video_index].append(np.stack(arrays_to_stack_psd_beta, axis=3))
            all_video_list_de_gamma[video_index].append(np.stack(arrays_to_stack_de_gamma, axis=3))
            all_video_list_psd_gamma[video_index].append(np.stack(arrays_to_stack_psd_gamma, axis=3))

    for video_index in range(40):
        video_array_de_theta = np.concatenate(all_video_list_de_theta[video_index], axis=0)
        video_array_psd_theta = np.concatenate(all_video_list_psd_theta[video_index], axis=0)
        video_array_de_alpha = np.concatenate(all_video_list_de_alpha[video_index], axis=0)
        video_array_psd_alpha = np.concatenate(all_video_list_psd_alpha[video_index], axis=0)
        video_array_de_beta = np.concatenate(all_video_list_de_beta[video_index], axis=0)
        video_array_psd_beta = np.concatenate(all_video_list_psd_beta[video_index], axis=0)
        video_array_de_gamma = np.concatenate(all_video_list_de_gamma[video_index], axis=0)
        video_array_psd_gamma = np.concatenate(all_video_list_psd_gamma[video_index], axis=0)

        arousal_label = 1 if video_index + 1 in config.DEAP_half_arousal_high else 0
        valence_label = 1 if video_index + 1 in config.DEAP_half_valence_high else 0
        arousal_labels = np.repeat(arousal_label, video_array_de_theta.shape[0])
        valence_labels = np.repeat(valence_label, video_array_de_theta.shape[0])

        processed_data_de_theta = {
            'data': video_array_de_theta,
            'arousal_labels': arousal_labels,
            'valence_labels': valence_labels
        }
        processed_data_psd_theta = {
            'data': video_array_psd_theta,
            'arousal_labels': arousal_labels,
            'valence_labels': valence_labels
        }
        processed_data_de_alpha = {
            'data': video_array_de_alpha,
            'arousal_labels': arousal_labels,
            'valence_labels': valence_labels
        }
        processed_data_psd_alpha = {
            'data': video_array_psd_alpha,
            'arousal_labels': arousal_labels,
            'valence_labels': valence_labels
        }
        processed_data_de_beta = {
            'data': video_array_de_beta,
            'arousal_labels': arousal_labels,
            'valence_labels': valence_labels
        }
        processed_data_psd_beta = {
            'data': video_array_psd_beta,
            'arousal_labels': arousal_labels,
            'valence_labels': valence_labels
        }
        processed_data_de_gamma = {
            'data': video_array_de_gamma,
            'arousal_labels': arousal_labels,
            'valence_labels': valence_labels
        }
        processed_data_psd_gamma = {
            'data': video_array_psd_gamma,
            'arousal_labels': arousal_labels,
            'valence_labels': valence_labels
        }

        result_dir_theta = 'dataset/DEAP/processed_data_theta/'
        result_dir_alpha = 'dataset/DEAP/processed_data_alpha/'
        result_dir_beta = 'dataset/DEAP/processed_data_beta/'
        result_dir_gamma = 'dataset/DEAP/processed_data_gamma/'
        if not os.path.exists(result_dir_theta):
            os.makedirs(result_dir_theta)
        if not os.path.exists(result_dir_alpha):
            os.makedirs(result_dir_alpha)
        if not os.path.exists(result_dir_beta):
            os.makedirs(result_dir_beta)
        if not os.path.exists(result_dir_gamma):
            os.makedirs(result_dir_gamma)
        sio.savemat(result_dir_theta + "DE_video" + str(video_index + 1).zfill(2) + ".mat", processed_data_de_theta)
        print("Saved video {} processed video data DE.".format(video_index + 1))
        sio.savemat(result_dir_theta + "PSD_video" + str(video_index + 1).zfill(2) + ".mat", processed_data_psd_theta)
        print("Saved video {} processed video data PSD.".format(video_index + 1))
        sio.savemat(result_dir_alpha + "DE_video" + str(video_index + 1).zfill(2) + ".mat", processed_data_de_alpha)
        print("Saved video {} processed video data DE.".format(video_index + 1))
        sio.savemat(result_dir_alpha + "PSD_video" + str(video_index + 1).zfill(2) + ".mat", processed_data_psd_alpha)
        print("Saved video {} processed video data PSD.".format(video_index + 1))
        sio.savemat(result_dir_beta + "DE_video" + str(video_index + 1).zfill(2) + ".mat", processed_data_de_beta)
        print("Saved video {} processed video data DE.".format(video_index + 1))
        sio.savemat(result_dir_beta + "PSD_video" + str(video_index + 1).zfill(2) + ".mat", processed_data_psd_beta)
        print("Saved video {} processed video data PSD.".format(video_index + 1))
        sio.savemat(result_dir_gamma + "DE_video" + str(video_index + 1).zfill(2) + ".mat", processed_data_de_gamma)
        print("Saved video {} processed video data DE.".format(video_index + 1))
        sio.savemat(result_dir_gamma + "PSD_video" + str(video_index + 1).zfill(2) + ".mat", processed_data_psd_gamma)
        print("Saved video {} processed video data PSD.".format(video_index + 1))

if __name__ == "__main__":
    data_processing_deap()
