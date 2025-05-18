import math
import re
import os
import glob

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.integrate import simpson as simps
from scipy.signal import butter, lfilter, welch
from scipy.io import loadmat

import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import config

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    对数据应用巴特沃斯带通滤波器
    
    参数:
    data -- 要过滤的输入信号数据
    lowcut -- 带通滤波器的低频截止频率(Hz)
    highcut -- 带通滤波器的高频截止频率(Hz)
    fs -- 采样频率(Hz)
    order -- 滤波器阶数，默认为5阶
    
    返回:
    y -- 经过带通滤波处理后的信号
    """
    
    # 计算奈奎斯特频率
    nyq = 0.5 * fs
    
    # 归一化截止频率
    low = lowcut / nyq
    high = highcut / nyq
    
    # 设计巴特沃斯带通滤波器
    b, a = butter(order, [low, high], btype='band')
    
    # 应用滤波器
    y = lfilter(b, a, data)
    
    return y

# 读取文件
def read_file(file):
    data = sio.loadmat(file)
    video_data = data['video_data']
    return video_data

def compute_DE(signal):
    variance = np.var(signal, ddof=1)
    return math.log(2 * math.pi * math.e * variance) / 2

def compute_PSD(window_signal, low, high, fs=128):
    # 1. 计算功率谱密度 (PSD)
    # 对于0.5秒窗口(64个点)，使用适当的nperseg值
    # nperseg=64会给出单一窗口估计，nperseg=32会有更好的频率分辨率但更多方差
    freqs, psd = welch(window_signal, fs=fs, nperseg=64, 
                             scaling='density', average='mean')
    
    # 根据波段范围 (14-31Hz) 确定PSD中对应的频率索引
    # 虽然信号已经过滤波，但我们仍需要确定PSD中对应的频率索引
    idx_band = np.logical_and(freqs >= low, freqs <= high)
    
    # 3. 计算频率分辨率，即功率谱密度(PSD)中相邻频率点之间的间隔。
    freq_res = freqs[1] - freqs[0]
    
    # 4. 使用Simpson积分法计算β波段功率
    # 如果滤波非常精确，这基本上是整个PSD的积分
    beta_power = simps(psd[idx_band], dx=freq_res)
    
    return beta_power  # 返回单个数值

# 划分窗口
def segment_signal(signal, window_size, overlap):
    segments = []
    step = window_size - overlap
    for start in range(0, len(signal) - window_size + 1, step):
        end = start + window_size
        segments.append(signal[start:end])
    # return np.array(segments)  
    return segments

def data_1Dto2D(data, Y=8, X=9):
    '''将一个窗口的1D数据转换为2D数据'''
    data_2D = np.zeros([Y, X])
    data_2D[0] = (0, 0, data[0], 0, 0, 0, data[13], 0, 0)
    data_2D[1] = (data[1], 0, data[2], 0, 0, 0, data[11], 0, data[12])
    data_2D[2] = (0, data[3], 0, 0, 0, 0, 0, data[10], 0)
    data_2D[3] = (data[4], 0, 0, 0, 0, 0, 0, 0, data[9])
    data_2D[4] = (0, 0, 0, 0, 0, 0, 0, 0, 0)
    data_2D[5] = (data[5], 0, 0, 0, 0, 0, 0, 0, data[8])
    data_2D[6] = (0, 0, 0, 0, 0, 0, 0, 0, 0)
    data_2D[7] = (0, 0, 0, data[6], 0, data[7], 0, 0, 0)
    return data_2D

def array_1Dto2D(data_list):
    '''将一个numpy数组当中所有的窗口数据转换为2D数据'''
    list_2D = []

    for window_data in data_list:
        data_2D = data_1Dto2D(window_data)
        list_2D.append(data_2D)

    return np.array(list_2D)

def process_video_data(file):
    video_data = read_file(file)
    match = re.search(r'video_(\d+)\.mat', file)
    video_number = None
    if match:
        video_number = match.group(1)
        print("视频编号:", video_number)
    else:
        raise ValueError("未找到视频编号")
    frequency = 128  # 采样频率

    all_de_theta = []
    all_de_alpha = []
    all_de_beta = []
    all_de_gamma = []
    all_psd_theta = []
    all_psd_alpha = []
    all_psd_beta = []
    all_psd_gamma = []

    for channel in range(14):
        channel_data = video_data[channel]

        # 初始化用于DE和PSD数组的numpy数组
        de_theta = np.zeros(shape=[0], dtype=float)
        de_alpha = np.zeros(shape=[0], dtype=float)
        de_beta = np.zeros(shape=[0], dtype=float)
        de_gamma = np.zeros(shape=[0], dtype=float)
        psd_theta = np.zeros(shape=[0], dtype=float)
        psd_alpha = np.zeros(shape=[0], dtype=float)
        psd_beta = np.zeros(shape=[0], dtype=float)
        psd_gamma = np.zeros(shape=[0], dtype=float)

        # 划分窗口
        window_data_list = segment_signal(channel_data, config.window_size, config.overlap)

        for window_data in window_data_list:
            # 进行带通滤波
            theta_eeg = butter_bandpass_filter(window_data, 4, 8, frequency)
            alpha_eeg = butter_bandpass_filter(window_data, 8, 14, frequency)
            beta_eeg = butter_bandpass_filter(window_data, 14, 31, frequency)
            gamma_eeg = butter_bandpass_filter(window_data, 31, 45, frequency)

            # 计算DE和PSD
            de_theta = np.append(de_theta, compute_DE(theta_eeg))
            de_alpha = np.append(de_alpha, compute_DE(alpha_eeg))
            de_beta = np.append(de_beta, compute_DE(beta_eeg))
            de_gamma = np.append(de_gamma, compute_DE(gamma_eeg))

            psd_theta = np.append(psd_theta, compute_PSD(theta_eeg, 4, 8))
            psd_alpha = np.append(psd_alpha, compute_PSD(alpha_eeg, 8, 14))
            psd_beta = np.append(psd_beta, compute_PSD(beta_eeg, 14, 31))
            psd_gamma = np.append(psd_gamma, compute_PSD(gamma_eeg, 31, 45))

        all_de_theta.append(de_theta)
        all_de_alpha.append(de_alpha)
        all_de_beta.append(de_beta)
        all_de_gamma.append(de_gamma)
        all_psd_theta.append(psd_theta)
        all_psd_alpha.append(psd_alpha)
        all_psd_beta.append(psd_beta)
        all_psd_gamma.append(psd_gamma)

    array_de_theta = np.array(all_de_theta).T
    array_de_alpha = np.array(all_de_alpha).T
    array_de_beta = np.array(all_de_beta).T
    array_de_gamma = np.array(all_de_gamma).T
    array_psd_theta = np.array(all_psd_theta).T
    array_psd_alpha = np.array(all_psd_alpha).T
    array_psd_beta = np.array(all_psd_beta).T
    array_psd_gamma = np.array(all_psd_gamma).T

    array_de_theta_2D = array_1Dto2D(array_de_theta)
    array_de_alpha_2D = array_1Dto2D(array_de_alpha)
    array_de_beta_2D = array_1Dto2D(array_de_beta) 
    array_de_gamma_2D = array_1Dto2D(array_de_gamma)
    array_psd_theta_2D = array_1Dto2D(array_psd_theta)
    array_psd_alpha_2D = array_1Dto2D(array_psd_alpha)
    array_psd_beta_2D = array_1Dto2D(array_psd_beta)
    array_psd_gamma_2D = array_1Dto2D(array_psd_gamma)

    arrays_to_stack = [
        # array_de_theta_2D,
        # array_de_alpha_2D,
        # array_de_beta_2D,
        # array_de_gamma_2D,
        array_psd_theta_2D,
        array_psd_alpha_2D,
        array_psd_beta_2D,
        array_psd_gamma_2D
    ]

    # shape: window_num*8*9*8((DE+PSD)*4brainwave)
    video_data_2d = np.stack(arrays_to_stack, axis=3)
    arousal_labels = np.repeat(config.AMIGO_video_arousal_labels[int(video_number) - 1], video_data_2d.shape[0])
    valence_labels = np.repeat(config.AMIGO_video_valence_labels[int(video_number) - 1], video_data_2d.shape[0])

    processed_data = {
        'data': video_data_2d,
        'arousal_labels': arousal_labels,
        'valence_labels': valence_labels
    }
    result_dir = 'dataset/amigo/processed_data/'
    sio.savemat(result_dir + "PSD_video" + str(video_number).zfill(2), processed_data)
    print("Saved video {} processed video data.".format(video_number))
        
if __name__ == "__main__":
    directory_path = "dataset/amigo/videos"

    # 使用 glob 模块查找所有 .mat 文件
    mat_files = glob.glob(os.path.join(directory_path, "*.mat"))

    # 处理排序并替换路径中的反斜杠为正斜杠
    sorted_mat_files = sorted(
        mat_files,
        key=lambda x: int(x.split(os.sep)[-1].split('_')[1].split('.')[0])
    )

    # 替换反斜杠为正斜杠并打印
    formatted_mat_files = [path.replace('\\', '/') for path in sorted_mat_files]

    # 输出结果
    for mat_file in formatted_mat_files:
        print('Video file:', mat_file)
        process_video_data(mat_file)