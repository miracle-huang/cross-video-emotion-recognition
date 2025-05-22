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

def calculate_0_and_nan(arr):
    '''计算数组中NaN和0的数量'''
    nan_count = np.isnan(arr).sum()
    non_nan_mask = ~np.isnan(arr)
    zero_count = (arr[non_nan_mask] == 0).sum()

    return nan_count, zero_count

# 判断数组是否为空以及形状是否一致
def is_valid_for_stacking(arrays):
    # 检查是否有空数组
    if not all(arr.size > 0 for arr in arrays):
        return False
    
    # 检查所有数组的形状是否一致
    shapes = [arr.shape for arr in arrays]
    if len(set(shapes)) > 1:
        return False
    
    return True