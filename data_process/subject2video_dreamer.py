import numpy as np
import scipy.io as sio
from collections import defaultdict

dreamer_dataset = sio.loadmat('cross-video-emotion-recognition/dataset/Dreamer/DREAMER.mat')    

dreamer_struct = dreamer_dataset['DREAMER']
dreamer_data = dreamer_struct[0, 0]['Data']

# 获取受试者数量和视频数量
n_subjects = dreamer_data.shape[1]
# 假设所有受试者观看了相同数量的视频
n_videos = 18  # 有18个视频

def subject2video():
    # 创建按视频组织的数据结构
    video_data = defaultdict(list)
    baseline_data = defaultdict(list)

    # 初始化dict，每个视频对应一个numpy数组
    for video_index in range(1, n_videos + 1):
        video_data[video_index] = np.empty((0, 14))
        baseline_data[video_index] = np.empty((0, 14))

    # 遍历所有受试者
    for subject_idx in range(n_subjects):
        subject_data = dreamer_data[0, subject_idx]
        
        # 获取EEG数据
        eeg_data = subject_data['EEG'][0, 0]
        stimuli = eeg_data['stimuli'][0, 0]
        baseline = eeg_data['baseline'][0, 0]
        
        # 遍历每个视频
        for video_index in range(n_videos):
            # 提取当前视频的EEG数据
            video_eeg_stimuli = stimuli[video_index][0]
            video_eeg_baseline = baseline[video_index][0]

            # 将EEG数据和评分添加到相应的视频数据结构中
            video_data[video_index + 1] = np.concatenate((video_data[video_index + 1], video_eeg_stimuli), axis=0)
            baseline_data[video_index + 1] = np.concatenate((baseline_data[video_index + 1], video_eeg_baseline), axis=0)

    # 将数据保存到mat文件中,每个视频一个文件
    for video_index in range(1, n_videos + 1):
        data = {
            'EEG': video_data[video_index],
            'Baseline': baseline_data[video_index],
        }
        sio.savemat('cross-video-emotion-recognition/dataset/Dreamer/videos/video_{}.mat'.format(video_index), data)
        print("Saved video {} data to file.".format(video_index))

if __name__ == "__main__":
    subject2video()