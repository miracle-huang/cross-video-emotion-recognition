random_seed = 42 # 随机种子

window_size = 64 # 窗口大小为0.5秒
window_size_10 = 1280 # 窗口大小为10秒
window_size_20 = 2560 # 窗口大小为20秒
overlap = 0
DREAMER_channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

DREAMER_video_valence_ratings = [3.17, 3.04, 4.57, 2.04, 3.22, 2.70, 4.52, 1.35, 1.39, 2.17, 3.96, 3.96, 4.39, 2.35, 2.48, 3.65, 1.52, 2.65]
DREAMER_video_arousal_ratings = [2.26, 3.00, 3.83, 4.26, 3.70, 3.83, 3.17, 3.96, 3.00, 3.30, 1.96, 2.61, 3.70, 2.22, 3.09, 3.35, 3.00, 3.91]
DREAMER_video_valence_labels = [1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0]
DREAMER_video_arousal_labels = [0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1]
DREAMER_all_videos_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
DREAMER_half_valence_low = [4, 8, 9, 17]
DREAMER_half_valence_high = [3, 7, 12, 13]
DREAMER_half_arousal_low = [1, 11, 12, 14]
DREAMER_half_arousal_high = [3, 4, 8, 18]

AMIGO_video_valence_ratings = [6.99, 7.58, 3.74, 7.14, 3.88, 3.56, 3.46, 3.55, 3.3, 2.91, 3.15, 5.81, 6.75, 6.93, 7.64, 6.34]
AMIGO_video_arousal_ratings = [4.08, 4.23, 4.12, 3.84, 4.42, 5.15, 5.01, 6.79, 6.0, 5.59, 6.52, 5.64, 6.05, 4.38, 5.5, 5.53]
AMIGO_video_valence_labels = [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
AMIGO_video_arousal_labels = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1]
AMIGO_all_videos_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
AMIGO_half_valence_low = [7, 9, 10, 11]
AMIGO_half_valence_high = [1, 2, 4, 15]
AMIGO_half_arousal_low = [1, 2, 3, 4]
AMIGO_half_arousal_high = [8, 9, 11, 13]

# valence and arousal content lists in DEAP dataset
# A complete list of valence and arousal order, from low to high
DEAP_all_videos_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
DEAP_valence_rating = [38, 37, 35, 23, 30, 39, 29, 21, 31, 32, 33, 34, 36, 24, 28, 25, 22, 10, 40, 26, 16, 27, 17, 7, 15, 5, 12, 2, 20, 6, 13, 1, 4, 8, 19, 9, 18, 14, 11, 3]
DEAP_arousal_rating = [26, 22, 28, 16, 15, 12, 17, 23, 29, 21, 27, 13, 25, 18, 24, 40, 33, 20, 14, 30, 11, 8, 31, 6, 39, 10, 19, 1, 7, 37, 35, 9, 38, 34, 36, 4, 3, 5, 2, 32]

# high 20 and low 20
DEAP_half_valence_low = [38, 37, 35, 23, 30, 39, 29, 21, 31, 32, 33, 34, 36, 24, 28, 25, 22, 10, 40, 26]
DEAP_half_valence_high = [16, 27, 17, 7, 15, 5, 12, 2, 20, 6, 13, 1, 4, 8, 19, 9, 18, 14, 11, 3]
DEAP_half_arousal_low = [26, 22, 28, 16, 15, 12, 17, 23, 29, 21, 27, 13, 25, 18, 24, 40, 33, 20, 14, 30]
DEAP_half_arousal_high = [11, 8, 31, 6, 39, 10, 19, 1, 7, 37, 35, 9, 38, 34, 36, 4, 3, 5, 2, 32]

# high 10 and low 10
DEAP_ten_valence_low = [38, 37, 35, 23, 30, 39, 29, 21, 31, 32]
DEAP_ten_valence_high = [13, 1, 4, 8, 19, 9, 18, 14, 11, 3]
DEAP_ten_arousal_low = [26, 22, 28, 16, 15, 12, 17, 23, 29, 21]
DEAP_ten_arousal_high = [35, 9, 38, 34, 36, 4, 3, 5, 2, 32]

# dataset path
DEAP_dataset_path = "D:/huangzhiying/cross-video-emotion-recognition/cross-video-emotion-recognition/dataset/DEAP/data_2d/with_base_0.5/"
AMIGO_dataset_path = "D:/huangzhiying/cross-video-emotion-recognition/cross-video-emotion-recognition/dataset/amigo/processed_data/"
DREAMER_dataset_path = "D:/huangzhiying/cross-video-emotion-recognition/cross-video-emotion-recognition/dataset/DREAMER/processed_data/"
# AMIGO_raw_window_path = "dataset/amigo/raw_window_data_10s/"
AMIGO_cnn_transformer_dataset_path = "dataset/amigo/cnn_transformer/window_10s"

# model parameters
epoch = 100
batch_size = 64
filters = [64, 128, 256]
kernel_size_list = [3, 3, 3, 1]
dropout_rate = 0.2
learning_rate = 0.001

# channel names mapping
amigo_channel_mapping = {
    0: 'AF3',
    1: 'F7',
    2: 'F3',
    3: 'FC5',
    4: 'T7',
    5: 'P7',
    6: 'O1',
    7: 'O2',
    8: 'P8',
    9: 'T8',
    10: 'FC6',
    11: 'F4',
    12: 'F8',
    13: 'AF4'
}