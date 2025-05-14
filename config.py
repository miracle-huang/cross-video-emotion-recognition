random_seed = 42 # 随机种子

window_size = 64 # 窗口大小为0.5秒
overlap = 0
DREAMER_channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

DREAMER_video_valence_ratings = [3.17, 3.04, 4.57, 2.04, 3.22, 2.70, 4.52, 1.35, 1.39, 2.17, 3.96, 3.96, 4.39, 2.35, 2.48, 3.65, 1.52, 2.65]
DREAMER_video_arousal_ratings = [2.26, 3.00, 3.83, 4.26, 3.70, 3.83, 3.17, 3.96, 3.00, 3.30, 1.96, 2.61, 3.70, 2.22, 3.09, 3.35, 3.00, 3.91]
DREAMER_video_valence_labels = [1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0]
DREAMER_video_arousal_labels = [0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1]

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

# DEAP dataset path
DEAP_dataset_path = "D:/huangzhiying/cross-video-emotion-recognition/cross-video-emotion-recognition/dataset/DEAP/data_2d/psd/"

# model parameters
epoch = 100
batch_size = 256
filters = [64, 128, 256]
kernel_size_list = [3, 3, 3, 1]
dropout_rate = 0.2
learning_rate = 0.001