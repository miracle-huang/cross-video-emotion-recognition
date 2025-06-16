import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import os
import numpy as np
import tensorflow as tf
from openpyxl import Workbook
import random
import time
from collections import Counter

import config
from dataloader.leave_one_video_out_2d_cnn_loader import LeaveOneVideoOutLoader
from models.cnn_2d_model import CnnTwoDimensionModel
from trainers.cnn_2d_trainer import CnnTwoDimensionTrainer
from utils import set_global_random_seed

# 禁用多线程和多进程
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'

# 确保设置生效
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# 打印当前设置的线程数
inter_op_threads = tf.config.threading.get_inter_op_parallelism_threads()
intra_op_threads = tf.config.threading.get_intra_op_parallelism_threads()
print("Inter-op threads:", inter_op_threads)
print("Intra-op threads:", intra_op_threads)

def leave_one_content_out(dataloader):
    '''
    random_seed - Random seed
    train_content_loader - The class used to load training data
    test_content_loader - The class used to load testing data
    '''
    train_data, val_data, test_data = dataloader.load_data(dataset_name='DEAP')
    input_shape = train_data['x_train'].shape[-3:]

    cnn_model_creater = CnnTwoDimensionModel(
        config.filters, config.kernel_size_list, config.dropout_rate, config.learning_rate, input_shape)
    cnn_model_a = cnn_model_creater.create_2d_cnn_model()
    cnn_model_v = cnn_model_creater.create_2d_cnn_model()

    trainer_a = CnnTwoDimensionTrainer(train_data, val_data, test_data, cnn_model_a, "arousal")
    acc_output_a, conf_matrix_a, f1_a = trainer_a.train_model()
    trainer_v = CnnTwoDimensionTrainer(train_data, val_data, test_data, cnn_model_v, "valence")
    acc_output_v, conf_matrix_v, f1_v = trainer_v.train_model()

    return acc_output_a, acc_output_v, f1_a, f1_v, conf_matrix_a, conf_matrix_v

def save_result_to_xlsx(
        random_seed, train_video_list, test_video_list):
    set_global_random_seed(random_seed)
    save_folder = f"result_folder/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    xlsx_name = os.path.join(save_folder, f"random_seed{random_seed}_batchsize{config.batch_size}.xlsx")
    wb = Workbook()
    ws = wb.active
    ws.append(["Content", "Arousal", "Valence", "f1_a", "f1_v", "conf_matrix_a", "conf_matrix_v"])
    for test_video in test_video_list:
        train_content_list_leave = [x for x in train_video_list if x != test_video]
        print(f"Leave one video out: {test_video}, train content list: {train_content_list_leave}")
        dataloader = LeaveOneVideoOutLoader(config.DEAP_dataset_path, random_seed, train_content_list_leave, [test_video])
        acc_output_a, acc_output_v, f1_a, f1_v, conf_matrix_a, conf_matrix_v = leave_one_content_out(dataloader)
        ws.append([
            f"{test_video}",
            f"{acc_output_a}",
            f"{acc_output_v}",
            f"{f1_a}",
            f"{f1_v}",
            f"{conf_matrix_a}",
            f"{conf_matrix_v}",
        ])
    wb.save(xlsx_name)

if __name__ == '__main__':
    start_time = time.time()
    train_video_list = config.DEAP_ten_valence_high + config.DEAP_ten_valence_low
    save_result_to_xlsx(
        config.random_seed, train_video_list, config.DEAP_all_videos_list)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"代码运行时间: {elapsed_time} 秒")