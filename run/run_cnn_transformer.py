import os
import torch
import random
import time
import numpy as np
from tqdm import tqdm
from openpyxl import Workbook
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from utils import set_global_random_seed
from trainers.cnn_transformer_trainer import train_cnn_transformer
from dataloader.cnn_transformer_data_loader import CNNTransformerDataLoader
import config

def save_result_to_xlsx(random_seed, train_video_list, test_video_list, dataset_dir, experiment_name="all_videos"):
    set_global_random_seed(random_seed)
    save_folder = f"result_folder/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    xlsx_name = os.path.join(save_folder, f"{experiment_name}_cnn_transformer_random_seed{random_seed}_batchsize{config.batch_size}.xlsx")
    wb = Workbook()
    ws = wb.active
    ws.append(["Content", "Arousal", "Valence", "f1_a", "f1_v", "conf_matrix_a", "conf_matrix_v"])

    for test_video in test_video_list:
        train_content_list_leave = [x for x in train_video_list if x != test_video]
        print(f"Leave one video out: {test_video}, train content list: {train_content_list_leave}")

        dataloader = CNNTransformerDataLoader(dataset_dir, random_seed=random_seed)
        train_data, val_data, test_data = dataloader.set_train_val_test(train_content_list_leave, [test_video])

        x_train, y_a_train, y_v_train = train_data['x_train'], train_data['y_a_train'], train_data['y_v_train']
        x_val, y_a_val, y_v_val = val_data['x_val'], val_data['y_a_val'], val_data['y_v_val']
        x_test, y_a_test, y_v_test = test_data['x_test'], test_data['y_a_test'], test_data['y_v_test']

        print("Arousal")
        acc_a, f1_a, cm_a = train_cnn_transformer(x_train, y_a_train, x_val, y_a_val, x_test, y_a_test, 
                                                  test_video_name=f'{test_video}_arousal', experiment_name=experiment_name)
        print("Valence")
        acc_v, f1_v, cm_v = train_cnn_transformer(x_train, y_v_train, x_val, y_v_val, x_test, y_v_test, 
                                                  test_video_name=f'{test_video}_valence', experiment_name=experiment_name)

        ws.append([
            f"{test_video}",
            f"{acc_a}",
            f"{acc_v}",
            f"{f1_a}",
            f"{f1_v}",
            f"{cm_a}",
            f"{cm_v}",
        ])
        wb.save(xlsx_name)

if __name__ == "__main__":
    start_time = time.time()
    DEAP_top_bottom_train_arousal_list = config.DEAP_ten_arousal_high + config.DEAP_ten_arousal_low
    DEAP_top_bottom_train_valence_list = config.DEAP_ten_valence_high + config.DEAP_ten_valence_low

    DREAMER_top_bottom_train_arousal_list = config.DREAMER_half_arousal_high + config.DREAMER_half_arousal_low
    DREAMER_top_bottom_train_valence_list = config.DREAMER_half_valence_high + config.DREAMER_half_valence_low

    AMIGOS_top_bottom_train_arousal_list = config.AMIGO_half_arousal_high + config.AMIGO_half_arousal_low
    AMIGOS_top_bottom_train_valence_list = config.AMIGO_half_valence_high + config.AMIGO_half_valence_low

    # save_result_to_xlsx(config.random_seed, config.DEAP_all_videos_list,
    #                     config.DEAP_all_videos_list, config.DEAP_cnn_transformer_dataset_path, experiment_name="DEAP_all_videos")
    # save_result_to_xlsx(config.random_seed, DEAP_top_bottom_train_arousal_list,
    #                     config.DEAP_all_videos_list, config.DEAP_cnn_transformer_dataset_path, experiment_name="DEAP_top_bottom_arousal")
    # save_result_to_xlsx(config.random_seed, DEAP_top_bottom_train_valence_list,
    #                     config.DEAP_all_videos_list, config.DEAP_cnn_transformer_dataset_path, experiment_name="DEAP_top_bottom_valence")

    # save_result_to_xlsx(config.random_seed, config.DREAMER_all_videos_list,
    #                     config.DREAMER_all_videos_list, config.DREAMER_cnn_transformer_dataset_path, experiment_name="DREAMER_all_videos")
    # save_result_to_xlsx(config.random_seed, DREAMER_top_bottom_train_arousal_list,
    #                     config.DREAMER_all_videos_list, config.DREAMER_cnn_transformer_dataset_path, experiment_name="DREAMER_top_bottom_arousal")
    # save_result_to_xlsx(config.random_seed, DREAMER_top_bottom_train_valence_list,
    #                     config.DREAMER_all_videos_list, config.DREAMER_cnn_transformer_dataset_path, experiment_name="DREAMER_top_bottom_valence")
    
    # save_result_to_xlsx(config.random_seed, AMIGOS_top_bottom_train_arousal_list,
    #                     config.AMIGO_all_videos_list, config.AMIGO_cnn_transformer_dataset_path, experiment_name="AMIGOS_top_bottom_arousal")
    # save_result_to_xlsx(config.random_seed, AMIGOS_top_bottom_train_valence_list,
    #                     config.AMIGO_all_videos_list, config.AMIGO_cnn_transformer_dataset_path, experiment_name="AMIGOS_top_bottom_valence")
    
    save_result_to_xlsx(config.random_seed, config.AMIGO_all_videos_list,
                        config.AMIGO_all_videos_list, config.AMIGO_cnn_transformer_dataset_path, experiment_name="AMIGO_all_videos_reverse")
    # save_result_to_xlsx(config.random_seed, config.DREAMER_all_videos_list,
    #                     config.DREAMER_all_videos_list, config.DREAMER_cnn_transformer_dataset_path, experiment_name="DREAMER_all_videos_reverse")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"代码运行时间: {elapsed_time} 秒")