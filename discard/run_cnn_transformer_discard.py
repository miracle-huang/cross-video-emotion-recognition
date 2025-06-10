import os
import torch
import random
import time
from tqdm import tqdm
from openpyxl import Workbook
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from models.cnn_transformer import Transformer
from data_process.discard.raw_window_data_loader import RawWindowDataLoader
from utils import set_global_random_seed
import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class SignalDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        # 假如 y 是 one-hot，转成整数
        if len(y.shape) == 2:
            self.y = torch.tensor(y.argmax(axis=1), dtype=torch.long)
        else:
            self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx].unsqueeze(0), self.y[idx]
    
class EarlyStopping:
    def __init__(self, patience=7, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def run_cnn_transformer(x_train, y_train, x_val, y_val, x_test, y_test):
    train_dataset = SignalDataset(x_train, y_train)
    val_dataset = SignalDataset(x_val, y_val)
    test_dataset = SignalDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    model = Transformer().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    early_stopping = EarlyStopping(patience=7, min_delta=1e-4)

    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(config.epoch):
        print(f"Start Epoch {epoch+1}/{config.epoch}")
        # --- Training ---
        model.train()
        total_train_loss = 0
         # 用tqdm包装train_loader显示进度条
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epoch}", unit="batch") as t_bar:
            for X_batch, y_batch in t_bar:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                optimizer.zero_grad()
                output = model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item() * X_batch.size(0)
        avg_train_loss = total_train_loss / len(train_dataset)

        # --- Validation ---
        model.eval()
        total_val_loss = 0
        val_preds, val_targets = [], []
        with torch.no_grad():
            with tqdm(val_loader, desc="Validation", unit="batch") as t_val:
                for X_batch, y_batch in t_val:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    output = model(X_batch)
                    loss = criterion(output, y_batch)
                    total_val_loss += loss.item() * X_batch.size(0)
                    preds = torch.argmax(output, 1).cpu().numpy()
                    targets = y_batch.cpu().numpy()
                    val_preds.extend(preds)
                    val_targets.extend(targets)
        avg_val_loss = total_val_loss / len(val_dataset)
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average='macro')

        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

        # --- Early Stopping Check ---
        early_stopping(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()  # Save best model

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # --- Load best model for test ---
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # --- Test ---
    model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            output = model(X_batch)
            preds = torch.argmax(output, 1).cpu().numpy()
            targets = y_batch.cpu().numpy()
            test_preds.extend(preds)
            test_targets.extend(targets)
    test_acc = accuracy_score(test_targets, test_preds)
    test_f1 = f1_score(test_targets, test_preds, average='macro')
    test_cm = confusion_matrix(test_targets, test_preds)

    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print("Confusion Matrix:\n", test_cm)

    return test_acc, test_f1, test_cm

def save_result_to_xlsx(random_seed, train_video_list, test_video_list):
    set_global_random_seed(random_seed)
    save_folder = f"result_folder/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    xlsx_name = os.path.join(save_folder, f"cnn_transformer_random_seed{random_seed}_batchsize{config.batch_size}.xlsx")
    wb = Workbook()
    ws = wb.active
    ws.append(["Content", "Arousal", "Valence", "f1_a", "f1_v", "conf_matrix_a", "conf_matrix_v"])
    for test_video in test_video_list:
        train_content_list_leave = [x for x in train_video_list if x != test_video]
        print(f"Leave one video out: {test_video}, train content list: {train_content_list_leave}")
        dataloader = RawWindowDataLoader(dataset_dir=config.AMIGO_raw_window_path, random_seed=42)
        train_data, val_data, test_data = dataloader.set_train_val_test(train_content_list_leave, [test_video])

        x_train, y_a_train, y_v_train = train_data['x_train'], train_data['y_a_train'], train_data['y_v_train']
        x_val, y_a_val, y_v_val = val_data['x_val'], val_data['y_a_val'], val_data['y_v_val']
        x_test, y_a_test, y_v_test = test_data['x_test'], test_data['y_a_test'], test_data['y_v_test']

        print("Arousal")
        acc_a, f1_a, cm_a = run_cnn_transformer(x_train, y_a_train, x_val, y_a_val, x_test, y_a_test)
        print("Valence")
        acc_v, f1_v, cm_v = run_cnn_transformer(x_train, y_v_train, x_val, y_v_val, x_test, y_v_test)

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
    save_result_to_xlsx(config.random_seed, config.AMIGO_all_videos_list, config.AMIGO_all_videos_list)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"代码运行时间: {elapsed_time} 秒")