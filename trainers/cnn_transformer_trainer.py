import torch
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from models.cnn_transformer import Transformer
from utils import set_global_random_seed
import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class EEGDataset(Dataset):
    def __init__(self, data_x, data_y):
        """
        data_x: shape (T, C, W) - NumPy array or torch tensor
        data_y: shape (T,) or (T, 1)
        """
        if isinstance(data_x, np.ndarray):
            self.data_x = torch.tensor(data_x, dtype=torch.float32)
        else:
            self.data_x = data_x.float()

        if isinstance(data_y, np.ndarray):
            self.data_y = torch.tensor(data_y, dtype=torch.long)
        else:
            self.data_y = data_y.long()

    def __len__(self):
        return self.data_x.shape[0]  # time_steps

    def __getitem__(self, idx):
        x = self.data_x[idx]  # shape: (C, W)
        y = self.data_y[idx]  # int
        return x, y
    
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

def train_cnn_transformer(x_train, y_train, x_val, y_val, x_test, y_test):
    train_dataset = EEGDataset(x_train, y_train)
    val_dataset = EEGDataset(x_val, y_val)
    test_dataset = EEGDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    model = Transformer().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    early_stopping = EarlyStopping(patience=7, min_delta=1e-4)

    best_val_loss = float('inf')
    best_model_state = None

    # 创建 tensorboard writer
    result_dir = getattr(config, "result_dir", "./results")
    log_dir = os.path.join(result_dir, "tensorboard_logs")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

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