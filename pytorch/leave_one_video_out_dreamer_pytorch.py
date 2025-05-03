import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import scipy.io as sio
import os
import pandas as pd
from sklearn.utils import shuffle
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau

import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

# 导入提供的模型
from pytorch.cnn_2d_model_pytorch import CnnTwoDimensionModel

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 检查是否有可用的GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 定义模型参数
filters = [64, 128, 256]  # 滤波器数量
kernel_size_list = [(3, 3), (3, 3), (3, 3), (1, 1)]  # 卷积核大小
dropout_rate = 0.2
learning_rate = 0.05
batch_size = 32
epochs = 100

# 早停参数
patience = 20  # 如果验证集性能在这么多个epoch内没有提升，则停止训练
min_delta = 0.001  # 最小改善阈值

# 将标签转换为独热编码
def to_one_hot(labels):
    """
    手动将0/1标签转换为独热编码
    
    参数:
        labels: 形状为(n_samples,)的标签数组
        
    返回:
        独热编码的标签，形状为(n_samples, 2)
    """
    n_samples = len(labels)
    one_hot = np.zeros((n_samples, 2))
    
    for i, label in enumerate(labels):
        one_hot[i, int(label)] = 1
        
    return one_hot

# 加载数据函数
def load_data(data_dir, label_type):
    """
    加载所有视频数据，按照视频文件名的后缀数字顺序排序
    
    参数:
        data_dir: 数据文件夹路径
        label_type: 'arousal' 或 'valence'
        
    返回:
        video_data: 包含所有视频数据的列表
        video_labels: 包含所有视频标签的列表
        video_names: 包含所有视频名称的列表
    """
    video_data = []
    video_labels = []
    video_names = []
    
    # 获取所有.mat文件
    mat_files = glob.glob(os.path.join(data_dir, "*.mat"))
    
    # 按照视频文件名后缀的数字进行排序
    def extract_number(filename):
        # 从文件名中提取数字部分（例如从"video_1.mat"中提取1）
        base_name = os.path.basename(filename)
        try:
            # 假设格式为"video_X.mat"，其中X是数字
            return int(base_name.split('_')[1].split('.')[0])
        except (IndexError, ValueError):
            # 如果提取失败，返回一个很大的数，确保排在最后
            return float('inf')
    
    # 按照提取的数字排序
    mat_files.sort(key=extract_number)
    
    for mat_file in mat_files:
            # 加载.mat文件
            mat_data = sio.loadmat(mat_file)
            
            # 提取数据和标签
            data = mat_data['video_data']  # 假设的键名，可能需要根据实际情况调整
            
            # 提取对应的标签并转换为独热编码
            if label_type == 'arousal':
                labels = mat_data['arousal_labels'].flatten()  # 转为一维数组
            else:  # valence
                labels = mat_data['valence_labels'].flatten()  # 转为一维数组
                
            # 转换为独热编码
            one_hot_labels = to_one_hot(labels)
            
            video_name = os.path.basename(mat_file).split('.')[0]
            
            video_data.append(data)
            video_labels.append(one_hot_labels)
            video_names.append(video_name)
    
    return video_data, video_labels, video_names

# 添加保存训练记录的函数
def save_training_history(history, video_name, emotion_dim):
    """
    将训练历史记录保存到Excel文件
    
    参数:
        history: 包含训练历史的字典
        video_name: 视频名称
        emotion_dim: 情感维度
    """
    # 创建保存训练记录的目录
    save_dir = "training_records"
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建DataFrame
    df = pd.DataFrame(history)
    
    # 保存到Excel
    file_path = os.path.join(save_dir, f"{video_name}_{emotion_dim}_training_history.xlsx")
    df.to_excel(file_path, index=False)
    
    print(f"训练记录已保存到 {file_path}")

# 早停类
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, verbose=True):
        """
        早停机制
        
        参数:
            patience: 容忍验证集性能不提升的轮数
            min_delta: 最小改善阈值
            verbose: 是否打印早停信息
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_acc):
        score = val_acc
        
        if self.best_score is None:
            self.best_score = score
        elif score <= self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.early_stop

# 训练函数
def train_model(model, train_loader, val_loader, epochs, video_name, emotion_dim):
    """
    训练模型
    
    参数:
        model: 模型实例
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        epochs: 训练轮数
    
    返回:
        训练好的模型
    """
    model.to(device)
    model.compile()  # 使用默认优化器
    criterion = nn.CrossEntropyLoss()

    # 添加学习率调度器
    scheduler = ReduceLROnPlateau(model.optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    # 添加早停机制
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
    
    best_val_acc = 0.0
    best_model_state = None

    # 用于记录每个epoch的训练过程
    train_history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rate': []
    }
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 将独热编码转换为类别索引
            targets_idx = torch.argmax(targets, dim=1)
            
            # 前向传播
            model.optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets_idx)
            
            # 反向传播和优化
            loss.backward()
            model.optimizer.step()
            
            # 统计
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets_idx).sum().item()
        
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                inputs, targets = inputs.to(device), targets.to(device)
                
                # 将独热编码转换为类别索引
                targets_idx = torch.argmax(targets, dim=1)
                
                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, targets_idx)
                
                # 统计
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets_idx).sum().item()
        
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        # 获取当前学习率
        current_lr = model.optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} - "
              f"LR: {current_lr:.6f}")
        
        # 记录当前epoch的训练过程
        train_history['epoch'].append(epoch+1)
        train_history['train_loss'].append(train_loss)
        train_history['train_acc'].append(train_acc)
        train_history['val_loss'].append(val_loss)
        train_history['val_acc'].append(val_acc)
        train_history['learning_rate'].append(current_lr)

        # 更新学习率调度器
        scheduler.step(val_acc)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f"找到新的最佳模型，验证准确率: {best_val_acc:.4f}")

        # 检查是否应该早停
        if early_stopping(val_acc):
            print(f"早停触发，在 {epoch+1} 个epoch后停止训练")
            break
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"已加载最佳模型，最佳验证准确率: {best_val_acc:.4f}")

    # 保存训练记录到Excel
    save_training_history(train_history, video_name, emotion_dim)
    
    return model

# 测试函数
def test_model(model, test_loader):
    """
    测试模型
    
    参数:
        model: 训练好的模型
        test_loader: 测试数据加载器
    
    返回:
        准确率
    """
    model.eval()
    correct = 0
    total = 0
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 将独热编码转换为类别索引
            targets_idx = torch.argmax(targets, dim=1)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            total += targets.size(0)
            correct += (predicted == targets_idx).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets_idx.cpu().numpy())
    
    accuracy = correct / total
    return accuracy, all_preds, all_targets

# 更新并保存结果到Excel
def update_and_save_results(results, results_file):
    """
    更新结果并保存到Excel文件
    
    参数:
        results: 包含结果的字典
        results_file: 保存结果的文件路径
    """
    df = pd.DataFrame(results)
    
    # 计算每个视频的指标
    for i in range(len(df)):
        # Arousal
        if df.loc[i, 'arousal_TP'] is not None:
            arousal_tp = df.loc[i, 'arousal_TP']
            arousal_fp = df.loc[i, 'arousal_FP']
            arousal_tn = df.loc[i, 'arousal_TN']
            arousal_fn = df.loc[i, 'arousal_FN']
            
            df.loc[i, 'arousal_precision'] = arousal_tp / (arousal_tp + arousal_fp) if (arousal_tp + arousal_fp) > 0 else 0
            df.loc[i, 'arousal_recall'] = arousal_tp / (arousal_tp + arousal_fn) if (arousal_tp + arousal_fn) > 0 else 0
            df.loc[i, 'arousal_f1'] = 2 * (df.loc[i, 'arousal_precision'] * df.loc[i, 'arousal_recall']) / (df.loc[i, 'arousal_precision'] + df.loc[i, 'arousal_recall']) if (df.loc[i, 'arousal_precision'] + df.loc[i, 'arousal_recall']) > 0 else 0
        
        # Valence
        if df.loc[i, 'valence_TP'] is not None:
            valence_tp = df.loc[i, 'valence_TP']
            valence_fp = df.loc[i, 'valence_FP']
            valence_tn = df.loc[i, 'valence_TN']
            valence_fn = df.loc[i, 'valence_FN']
            
            df.loc[i, 'valence_precision'] = valence_tp / (valence_tp + valence_fp) if (valence_tp + valence_fp) > 0 else 0
            df.loc[i, 'valence_recall'] = valence_tp / (valence_tp + valence_fn) if (valence_tp + valence_fn) > 0 else 0
            df.loc[i, 'valence_f1'] = 2 * (df.loc[i, 'valence_precision'] * df.loc[i, 'valence_recall']) / (df.loc[i, 'valence_precision'] + df.loc[i, 'valence_recall']) if (df.loc[i, 'valence_precision'] + df.loc[i, 'valence_recall']) > 0 else 0
    
    # 保存到Excel
    df.to_excel(results_file, index=False)
    print(f"当前结果已更新并保存到 {results_file}")

# 主函数
def main():
    # 数据目录
    data_dir = "cross-video-emotion-recognition/dataset/Dreamer/processed_de_psd"  # 请替换为实际数据目录
    
    # 结果保存路径
    results_file = "emotion_recognition_results.xlsx"
    
    # 情感维度
    emotion_dimensions = ['arousal', 'valence']
    
    # 存储结果
    results = {
        'video_name': [],
        'arousal_accuracy': [],
        'valence_accuracy': [],
        'arousal_TP': [],  # True Positive
        'arousal_FP': [],  # False Positive
        'arousal_TN': [],  # True Negative
        'arousal_FN': [],  # False Negative
        'valence_TP': [],
        'valence_FP': [],
        'valence_TN': [],
        'valence_FN': []
    }
    
    for emotion_dim in emotion_dimensions:
        print(f"\n开始 {emotion_dim} 维度的训练和测试...")
        
        # 加载所有视频数据
        video_data, video_labels, video_names = load_data(data_dir, emotion_dim)
        
        # Leave-one-video-out 交叉验证
        for test_idx in range(len(video_names)):
            test_video_name = video_names[test_idx]
            print(f"\n测试视频: {test_video_name}")
            
            # 准备测试数据
            test_data = video_data[test_idx]
            test_labels = video_labels[test_idx]
            
            # 准备训练数据
            train_data = []
            train_labels = []
            for i in range(len(video_names)):
                if i != test_idx:
                    train_data.append(video_data[i])
                    train_labels.append(video_labels[i])
            
            # 合并训练数据
            train_data = np.vstack(train_data)
            train_labels = np.vstack(train_labels)
            
            # 打乱训练数据
            train_data, train_labels = shuffle(train_data, train_labels, random_state=42)
            
            # 转换为PyTorch张量
            train_data = torch.FloatTensor(train_data)
            train_labels = torch.FloatTensor(train_labels)
            test_data = torch.FloatTensor(test_data)
            test_labels = torch.FloatTensor(test_labels)
            
            # 创建数据加载器
            train_dataset = TensorDataset(train_data, train_labels)
            test_dataset = TensorDataset(test_data, test_labels)
            
            # 划分一部分训练数据作为验证集
            train_size = int(0.8 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
            
            train_loader = DataLoader(
            train_subset, 
            batch_size=batch_size, 
            shuffle=True, 
            drop_last=True  # 丢弃最后一个不完整批次
            )            
            val_loader = DataLoader(val_subset, batch_size=batch_size)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
            
            # 创建模型
            model = CnnTwoDimensionModel(filters, kernel_size_list, dropout_rate, learning_rate)
            
            # 训练模型，传入视频名称和情感维度
            model = train_model(model, train_loader, val_loader, epochs, test_video_name, emotion_dim)
            
            # 测试模型
            accuracy, y_true, y_pred = test_model(model, test_loader)
            print(f"{emotion_dim} 测试准确率: {accuracy:.4f}")

            # 计算混淆矩阵
            cm = confusion_matrix(y_true, y_pred)

            # 提取混淆矩阵中的值
            # 假设是二分类问题，混淆矩阵为2x2
            # [[TN, FP], [FN, TP]]
            tn, fp, fn, tp = cm.ravel()
            
            # 存储结果
            if emotion_dim == 'arousal':
                if test_video_name not in results['video_name']:
                    results['video_name'].append(test_video_name)
                    results['arousal_accuracy'].append(accuracy)
                    results['arousal_TP'].append(tp)
                    results['arousal_FP'].append(fp)
                    results['arousal_TN'].append(tn)
                    results['arousal_FN'].append(fn)
                    # 初始化valence的值为None，后续会更新
                    results['valence_accuracy'].append(None)
                    results['valence_TP'].append(None)
                    results['valence_FP'].append(None)
                    results['valence_TN'].append(None)
                    results['valence_FN'].append(None)
                else:
                    idx = results['video_name'].index(test_video_name)
                    results['arousal_accuracy'][idx] = accuracy
                    results['arousal_TP'][idx] = tp
                    results['arousal_FP'][idx] = fp
                    results['arousal_TN'][idx] = tn
                    results['arousal_FN'][idx] = fn
            else:  # valence
                if test_video_name not in results['video_name']:
                    results['video_name'].append(test_video_name)
                    results['valence_accuracy'].append(accuracy)
                    results['valence_TP'].append(tp)
                    results['valence_FP'].append(fp)
                    results['valence_TN'].append(tn)
                    results['valence_FN'].append(fn)
                    # 初始化arousal的值为None，后续会更新
                    results['arousal_accuracy'].append(None)
                    results['arousal_TP'].append(None)
                    results['arousal_FP'].append(None)
                    results['arousal_TN'].append(None)
                    results['arousal_FN'].append(None)
                else:
                    idx = results['video_name'].index(test_video_name)
                    results['valence_accuracy'][idx] = accuracy
                    results['valence_TP'][idx] = tp
                    results['valence_FP'][idx] = fp
                    results['valence_TN'][idx] = tn
                    results['valence_FN'][idx] = fn

            # 每完成一个视频的测试就更新并保存结果
            update_and_save_results(results, results_file)
    
    # 所有测试完成后，计算并添加平均值
    df = pd.read_excel(results_file)
    
    # 计算平均值
    avg_arousal_acc = df['arousal_accuracy'].mean()
    avg_valence_acc = df['valence_accuracy'].mean()
    avg_arousal_tp = df['arousal_TP'].mean()
    avg_arousal_fp = df['arousal_FP'].mean()
    avg_arousal_tn = df['arousal_TN'].mean()
    avg_arousal_fn = df['arousal_FN'].mean()
    avg_valence_tp = df['valence_TP'].mean()
    avg_valence_fp = df['valence_FP'].mean()
    avg_valence_tn = df['valence_TN'].mean()
    avg_valence_fn = df['valence_FN'].mean()
    
    # 如果已经有Average行，则更新它；否则添加新行
    if 'Average' in df['video_name'].values:
        avg_idx = df[df['video_name'] == 'Average'].index[0]
        df.loc[avg_idx, 'arousal_accuracy'] = avg_arousal_acc
        df.loc[avg_idx, 'valence_accuracy'] = avg_valence_acc
        df.loc[avg_idx, 'arousal_TP'] = avg_arousal_tp
        df.loc[avg_idx, 'arousal_FP'] = avg_arousal_fp
        df.loc[avg_idx, 'arousal_TN'] = avg_arousal_tn
        df.loc[avg_idx, 'arousal_FN'] = avg_arousal_fn
        df.loc[avg_idx, 'valence_TP'] = avg_valence_tp
        df.loc[avg_idx, 'valence_FP'] = avg_valence_fp
        df.loc[avg_idx, 'valence_TN'] = avg_valence_tn
        df.loc[avg_idx, 'valence_FN'] = avg_valence_fn
    else:
        # 添加平均行
        avg_row = pd.DataFrame({
            'video_name': ['Average'],
            'arousal_accuracy': [avg_arousal_acc],
            'valence_accuracy': [avg_valence_acc],
            'arousal_TP': [avg_arousal_tp],
            'arousal_FP': [avg_arousal_fp],
            'arousal_TN': [avg_arousal_tn],
            'arousal_FN': [avg_arousal_fn],
            'valence_TP': [avg_valence_tp],
            'valence_FP': [avg_valence_fp],
            'valence_TN': [avg_valence_tn],
            'valence_FN': [avg_valence_fn]
        })
        df = pd.concat([df, avg_row], ignore_index=True)
    
    # 计算平均指标
    if 'arousal_precision' in df.columns:
        df.loc[df['video_name'] == 'Average', 'arousal_precision'] = df[df['video_name'] != 'Average']['arousal_precision'].mean()
        df.loc[df['video_name'] == 'Average', 'arousal_recall'] = df[df['video_name'] != 'Average']['arousal_recall'].mean()
        df.loc[df['video_name'] == 'Average', 'arousal_f1'] = df[df['video_name'] != 'Average']['arousal_f1'].mean()
        
        df.loc[df['video_name'] == 'Average', 'valence_precision'] = df[df['video_name'] != 'Average']['valence_precision'].mean()
        df.loc[df['video_name'] == 'Average', 'valence_recall'] = df[df['video_name'] != 'Average']['valence_recall'].mean()
        df.loc[df['video_name'] == 'Average', 'valence_f1'] = df[df['video_name'] != 'Average']['valence_f1'].mean()
    
    # 保存最终结果
    df.to_excel(results_file, index=False)
    print(f"\n最终结果已保存到 {results_file}")

if __name__ == "__main__":
    main()