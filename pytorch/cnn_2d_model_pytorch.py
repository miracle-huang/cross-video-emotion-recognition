import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import config

class CnnTwoDimensionModel(nn.Module):
    '''
    filters - list of the number of Convolution kernel of the model
    kernel_size_list - List of convolution kernel sizes
    dropout_rate - from config
    learning_rate - from config
    '''
    def __init__(self, filters, kernel_size_list, dropout_rate, learning_rate):
        super(CnnTwoDimensionModel, self).__init__()
        self.filters = filters
        self.kernel_size_list = kernel_size_list
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        # 构建模型层
        self.conv1 = nn.Conv2d(4, self.filters[0], self.kernel_size_list[0], padding='same')
        self.bn1 = nn.BatchNorm2d(self.filters[0])
        self.dropout1 = nn.Dropout(self.dropout_rate)
        
        self.conv2 = nn.Conv2d(self.filters[0], self.filters[1], self.kernel_size_list[1], padding='same')
        self.bn2 = nn.BatchNorm2d(self.filters[1])
        self.dropout2 = nn.Dropout(self.dropout_rate)
        
        self.conv3 = nn.Conv2d(self.filters[1], self.filters[2], self.kernel_size_list[2], padding='same')
        self.bn3 = nn.BatchNorm2d(self.filters[2])
        self.dropout3 = nn.Dropout(self.dropout_rate)
        
        self.conv4 = nn.Conv2d(self.filters[2], self.filters[0], self.kernel_size_list[3], padding='same')
        self.pool1 = nn.MaxPool2d(2, 2)
        self.bn4 = nn.BatchNorm2d(self.filters[0])
        self.dropout4 = nn.Dropout(self.dropout_rate)
        
        # 计算展平后的特征维度
        flattened_size = (8 // 2) * (9 // 2) * self.filters[0]
        
        self.dense1 = nn.Linear(flattened_size, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.dropout5 = nn.Dropout(self.dropout_rate)
        
        self.out = nn.Linear(512, 2)
        
    def forward(self, x):
        # 卷积层块1
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.dropout1(x)
        
        # 卷积层块2
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.dropout2(x)
        
        # 卷积层块3
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = self.dropout3(x)
        
        # 卷积层块4
        x = F.relu(self.conv4(x))
        x = self.pool1(x)
        x = self.bn4(x)
        x = self.dropout4(x)
        
        # 展平操作
        x = torch.flatten(x, 1)
        
        # 全连接层
        x = F.relu(self.dense1(x))
        x = self.bn5(x)
        x = self.dropout5(x)
        
        # 输出层
        x = self.out(x)
        x = F.softmax(x, dim=1)
        
        return x
    
    def create_2d_cnn_model(self):
        """为了保持与原代码接口一致，返回模型实例"""
        return self
    
    def compile(self, optimizer=None):
        """
        设置优化器，在PyTorch中我们通常在训练循环中处理损失函数
        """
        if optimizer is None:
            self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        else:
            self.optimizer = optimizer
        
        # 在PyTorch中，损失函数通常在训练循环中定义，而不是在模型中
        self.criterion = nn.CrossEntropyLoss()
        
        return self