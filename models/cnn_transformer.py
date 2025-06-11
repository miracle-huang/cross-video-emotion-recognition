import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 200):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)  # buffer 表示不作为参数更新

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]
    
    
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(2, 64, 65, stride=1)
        self.conv2 = nn.Conv1d(64, 128, 33, stride=1)
        self.conv3 = nn.Conv1d(128, 256, 17, stride=1)

        self.dropout1 = nn.Dropout(0.1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        #self.bn4 = nn.BatchNorm1d(512)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, 256))

        # 最大序列长度设置为200，实际会自动裁剪
        self.position_encoding = PositionalEncoding(d_model=256, max_len=1600)

        encoderLayer = nn.TransformerEncoderLayer(d_model = 256, nhead = 4, dropout = 0.2, batch_first = True)
        self.encoder = nn.TransformerEncoder(encoderLayer, num_layers = 2)

        self.linear = nn.Linear(256, 2)

    def forward(self, x):
        # x: (batch_size, 1, seq_len)
        x = self.relu(self.bn1(self.conv1(x)))  # shape: (B, 64, L1)
        x = self.relu(self.bn2(self.conv2(x)))  # shape: (B, 128, L2)
        x = self.relu(self.bn3(self.conv3(x)))  # shape: (B, 256, L3)

        x = x.transpose(1, 2)  # shape: (B, seq_len, d_model=256)

        # 加入 cls_token
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # shape: (B, 1, 256)
        x = torch.cat((cls_tokens, x), dim=1)  # shape: (B, seq_len+1, 256)

        x = self.position_encoding(x) # 添加位置编码

        # Transformer 编码
        x = self.encoder(x) # shape: (B, seq_len+1, 256)

        # 使用 cls_token 对应输出进行分类
        out = self.linear(x[:, 0, :])  # shape: (B, 2)
        return out