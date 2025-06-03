import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, l):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

        # Create a tensor of zeros with shape (sequence_len, d_model)
        pe = torch.zeros(l, d_model)

        # Determine the device (GPU or CPU)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        # Move the positional encoding tensor to the determined device
        pe = pe.to(device)

        # Calculate the positional encodings
        for pos in range(l):
            for i in range(0, d_model, 2):
                pe[pos, i] = np.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = np.cos(pos / 10000 ** ((2 * (i + 1) / d_model)))
        
        # Add a batch dimension and set requires_grad to False
        self.pe = pe.unsqueeze(0)
        self.pe.requires_grad = False

    def forward(self, x):
        # Add positional encoding to the input tensor
        ret = np.sqrt(self.d_model) * x + self.pe
        return ret
    
    
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(1, 64, 65)
        self.conv2 = nn.Conv1d(64, 128, 33)
        self.conv3 = nn.Conv1d(128, 256, 17)
        #self.conv4 = nn.Conv1d(256, 512, 7)

        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        torch.nn.init.kaiming_normal_(self.conv3.weight)
        #torch.nn.init.kaiming_normal_(self.conv4.weight)

        self.dropout1 = nn.Dropout(0.1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        #self.bn4 = nn.BatchNorm1d(512)

        self.cls = nn.Parameter(torch.zeros(1, 1, 256))

        self.positionembedding = PositionalEncoding(256, 1169) # 如果更换卷积层参数或输入长度，这里需要相应调整
        #self.positionEmbedding = nn.Parameter(torch.zeros(1, 529, 256))

        encoderLayer = nn.TransformerEncoderLayer(d_model = 256, nhead = 4, dropout = 0.2, batch_first = True)
        self.encoder = nn.TransformerEncoder(encoderLayer, num_layers = 4)

        self.linear = nn.Linear(256, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.conv3(x)
        #x = self.dropout1(x)
        x = self.relu(x)
        x = self.bn3(x)
        #x = self.conv4(x)
        #x = self.relu(x)
        #x = self.bn4(x)
        x = torch.transpose(x, 2, 1)
        clsToken = self.cls.repeat_interleave(x.shape[0], dim = 0)
        x = torch.cat((clsToken, x), dim = 1)
        x = self.positionembedding(x)
        x = self.encoder(x)
        x = self.linear(x[:, 0, :])
        return x