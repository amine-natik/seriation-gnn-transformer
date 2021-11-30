import torch
import torch.nn as nn
import math


class KernelEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        w = torch.randn(d_model)
        b = 2 * math.pi * torch.rand(d_model)
        pe = (2 / math.sqrt(d_model)) * torch.cos(position * w + b).unsqueeze(0)
        self.register_buffer('pe', pe)
        self.pe = pe

    def forward(self, x):
        x = x + self.pe[:,:x.size(1)]
        return self.dropout(x)
