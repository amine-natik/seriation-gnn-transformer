import torch
import torch.nn as nn
import math

class SinusoidalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len = 2708):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        x = x + self.pe[:,:x.size(1)]
        return self.dropout(x)
