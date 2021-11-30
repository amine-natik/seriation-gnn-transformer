import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from positional_encodings.sinusoidal_encoding import SinusoidalEncoding
import torch


class RawTransformer(nn.Module):
    def __init__(self, n_tokens, d_model, n_head, dim_feedforward,
                 num_layers, n_classes, dropout, max_len):
        super(RawTransformer, self).__init__()
        self.d_model = d_model
        self.positional_encoding = SinusoidalEncoding(d_model=d_model, dropout=dropout, max_len=max_len)
        encoder_layers = TransformerEncoderLayer(d_model=d_model, nhead=n_head,
                                                 dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.encoder = nn.Linear(n_tokens, d_model)
        self.decoder = nn.Linear(d_model, n_classes)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        self.encoder.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, src):
        src = self.encoder(src)
        src = self.dropout(F.relu(src)).unsqueeze(0)
        src = self.positional_encoding(src)
        src = self.transformer_encoder(src).reshape(-1, self.d_model)
        src = self.decoder(src)
        return F.log_softmax(src, dim=1)
