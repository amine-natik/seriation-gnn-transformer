import torch
import torch.nn as nn
import math

class KernelEncoding(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        seq_len_part = {"trainer":140, "val":500, "test":1000}
        for partition in ["trainer", "val", "test"]:
            seq_len = seq_len_part[partition]
            position = (1 / seq_len) * torch.arange(seq_len).unsqueeze(1)
            w = torch.randn(d_model)
            b = 2 * math.pi * torch.rand(d_model)
            pos_embedding = (2/math.sqrt(d_model)) * torch.cos(position * w + b).unsqueeze(0)
            name = "pos_embedding_" + partition
            self.register_buffer(name, pos_embedding)
            if partition == "trainer":
                self.pos_embeddings_train = pos_embedding.to('cuda')
            if partition == "val":
                self.pos_embeddings_val = pos_embedding.to('cuda')
            if partition == "test":
                self.pos_embeddings_test = pos_embedding.to('cuda')

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len == 140:
            pos_embedding = self.pos_embeddings_train
        if seq_len == 500:
            pos_embedding = self.pos_embeddings_val
        if seq_len == 1000:
            pos_embedding = self.pos_embeddings_test
        else:
            raise NotImplementedError("Graph Length not supported")

        x = x + pos_embedding
        return self.dropout(x)
