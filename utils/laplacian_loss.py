import torch
import torch.nn as nn


class LaplacianLoss:
    def __init__(self, graph, alpha, sigma):
        self.graph = graph
        self.alpha = alpha
        self.loss_cls = nn.NLLLoss()
        self.sigma = sigma

    def laplacian_reg(self, embedding):
        lap_reg = 0
        for (u, v) in self.graph.edge_index.transpose(0, 1):
            u, v = u.item(), v.item()
            graph_dist = torch.linalg.norm(self.graph.x[u] - self.graph.x[v]) ** 2
            emb_dist = torch.linalg.norm(embedding[u] - embedding[v]) ** 2
            lap_reg += torch.exp(-graph_dist / self.sigma) * emb_dist
        return lap_reg

    def __call__(self, predictions, targets, embedding):
        return self.loss_cls(predictions, targets) + self.alpha * self.laplacian_reg(embedding)