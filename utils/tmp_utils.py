from sklearn.metrics import pairwise_distances
from sklearn.manifold import SpectralEmbedding
import torch


def get_affinity_matrix(graph):
    n = graph.x.size(0)
    D = float('inf') * torch.ones((n, n))
    tmp = []
    for (u, v) in graph.edge_index.transpose(0, 1):
        u, v = u.item(), v.item()
        D[u, v] = torch.linalg.norm(graph.x[u] - graph.x[v]) ** 2
        tmp.append(D[u, v])

    sigma = torch.sort(torch.tensor(tmp))[0][-5]
    affinity_matrix = torch.exp(- D / sigma)
    return affinity_matrix


def get_embedding(affinity_matrix, dim, method):
    if method == 'Spectral Embedding':
        model = SpectralEmbedding(n_components=dim, affinity='precomputed', eigen_solver='lobpcg')
    else:
        raise NotImplemented("Method not implemented")

    print('*** computing embedding...')
    embedding = model.fit_transform(affinity_matrix)
    print('*** Done !')
    return embedding


def get_distances(embedding):
    distances = torch.tensor(pairwise_distances(embedding))
    for i in range(distances.size(0)):
        distances[i, i] = float('inf')
    return distances


def get_permutation(distances):
    n = distances.size(0)
    inf_matrix = float('inf') * torch.ones(n)
    indices = [100]
    for _ in range(n-1):
        idx = indices[-1]
        next_idx = torch.argmin(distances[idx, :])
        indices.append(next_idx.item())
        distances[idx, :] = inf_matrix
        distances[:, idx] = inf_matrix

    permutation = torch.tensor(indices)
    return permutation

