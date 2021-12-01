import torch
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
from scipy.sparse.linalg import eigsh
import os
from pathlib import Path
from utils.tmp_utils import *


def se_ordering(graph, save_dir, idx):
    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir()
    save_path = save_dir.joinpath(f".permutation_se_idx:{idx}.pt")

    if os.path.isfile(save_path):
        permutation = torch.load(save_path)
    else:
        affinity_matrix = get_affinity_matrix(graph)
        embedding = get_embedding(affinity_matrix, dim=idx, method='Spectral Embedding')
        distances = get_distances(embedding)
        permutation = get_permutation(distances)
        torch.save(permutation, save_path)

    return permutation


def laplacian_ordering(graph, save_dir, idx):
    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir()
    save_path = save_dir.joinpath('.eigenvectors.pt')

    if os.path.isfile(save_path):
        eigenvectors = torch.load(save_path)
    else:
        laplacian = get_laplacian(graph.edge_index)
        indices = laplacian[0]
        attributes = laplacian[1]
        laplacian_matrix = to_scipy_sparse_matrix(indices, attributes)
        print('*** computing spectrum...')
        _, eigenvectors = eigsh(A=laplacian_matrix, k=2707, which='LA')
        print('*** Done !')
        torch.save(eigenvectors, save_path)

    fiedler = torch.tensor(eigenvectors[:, idx])
    permutation = torch.argsort(fiedler)
    return permutation


def randomly_permute_labels(graph):
    labels = graph.y
    random_permutation = torch.randperm(labels.shape[0])
    graph.y = labels[random_permutation]
    return graph


def no_ordering(graph):
    return torch.arange(graph.x.shape[0])


def label_ordering(graph):
    return torch.argsort(graph.y)


def random_ordering(graph):
    return torch.randperm(graph.x.shape[0])
