from torch_geometric.datasets import Planetoid
from utils.preprocessing import label_ordering, random_ordering, no_ordering, laplacian_ordering

def load_graph(cfg):
    dataset = Planetoid(root=cfg.data_root, name=cfg.data)
    graph = dataset[0]
    return graph

def process_graph(graph, process_method, save_dir=None):
    if process_method == 'no_ordering':
        permutation = no_ordering(graph)

    elif process_method == 'laplacian_ordering':
        permutation = laplacian_ordering(graph, save_dir)

    elif process_method == 'label_ordering':
        permutation = label_ordering(graph)

    elif process_method == 'random_ordering':
        permutation = random_ordering(graph)

    else:
        raise NotImplemented("Preprocessing method not supported")

    for attr in ['x', 'y', 'train_mask', 'val_mask', 'test_mask']:
        graph[attr] = graph[attr][permutation]

    return graph