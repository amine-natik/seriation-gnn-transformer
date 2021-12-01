from torch_geometric.datasets import Planetoid
from utils.preprocessing import label_ordering, random_ordering, no_ordering, laplacian_ordering, se_ordering


def load_graph(data_root, data_name):
    dataset = Planetoid(root=data_root, name=data_name)
    graph = dataset[0]
    return graph


def process_graph(graph, process_method, save_dir=None, idx=None):
    if process_method == 'no_ordering':
        permutation = no_ordering(graph)

    elif process_method == 'se_ordering':
        permutation = se_ordering(graph, save_dir, idx)

    elif process_method == 'laplacian_ordering':
        permutation = laplacian_ordering(graph, save_dir, idx)

    elif process_method == 'label_ordering':
        permutation = label_ordering(graph)

    elif process_method == 'random_ordering':
        permutation = random_ordering(graph)

    else:
        raise NotImplemented("Preprocessing method not supported")

    for attr in ['x', 'y', 'train_mask', 'val_mask', 'test_mask']:
        graph[attr] = graph[attr][permutation]

    return graph
