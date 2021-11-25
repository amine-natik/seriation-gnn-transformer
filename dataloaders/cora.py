from torch_geometric.datasets import Planetoid
from utils.preprocessing import label_ordering, discrete_ordering

def load_graph(cfg):
    dataset = Planetoid(root=cfg.data_root, name=cfg.data)
    graph = dataset[0]
    return graph

def get_dataloader(graph, load_method):
    dataloader = dict()
    dataloader['edge_index'] = graph.edge_index
    inputs_train, targets_train = graph.x[graph.train_mask], graph.y[graph.train_mask]
    inputs_val, targets_val = graph.x[graph.val_mask], graph.y[graph.val_mask]
    inputs_test, targets_test = graph.x[graph.test_mask], graph.y[graph.test_mask]

    if load_method == 'raw':
        pass

    elif load_method == 'label_ordering':
        inputs_train, targets_train = label_ordering(inputs_train, targets_train)
        inputs_val, targets_val = label_ordering(inputs_val, targets_val)
        inputs_test, targets_test = label_ordering(inputs_test, targets_test)

    elif load_method == 'discrete_ordering':
        inputs_train, targets_train = discrete_ordering(inputs_train, targets_train)
        inputs_val, targets_val = discrete_ordering(inputs_val, targets_val)
        inputs_test, targets_test = discrete_ordering(inputs_test, targets_test)

    else:
        raise NotImplemented("Preprocessing method not supported")
    dataloader['trainer'] = [(inputs_train, targets_train)]
    dataloader['val'] = [(inputs_val, targets_val)]
    dataloader['test'] = [(inputs_test, targets_test)]
    return dataloader


