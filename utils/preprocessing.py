import torch

def label_ordering(data, targets): #Ordering with respect to the labels
    perm = torch.argsort(targets)
    data = data[perm]
    targets = targets[perm]
    return data, targets

def discrete_ordering(data, targets):
    n = data.shape[0]
    data, targets = label_ordering(data, targets)
    pos = (1/n) * torch.arange(n)
    data = (1/2) * (data + pos.unsqueeze(1))
    return data, targets
