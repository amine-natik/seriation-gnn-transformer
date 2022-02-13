import torch
from pathlib import Path


def set_experiment_name(**kwargs):
    return "-".join([f'{key}_{kwargs[key]}' for key in kwargs])


def save_checkpoint(model, test_accuracy, epoch, exp_name, save_dir):
    checkpoint = {
        'model': model.state_dict(),
        'test_accuracy': test_accuracy,
        'epoch': epoch,
    }
    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir()
    save_path = save_dir.joinpath(exp_name)
    torch.save(checkpoint, save_path)


def load_checkpoint(save_dir, exp_name):
    save_path = Path(save_dir, exp_name)
    return torch.load(save_path)


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1)
    correct = (predictions == targets).sum()
    acc = 100 * (int(correct) / targets.shape[0])
    return acc