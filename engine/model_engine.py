import optuna.exceptions
import torch
from utils.utils import save_checkpoint
from dataloaders.cora import get_dataloader

class ModelEngine:
    def __init__(self, model, loss_fn, optimizer, graph, load_method, log, device, num_epochs, exp_name, save_dir, show_log, save_ckpt):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.graph = graph
        self.load_method = load_method
        self.dataloader = get_dataloader(self.graph, self.load_method)
        self.num_epochs = num_epochs
        self.device = device
        self.log = log
        self.exp_name = exp_name
        self.save_dir = save_dir
        self.show_log = show_log
        self.save_ckpt = save_ckpt

    def batch_loss_backward(self, predictions, targets):
        self.optimizer.zero_grad()
        loss = self.loss_fn(predictions, targets)
        loss.backward()
        self.optimizer.step()
        return loss

    def train_one_epoch(self, epoch):
        train_loss = 0
        correct = 0
        total = 0
        train_accuracy = 0
        for _, (inputs, targets) in enumerate(self.dataloader['trainer']):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            predictions = self.model(inputs)
            loss = self.batch_loss_backward(predictions, targets)
            train_loss += loss.item()
            _, predicted = predictions.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            train_accuracy = 100. * (correct/total)
        return train_accuracy

    def train(self, trial = None):
        self.model.train()
        for epoch in range(self.num_epochs):
            train_accuracy = self.train_one_epoch(epoch)
            validation_accuracy = self.validate(is_test=False)

            if self.show_log and (epoch + 1) % 10 == 0:
                self.log.info(f"Epoch [{epoch + 1}|{self.num_epochs}]: Train accuracy={train_accuracy:.4f}% -- Validation accuracy={validation_accuracy:.4f}%")

            if trial is not None:
                trial.report(validation_accuracy, epoch)

                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

        test_accuracy = self.validate(is_test=True)
        self.log.info(f"Model test accuracy={test_accuracy:.4f}%")
        if self.save_ckpt:
            self.log.info(f'Saving model to: {self.save_dir}')
            save_checkpoint(self.model, test_accuracy, self.num_epochs, self.exp_name, self.save_dir)

    def validate(self, is_test):
        self.model.eval()
        correct = 0
        total = 0
        if is_test:
            dataloader = self.dataloader['test']
        else:
            dataloader = self.dataloader['val']

        with torch.no_grad():
            for _, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        accuracy = 100. * (correct/total)

        return accuracy

    def load_model(self, state_dict, device):
        self.model.load_state_dict(state_dict)
        self.model.to(device)