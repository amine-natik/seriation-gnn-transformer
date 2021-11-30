import optuna.exceptions
import torch
from utils.utils import save_checkpoint, accuracy


class ModelEngine:
    def __init__(self, model, graph, loss_fn, optimizer, exp_name, num_epochs, save_dir, save_ckpt, log, show_log):
        self.model = model
        self.graph = graph
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.exp_name = exp_name
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        self.save_ckpt = save_ckpt
        self.log = log
        self.show_log = show_log

    def loss_backward(self, predictions, targets):
        self.optimizer.zero_grad()
        loss = self.loss_fn(predictions, targets)
        loss.backward()
        self.optimizer.step()
        return loss

    def train_one_epoch(self):
        inputs, targets, mask = self.graph.x, self.graph.y, self.graph.train_mask
        predictions = self.model(inputs)
        _ = self.loss_backward(predictions[mask], targets[mask])
        train_accuracy = accuracy(predictions[mask], targets[mask])
        return train_accuracy

    def train(self, trial=None):
        self.model.train()
        for epoch in range(self.num_epochs):
            train_accuracy = self.train_one_epoch()
            validation_accuracy = self.validate(is_test=False)

            if self.show_log:
                self.log.info(
                    f"Epoch [{epoch + 1}|{self.num_epochs}]: Train accuracy={train_accuracy:.4f}% -- Validation accuracy={validation_accuracy:.4f}%")

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
        inputs, targets = self.graph.x, self.graph.y
        if is_test:
            mask = self.graph.test_mask
        else:
            mask = self.graph.val_mask

        with torch.no_grad():
            predictions = self.model(inputs)
            val_accuracy = accuracy(predictions[mask], targets[mask])

        return val_accuracy

    def load_model(self, state_dict, device):
        self.model.load_state_dict(state_dict)
        self.model.to(device)
