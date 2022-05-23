import json
from typing import Optional

import torch

metrics = ["loss", "accuracy", "precision", "recall", "f1"]


class ConfusionMatrix:
    def __init__(self):
        self._confusion_matrix = torch.zeros(2, 2)

    def update(self, predictions, labels):
        for prediction, label in zip(predictions.view(-1), labels.view(-1)):
            self._confusion_matrix[prediction.long(), label.long()] += 1

    @property
    def accuracy(self):
        accuracy = self._confusion_matrix.diag().sum() / torch.sum(self._confusion_matrix)
        return accuracy.item()

    @property
    def precision(self):
        precision = self._confusion_matrix[0, 0] / (self._confusion_matrix[0, 0] + self._confusion_matrix[0, 1])
        return precision.item()

    @property
    def recall(self):
        recall = self._confusion_matrix[0, 0] / (self._confusion_matrix[0, 0] + self._confusion_matrix[1, 0])
        return recall.item()

    @property
    def f1(self):
        precision = self.precision
        recall = self.recall
        return 2 * precision * recall / (precision + recall)

    def all_metrics(self):
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1
        }

    @property
    def confusion_matrix(self):
        return self._confusion_matrix


class MetricStats:
    # useful only for multiple experiments
    def __init__(self):
        self.sum = {name: 0 for name in metrics}
        self.sumsq = {name: 0 for name in metrics}
        self.n = 0

    def update(self, _metrics):
        for name, value in _metrics.items():
            self.sum[name] += value
            self.sumsq[name] += value ** 2
        self.n += 1

    def get_stats(self):
        return {
            name: {
                "mean": self.sum[name] / self.n,
                "std": self.sumsq[name] / self.n - (self.sum[name] / self.n) ** 2
            } for name in metrics
        }


class MetricFactory:
    @staticmethod
    def from_loss_and_confmat(loss, confmat, prefix: Optional[str] = None):
        _metrics = {
            "loss": loss,
            "accuracy": confmat.accuracy,
            "precision": confmat.precision,
            "recall": confmat.recall,
            "f1": confmat.f1,
        }
        return {prefix: _metrics} if prefix else _metrics


class MetricLogger:
    def __init__(self, seed: int, hyperparameters: dict, verbose=True):
        self._verbose = verbose
        self._hyperparameters = hyperparameters
        self._seed = seed
        self._metrics = {"train": {}, "test": {}}
        if self._verbose:
            print(f"Hyperparameters: {json.dumps(hyperparameters, indent=4)}")
            print(f"Seed: {seed}")

    def log_train(self, epoch, train_metrics, val_metrics):
        self._metrics["train"][epoch] = {
            "train": train_metrics,
            "val": val_metrics
        }
        if self._verbose:
            print(f"Epoch {epoch}")
            print(json.dumps(self._metrics["train"][epoch], indent=4))

    def log_test(self, test_metrics):
        self._metrics["test"] = test_metrics
        if self._verbose:
            print(f"Test")
            print(json.dumps(self._metrics["test"], indent=4))

    def save_to_file(self, filename):
        with open(filename, 'w') as f:
            json.dump({
                "seed": self._seed,
                "hyperparameters": self._hyperparameters,
                "metrics": self._metrics
            }, f, indent=4)

    def get_test_metrics(self):
        return self._metrics["test"]
