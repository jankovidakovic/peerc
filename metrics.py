import json
from typing import Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch

metrics = ["loss", "accuracy", "precision", "recall", "f1"]


# def eval_perf_multi(Y, Y_):
#     precisions = []
#     recalls = []
#     n = max(Y_) + 1
#     M = np.bincount(n * Y_ + Y, minlength=n * n).reshape(n, n)
#     # TODO figure out how to do this without the loop
#     for i in range(n):
#         tp_i = M[i, i]
#         fn_i = np.sum(M[i, :]) - tp_i
#         fp_i = np.sum(M[:, i]) - tp_i
#         # recall_i = 0 if tp_i + fn_i == 0 else tp_i / (tp_i + fn_i)
#         # precision_i = 0 if tp_i + fp_i == 0 else tp_i / (tp_i + fp_i)
#         # precisions.append(precision_i)
#         # recalls.append(recall_i)
#
#     accuracy = np.trace(M) / np.sum(M)
#
#     return accuracy, precisions, recalls, M


class ConfusionMatrix:
    def __init__(self, num_classes: int = 2, device: str = "cuda"):
        self.num_classes = num_classes
        self._confusion_matrix = torch.zeros(num_classes, num_classes, device=device)
        self._tp = torch.zeros(num_classes, device=device)
        self._fp = torch.zeros(num_classes, device=device)
        self._fn = torch.zeros(num_classes, device=device)
        self._tn = torch.zeros(num_classes, device=device)
        self._yt = []
        self._yp = []

    def update(self, y_true, y_pred):
        # TODO - we actually pass y_pred as y_true but this should stil work
        # y_pred is of shape (batch_size, num_classes)
        # y_true is of shape (batch_size)
        self._confusion_matrix += torch.bincount(
            self.num_classes * y_true + y_pred, minlength=self.num_classes ** 2
            # the above line is equivalent to the following:
            # self._confusion_matrix[y_true, y_pred] += 1
        ).reshape(self.num_classes, self.num_classes)

        # cache tp, fp, fn
        for i in range(self.num_classes):
            self._tp[i] = self._confusion_matrix[i, i]
            self._fp[i] = self._confusion_matrix[i, :].sum() - self._confusion_matrix[i, i]
            self._fn[i] = self._confusion_matrix[:, i].sum() - self._confusion_matrix[i, i]
            self._tn[i] = self._confusion_matrix.sum() - self._tp[i] - self._fp[i] - self._fn[i]

            # before caching y_true and y_pred, we need to detach them from the computation graph
        self._yp.extend(y_true.cpu().detach().numpy())  # this is not an error
        self._yt.extend(y_pred.cpu().detach().numpy())
        # for prediction, label in zip(predictions.view(-1), labels.view(-1)):
        #    self._confusion_matrix[prediction.long(), label.long()] += 1

    def accuracy(self, ignore_classes: Optional[list[int]] = None):
        # include_classes = [i for i in range(self.num_classes) if i not in ignore_classes]
        return accuracy_score(self._yt, self._yp)
        # if ignore_classes is None:
        #     accuracy = (self._tp.sum()) / self._confusion_matrix.sum()
        #     # tn is removed
        # else:
        #     include_classes = list(set(range(self.num_classes)) - set(ignore_classes))
        #     accuracy = (self._tp[include_classes].sum()) \
        #                / self._confusion_matrix[include_classes, :][:, include_classes].sum()
            # tn is removed

        # return accuracy.item()

    # TODO lookup macro metrics

    def micro_precision(self, ignore_classes: Optional[list[int]] = None):
        # return self.accuracy(ignore_classes)  # micro-precision is the same as accuracy
        include_classes = [i for i in range(self.num_classes) if i not in ignore_classes]
        return precision_score(self._yt, self._yp, average="micro", labels=include_classes)

    def micro_recall(self, ignore_classes: Optional[list[int]] = None):
        include_classes = [i for i in range(self.num_classes) if i not in ignore_classes]
        return recall_score(self._yt, self._yp, average="micro", labels=include_classes)

    def micro_f1(self, ignore_classes: Optional[list[int]] = None):
        include_classes = [i for i in range(self.num_classes) if i not in ignore_classes]
        return f1_score(self._yt, self._yp, average="micro", labels=include_classes)

    def all_metrics(self, ignore_classes: Optional[list[int]] = None) -> dict:
        accuracy = self.accuracy(ignore_classes)
        precision = self.micro_precision(ignore_classes)
        recall = self.micro_recall(ignore_classes)
        f1 = self.micro_f1(ignore_classes)
        # return metrics as a dictionary
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    @property
    def confusion_matrix(self):
        return self._confusion_matrix


# TODO - write test for this


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
    def from_loss_and_confmat(loss, confmat, prefix: Optional[str] = None, ignore_classes: Optional[list[int]] = None):
        _metrics = {
            "loss": loss,
            "accuracy": confmat.accuracy(ignore_classes),
            "precision": confmat.micro_precision(ignore_classes),
            "recall": confmat.micro_recall(ignore_classes),
            "f1": confmat.micro_f1(ignore_classes),
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
