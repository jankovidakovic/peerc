import json
import math
from typing import Optional, Iterable, Union

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

default_ignore_classes: Iterable[int] = frozenset({0})


class ClassificationMetrics:
    def __init__(self, num_classes: int = 4):
        self.num_classes = num_classes
        self._y_true = []
        self._y_pred = []

    def add_data(self, y_true, y_pred) -> None:
        """Adds the provided classification data.

        Each point in the provided vector should correspond to a result of a classification.
        That means that the vectors shouldn't be one-hot encoded.

        :param y_true: The vector of true labels.
        :param y_pred: The vector of predicted labels.
        """

        self._y_true.extend(y_true)
        self._y_pred.extend(y_pred)

    def accuracy(self) -> float:
        """Returns the accuracy metric, computed on all the data added so far.

        :return: Accuracy score.
        """
        return accuracy_score(self._y_true, self._y_pred)

    def precision(self, *,
                  average: str = "micro",
                  ignore_classes: Optional[Iterable[int]] = default_ignore_classes) -> float:
        """ Returns the precision score, computed on all the data added so far.

        Optionally, ignores some classes. The metric can also be computed in a macro or micro
        manner.

        :param average: type of averaging. "micro" corresponds to the micro-averaging of the
        precision, "macro" corresponds to the macro-averaging of the precision.
        :param ignore_classes: indices of classes to ignore. Note that the indices should
        correspond to the indices of the classes in the dataset, and that depends on the
        way that the labels are encoded. For example, for LSTM-baseline, label "0" corresponds
        to the class "other", but this may be different for other implementations.
        :return: Precision score.
        """

        include_classes = [i for i in range(self.num_classes) if i not in ignore_classes]
        return precision_score(self._y_true, self._y_pred, average=average, labels=include_classes)

    def recall(self, *,
               average: str = "micro",
               ignore_classes: Optional[Iterable[int]] = default_ignore_classes) -> float:
        """
        Returns the recall score, computed on all the data added so far.

        Optionally, ignores some classes. The metric can also be computed in a macro or micro
        manner.

        :param average: type of averaging. "micro" corresponds to the micro-averaging of the
        recall, "macro" corresponds to the macro-averaging of the recall.
        :param ignore_classes: indices of classes to ignore. Note that the indices should
        correspond to the indices of the classes in the dataset, and that depends on the
        way that the labels are encoded. For example, for LSTM-baseline, label "0" corresponds
        to the class "other", but this may be different for other implementations.
        :return: Recall score.
        """

        include_classes = [i for i in range(self.num_classes) if i not in ignore_classes]
        return recall_score(self._y_true, self._y_pred, average=average, labels=include_classes)

    def f1_score(self, *,
                 average: str = "micro",
                 ignore_classes: Optional[Iterable[int]] = default_ignore_classes) -> float:
        """
        Returns the f1-score, computed on all the data added so far.

        Optionally, ignores some classes. The metric can also be computed in a macro or micro
        manner.

        :param average: type of averaging. "micro" corresponds to the micro-averaging of the
        f1-score, "macro" corresponds to the macro-averaging of the f1-score.
        :param ignore_classes: indices of classes to ignore. Note that the indices should
        correspond to the indices of the classes in the dataset, and that depends on the
        way that the labels are encoded. For example, for LSTM-baseline, label "0" corresponds
        to the class "other", but this may be different for other implementations.
        :return: F1 score.
        """

        include_classes = [i for i in range(self.num_classes) if i not in ignore_classes]
        return f1_score(self._y_true, self._y_pred, average=average, labels=include_classes)

    def all_metrics(self, *,
                    average: str = "micro",
                    ignore_classes: Optional[Iterable[int]] = default_ignore_classes,
                    return_info: bool = False
                    ) -> Union[tuple[dict[str, float], dict[str, Union[int, str]]], dict[str, float]]:
        """Computes all the metrics, and returns them in a dictionary.

        :param average:  type of averaging. "micro" corresponds to the micro-averaging of the
        metrics, "macro" corresponds to the macro-averaging of the metrics.
        :param ignore_classes:  indices of classes to ignore. Note that the indices should
        correspond to the indices of the classes in the dataset, and that depends on the
        way that the labels are encoded. For example, for LSTM-baseline, label "0" corresponds
        to the class "other", but this may be different for other implementations.
        :param return_info:  Whether or not to return the information about metrics. If set to
        True, two dictionaries are returned: one with the metrics, and one with the information
        about the number of classes, the number of samples, type of averaging, and the
        indices of the ignored classes.
        :return: A dictionary with the metrics, or a tuple with two dictionaries, the first
        one with the metrics, and the second one with the information about the metrics.
        """

        accuracy = self.accuracy()
        precision = self.precision(average=average, ignore_classes=ignore_classes)
        recall = self.recall(average=average, ignore_classes=ignore_classes)
        _f1_score = self.f1_score(average=average, ignore_classes=ignore_classes)
        # return metrics as a dictionary

        all_metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": _f1_score,
        }

        if return_info:
            info = {
                "num_classes": self.num_classes,
                "num_samples": len(self._y_true),
                "average": average,
                "ignore_classes": "[" + ",".join(str(i) for i in ignore_classes) + "]"
            }
            return all_metrics, info

        return all_metrics

    def confusion_matrix(self):
        """Returns the confusion matrix, computed on all the data added so far.

        Rows correspond to the true labels, and columns correspond to the predicted labels.
        :return: Confusion matrix.
        """
        return confusion_matrix(self._y_true, self._y_pred)


class MetricStats:
    """Class that computes the elementary summary statistics of metrics.

    The class can be used to compute the mean, standard deviation, and standard error of
    the metrics. Multiple metrics can be added to the class, and the statistics are computed
    on all the metrics added so far.
    """

    def __init__(self):
        # sums of metric values
        self.sum: dict[str, int] = {}

        # sums of metric values squared
        self.sumsq: dict[str, int] = {}

        # number of samples
        self.n: int = 0

    def update(self, _metrics: dict[str, float]) -> None:
        """Updates the inner state with the provided values.

        :param _metrics:  Dictionary with the metrics. Each key should correspond to a metric
        name, and the value should be the metric value.
        """
        for name, value in _metrics.items():
            # check if the metric is in the list of metrics
            if name not in self.sum:
                self.sum[name] = 0
                self.sumsq[name] = 0
            self.sum[name] += value
            self.sumsq[name] += value ** 2
        self.n += 1

    def get_stats(self) -> dict[str, dict[str, float]]:
        """Returns the mean, standard deviation, and standard error of the metrics.

        :return: a dictionary of summary statistics. Keys are the metric names, and the values
        are dictionaries with the mean, standard deviation, and standard error.
        """

        return {
            name: {
                "mean": self.sum[name] / self.n,
                "std": self.sumsq[name] / self.n - (self.sum[name] / self.n) ** 2,
                "std_err": math.sqrt(self.sumsq[name] / self.n - (self.sum[name] / self.n) ** 2) / math.sqrt(self.n)
            } for name in self.sum
        }

    # TODO - implement significance testing


class MetricFactory:
    """ Class that creates dictionaries with metrics.

    """

    @staticmethod
    def from_loss_and_cls_metrics(
            loss: float,
            cls_metrics: ClassificationMetrics,
            average: str = "micro",
            ignore_classes: Optional[Iterable[int]] = default_ignore_classes):
        """Creates a dictionary of metrics from the given parameters.

        :param loss: Loss value.
        :param cls_metrics: Classification metrics.
        :param average: Type of averaging. "micro" corresponds to the micro-averaging of the
        metrics, "macro" corresponds to the macro-averaging of the metrics.
        :param ignore_classes: Indices of classes to ignore. Note that the indices should
        correspond to the indices of the classes in the dataset, and that depends on the
        way that the labels are encoded. For example, for LSTM-baseline, label "0" corresponds
        to the class "other", but this may be different for other implementations.
        :return: A dictionary with the metrics.
        """
        return {
            "loss": loss,
            **cls_metrics.all_metrics(average=average, ignore_classes=ignore_classes)
        }


class MetricLogger:
    """ Class that logs metrics.

    """

    def __init__(self, seed: int, hyperparameters: dict, verbose: bool = True):
        """ Initializes the logger with all the parameters that can affect the metrics.

        Metrics are stored in a dictionary. The keys correspond to the names of different
        dataset splits, and the values are dictionaries with the metrics. For training and
        validation splits, metrics can be saved for multiple epochs.

        :param seed: random seed used for the experiment run.
        :param hyperparameters:  Hyperparameters used for the experiment run.
        :param verbose: if set to True, prints the hyperparameters and the seed.
        """

        self._hyperparameters: dict = hyperparameters
        self._seed: int = seed
        self._metrics: dict[str, dict] = {}
        if verbose:
            print(f"Hyperparameters: {json.dumps(hyperparameters, indent=4)}")
            print(f"Seed: {seed}")

    @property
    def metrics(self) -> dict[str, dict]:
        """ Returns the metrics.

        :return: The metrics.
        """
        return self._metrics

    def log(self, _metrics: dict, epoch: Optional[int] = None, split: str = "train", verbose: bool = True) -> None:
        """ Logs the metrics.

        :param _metrics: a dictionary of metrics. The keys correspond to names of the metrics,
        and the values are the values of the metrics. Any dictionary obtained by
        calling methods from the MetricFactory class can be used.

        :param epoch: epoch number. If None, the metrics are assumed to be for the test split.
        :param split: dataset split. If epoch is None, the metrics are assumed to be for the
        test split.
        :param verbose: if set to True, prints the metrics.
        :return:
        """
        if split not in self._metrics:
            self._metrics[split] = {}
        if epoch is not None:  # train or val metrics
            self._metrics[split][epoch] = _metrics
        else:
            self._metrics[split] = _metrics  # test metrics

        if verbose:
            if epoch:  # if the epoch is not None
                print(f"Epoch {epoch}", end="; ")
            print(f"Metrics on {split}: ")
            if epoch:
                print(json.dumps(self._metrics[split][epoch], indent=4))
            else:
                print(json.dumps(self._metrics[split], indent=4))

    def save_to_file(self, filename) -> None:
        """ Saves the metrics to a file.

        The metrics are saved in a JSON format.
        The seed and the hyperparameters are saved as well.

        :param filename: Name of the file
        """
        with open(filename, 'w') as f:
            json.dump({
                "seed": self._seed,
                "hyperparameters": self._hyperparameters,
                "metrics": self._metrics
            }, f, indent=4)


def emo_metrics(eval_pred):
    y_pred, y_true = eval_pred

    # convert y_pred to logits
    y_pred = np.argmax(y_pred, axis=-1)

    metric_calc = ClassificationMetrics()
    metric_calc.add_data(y_true, y_pred)
    all_metrics = metric_calc.all_metrics()

    return all_metrics