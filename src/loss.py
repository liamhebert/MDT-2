"""Loss classes."""

import abc
from dataclasses import dataclass
from enum import StrEnum
from typing import Tuple, Union

import torch
from torch.nn import functional as F
from torchmetrics import MaxMetric
from torchmetrics import MeanMetric
from torchmetrics import Metric
from torchmetrics import MetricCollection
from torchmetrics import MinMetric
from torchmetrics import SumMetric
from torchmetrics.classification import Accuracy
from torchmetrics.classification import F1Score
from torchmetrics.classification import Precision
from torchmetrics.classification import Recall


@dataclass
class Labels:
    """Dataclass to format possible label formats."""

    y: torch.Tensor
    y_mask: torch.Tensor | None
    hard_y: torch.Tensor | None


class LastValueMetric(SumMetric):
    """Metric that returns the last value it was updated with."""

    def update(self, value: Union[float, torch.Tensor]) -> None:
        """Update last value aggregate with new value.

        Args:
            value: The tensor value to keep
        """
        self.sum_value = self.sum_value * 0.0  # Reset the sum_value each time
        super().update(value)


class Loss(abc.ABC, torch.nn.Module):
    """Abstract class for loss functions."""

    @abc.abstractmethod
    def __call__(
        self,
        node_embeddings: torch.Tensor,
        graph_embeddings: torch.Tensor,
        ys: Labels,
    ) -> dict[str, torch.Tensor]:
        """Compute the loss function.

        Args:
            y_pred: The predicted values.
            y_true: The true values.

        Returns:
            The loss value.
        """
        pass


# TODO(liamhebert): create whatever loss set up we want here.
class NodeCrossEntropyLoss(Loss):
    """Cross-entropy loss function."""

    def __init__(self, positive_weight, negative_weight):
        self.weight = torch.tensor(
            [positive_weight, negative_weight], requires_grad=False
        )

        super().__init__()

    def build_batch_metric_aggregators(
        self,
    ) -> Tuple[dict[str, Metric | None]]:
        """Build metric collectors for batch metrics.

        TODO(liamhebert): Write more docs here
        """
        return {
            "positive_classification": MetricCollection(
                [
                    MetricCollection(
                        Recall(task="binary", average="macro"),
                        Precision(task="binary", average="macro"),
                        F1Score(task="binary", average="macro"),
                        prefix="macro_",
                    ),
                    MetricCollection(
                        Recall(task="binary", average="micro"),
                        Precision(task="binary", average="micro"),
                        F1Score(task="binary", average="micro"),
                        prefix="micro_",
                    ),
                ],
                prefix="positive_",
            ),
            "negative_classification": MetricCollection(
                [
                    MetricCollection(
                        Recall(task="binary", average="macro"),
                        Precision(task="binary", average="macro"),
                        F1Score(task="binary", average="macro"),
                        prefix="macro_",
                    ),
                    MetricCollection(
                        Recall(task="binary", average="micro"),
                        Precision(task="binary", average="micro"),
                        F1Score(task="binary", average="micro"),
                        prefix="micro_",
                    ),
                ],
                prefix="negative_",
            ),
            "overall_classification": MetricCollection(
                [
                    MetricCollection(
                        Recall(task="binary", average="macro"),
                        Precision(task="binary", average="macro"),
                        F1Score(task="binary", average="macro"),
                        prefix="macro_",
                    ),
                    MetricCollection(
                        Recall(task="binary", average="micro"),
                        Precision(task="binary", average="micro"),
                        F1Score(task="binary", average="micro"),
                        prefix="micro_",
                    ),
                ],
                prefix="overall_",
            ),
            "loss": MeanMetric(),
            "accuracy": MeanMetric(),
        }

    def compute_batch_metrics(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        metrics: dict[str, Metric],
    ) -> None:
        """Update metrics with new batch."""
        with torch.no_grad():
            ...

    def build_epoch_metric_aggregators(
        self,
    ) -> Tuple[dict[str, Metric | None], dict[str, Metric | None]]:
        """Build run-level metric aggregators."""
        return {
            f"best_{scope}_{metric_type}__{metric}": MaxMetric()
            for scope in ["positive", "negative", "overall"]
            for metric_type in ["macro", "micro"]
            for metric in ["f1", "recall", "precision"]
        } + {
            "best_loss": MinMetric(),
        }

    def compute_epoch_metrics(
        self,
        batch_metrics: dict[str, Metric],
        epoch_metrics: dict[str, Metric],
    ) -> None:
        """Update run-level metric aggregators."""
        for key, metric in batch_metrics.items():
            if "best_" + key in epoch_metrics:
                epoch_metrics["best_" + key].update(metric.compute())

    def __call__(
        self,
        node_embeddings: torch.Tensor,
        graph_embeddings: torch.Tensor,
        ys: Labels,
    ) -> dict[str, torch.Tensor]:
        """Compute the cross-entropy loss.

        Args:
            y_pred: The predicted values.
            y_true: The true values.

        Returns:
            The cross-entropy loss value.
        """
        del graph_embeddings
        targets = ys.y
        target_mask = ys.y_mask

        logits = node_embeddings[target_mask]
        targets = torch.flatten(targets)

        metrics = self.compute_metrics(logits, targets)
        # TODO(liamhebert): Check whether we can use "none" for a reduce
        metrics["loss"] = F.cross_entropy(
            logits, targets, weight=self.weight, reduction="none"
        )

        return metrics
