"""
Loss classes.
"""

import abc

import torch
from torchmetrics import Metric
from torchmetrics import SumMetric
from torchmetrics import MetricCollection
from typing import Mapping
from torch.distributed.nn import functional as distF


class LastValueMetric(SumMetric):
    """
    Metric that returns the last value it was updated with.
    """

    def update(self, value: float | torch.Tensor) -> None:
        """Update last value aggregate with new value.

        Args:
            value: The tensor value to keep
        """
        self.sum_value = self.sum_value * 0.0  # Reset the sum_value each time
        super().update(value)


class Loss(abc.ABC, torch.nn.Module):
    """
    Abstract class for loss functions.
    """

    is_distributed: bool = torch.distributed.is_initialized()

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        """Dummy all-gather function that does nothing. Useful for testing."""
        if self.is_distributed:
            gathered = distF.all_gather(x)
            return torch.stack(gathered, dim=0)
        return x

    @abc.abstractmethod
    def build_batch_metric_aggregators(
        self,
    ) -> Mapping[str, Metric | MetricCollection]:
        """Build metric collectors for batch metrics.

        TODO(liamhebert): Write more docs here
        """
        ...

    @abc.abstractmethod
    def build_epoch_metric_aggregators(
        self,
    ) -> Mapping[str, Metric | MetricCollection]:
        """
        Build run-level metric aggregators for each metric.
        """
        # TODO(liamhebert): Consider building this dynamically based on
        ...

    @abc.abstractmethod
    def compute_batch_metrics(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        loss: torch.Tensor,
        metrics: dict[str, Metric | MetricCollection],
    ) -> Mapping[str, torch.Tensor | Metric]:
        """Update metric objects with new batch.

        Args:
            logits: The predicted values with shape (B, C).
            targets: The true index values with shape (B,) within [0, C].
            loss: The loss value per sample with shape (B, ).
            metrics: The metric objects to update, which should include "loss"
                and "classification".

        Returns:
            Dictionary of metric values for the batch, which must contain
            - "loss": The loss value with shape (B,)
        """
        ...

    @abc.abstractmethod
    def compute_epoch_metrics(
        self,
        batch_metrics: dict[str, Metric | MetricCollection],
        epoch_metrics: dict[str, Metric | MetricCollection],
    ) -> Mapping[str, torch.Tensor]:
        """Update run-level metric aggregator with epoch metrics.

        This should be called at the end of each epoch to capture the best value
        for each metric. At the end of this function, we will reset all
        batch_metrics back to 0 for the next epoch.

        Args:
            batch_metrics: The metric objects for the epoch.
            epoch_metrics: The metric objects for the run.

        Returns:
            Dictionary of metric values for the epoch
        """
        ...

    @abc.abstractmethod
    def forward(
        self,
        node_embeddings: torch.Tensor,
        graph_embeddings: torch.Tensor,
        ys: Mapping[str, torch.Tensor],
        batch_metrics: dict[str, Metric | MetricCollection] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the cross-entropy loss.

        Args:
            y_pred: The predicted values.
            y_true: The true values.

        Returns:
            The cross-entropy loss value.
        """
        ...
