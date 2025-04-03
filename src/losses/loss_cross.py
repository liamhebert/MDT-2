"""
Implementation for cross-entropy loss function.
"""

from typing import Literal, Mapping

import torch
from torch.nn import functional as F
from torchmetrics import Accuracy
from torchmetrics import F1Score
from torchmetrics import MaxMetric
from torchmetrics import MeanMetric
from torchmetrics import Metric
from torchmetrics import MetricCollection
from torchmetrics import MinMetric
from torchmetrics import Precision
from torchmetrics import Recall

from components.output_head import SimpleOutputHead
from losses.loss_abstract import Loss
from data.types import Labels


# TODO(liamhebert): create whatever loss set up we want here.
class NodeCrossEntropyLoss(Loss):
    """
    Cross-entropy loss function.
    """

    output_head: SimpleOutputHead
    weights: torch.Tensor

    def __init__(
        self,
        weights: list[float],
        output_head: SimpleOutputHead,
    ):
        super().__init__()
        assert len(weights) == 2
        assert any(w > 0 for w in weights)

        self.weight = torch.tensor(weights, requires_grad=False)

        self.output_head = output_head
        assert self.output_head.output_dim == 2

    def build_batch_metric_aggregators(
        self,
    ) -> dict[str, MetricCollection | Metric]:
        """Build metric collectors for batch metrics.

        TODO(liamhebert): Write more docs here
        """

        def make_metric_group(
            average: Literal["micro", "macro", "weighted", "none"],
        ) -> MetricCollection:
            return MetricCollection(
                (
                    {  # type: ignore
                        "recall": Recall(
                            task="multiclass",
                            num_classes=2,
                            average=average,
                            ignore_index=-100,
                        ),
                        "precision": Precision(
                            task="multiclass",
                            num_classes=2,
                            average=average,
                            ignore_index=-100,
                        ),
                        "f1": F1Score(
                            task="multiclass",
                            num_classes=2,
                            average=average,
                            ignore_index=-100,
                        ),
                    }
                    | (
                        {  # type: ignore
                            "accuracy": Accuracy(
                                task="multiclass",
                                num_classes=2,
                                average=average,
                                ignore_index=-100,
                            )
                        }
                        if average != "none"
                        else {}
                    )
                ),
                prefix=f"{average}_",
            )
            # mypy: enable-error-code="arg-type"

        return {
            "classification": MetricCollection(
                [
                    make_metric_group("macro"),  # type: ignore
                    make_metric_group("weighted"),  # type: ignore
                    make_metric_group("none"),  # type: ignore
                ]
            ),
            "loss": MeanMetric(),
        }

    def compute_batch_metrics(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        loss: torch.Tensor,
        metrics: dict[str, Metric],
    ) -> Mapping[str, torch.Tensor | Metric]:
        """Update metric objects with new batch.

        Args:
            logits: The predicted values with shape (B, C).
            targets: The true index values with shape (B,) within [0, C].
            loss: The loss value per sample with shape (B, ).
            metrics: The metric objects to update, which should include "loss"
                and "classification".

        Returns:
            Dictionary of metric values for the batch, containing
            - "loss": The loss value with shape (B,)
            - "classification": The classification metrics, organized by
                "(macro, weighed, class_0, class_1)_(metric_name)" for each
                metric. "none_(metric_name)_class_(i)"
        """
        batch_size, num_classes = logits.shape
        assert targets.shape == (
            batch_size,
        ), f"Unexpected shape: {targets.shape=}, {batch_size=}"
        assert num_classes == 2

        return_metrics = {}
        classification_metrics: dict[str, torch.Tensor] = metrics[
            "classification"
        ].forward(logits, targets)

        for key, value in classification_metrics.items():
            if value.shape == (num_classes,):
                # Unpack class-wise metrics into separate keys
                for i, v in enumerate(value):
                    class_key = key.replace("none", f"class_{i}")
                    return_metrics[class_key] = v
            else:
                assert value.shape == (), f"Unexpected shape: {value.shape}"
                return_metrics[key] = value

        return_metrics["loss"] = metrics["loss"].forward(loss)

        effective_batch_size = (targets != -100).sum().float()
        return_metrics["weight"] = effective_batch_size

        return return_metrics

    def build_epoch_metric_aggregators(
        self,
    ) -> Mapping[str, Metric | MetricCollection]:
        """
        Build run-level metric aggregators for each metric.
        """
        # TODO(liamhebert): Consider building this dynamically based on
        # batch_metrics

        return {
            f"best_{metric_type}_{metric}": MaxMetric()
            for metric_type in ["macro", "weighted", "class_0", "class_1"]
            for metric in ["f1", "recall", "precision", "accuracy"]
        } | {
            "best_loss": MinMetric(),
        }

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
            Dictionary of metric values for the epoch, containing
            - "best_loss": The best loss value
            - "best_classification": The best classification metrics, organized
                by "(macro, weighed, class_0, class_1)_(metric_name)" for each
                metric.
        """
        # Update the best metrics
        epoch_vals = {}
        metrics = batch_metrics["classification"].compute()
        for metric_type in ["macro", "weighted"]:
            for metric in ["f1", "recall", "precision", "accuracy"]:
                key = f"best_{metric_type}_{metric}"
                epoch_vals[key] = epoch_metrics[key].forward(
                    metrics[f"{metric_type}_{metric}"]
                )

        for metric in ["f1", "recall", "precision"]:
            key = f"best_class_0_{metric}"
            epoch_vals[key] = epoch_metrics[key].forward(
                metrics["none_" + metric][0]
            )

            key = f"best_class_1_{metric}"
            epoch_vals[key] = epoch_metrics[key].forward(
                metrics["none_" + metric][1]
            )

        epoch_vals["best_loss"] = epoch_metrics["best_loss"].forward(
            batch_metrics["loss"].compute()
        )

        for metric in batch_metrics.values():
            metric.reset()

        return epoch_vals

    def forward(
        self,
        node_embeddings: torch.Tensor,
        graph_embeddings: torch.Tensor,
        ys: Mapping[str, torch.Tensor],
        batch_metrics: dict[str, Metric | MetricCollection] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the cross-entropy loss.

        Args:
            node_embeddings: The embedding for each node in the batch. Shape
                (B, C, D), where C is the maximum number of comments in a
                discussion.
            graph_embeddings: The embedding for each graph in the batch. Shape
                (B, D).
            ys: Dictionary for the labels in the batch. Within that dictionary,
                this loss uses
                - y: The true label for each node. Note that each node will not
                    have a label, and therefore should be filtered using y_mask.
                    Shape (B, C).
            batch_metrics:
                A dictionary of metric objects to update with the batch metrics.
                This should be called with "compute_batch_metrics". If None, no
                metrics will be computed and only the loss will be returned.

        Returns:
            A dictionary of metrics corresponding
        """
        del graph_embeddings
        assert ys.keys() == {Labels.Ys}
        targets = ys[Labels.Ys]

        batch_size, _ = node_embeddings.shape
        assert targets.shape == (
            batch_size,
        ), f"{targets.shape=}, {batch_size=}"

        logits = self.output_head(node_embeddings)

        logits = torch.reshape(logits, (-1, 2))
        targets = torch.flatten(targets)

        loss = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            reduction="mean",
            ignore_index=-100,
        )

        if batch_metrics is not None:
            metrics = self.compute_batch_metrics(
                logits, targets, loss, batch_metrics
            )
        else:
            metrics = {}

        return loss, metrics
