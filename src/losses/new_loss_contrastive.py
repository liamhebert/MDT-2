"""Contrastive loss function for pretraining with contrastive learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import (
    Accuracy,
    F1Score,
    MaxMetric,
    MeanMetric,
    MinMetric,
    Metric,
    MetricCollection,
    Precision,
    Recall,
)

from losses.loss_abstract import Loss  # Assuming this import is necessary
from data.types import ContrastiveLabels  # Assuming this import is necessary
from typing import Literal, Mapping


class ContrastiveLossWithMetrics(Loss):
    """Optimized Contrastive loss function with integrated metrics."""

    def __init__(
        self,
        num_classes: int,
        soft_negative_weight: float = 0.0,
        adaptive_soft_negative_weight: bool = False,
        temperature: float = 10.0,
        bias: float = -10.0,
        learnable_temperature: bool = True,
        force_all_gather: bool = False,
    ):
        """Initializes the contrastive loss.

        Args:
            num_classes: Number of classes for classification metrics.
            soft_negative_weight: Weight for soft negative pairs.
            adaptive_soft_negative_weight: Adapt soft negative weight.
            temperature: Softmax temperature.
            bias: Similarity matrix bias.
            learnable_temperature: Whether temperature is learnable.
        """
        super().__init__()

        assert isinstance(soft_negative_weight, float)
        # No need to wrap in a Tensor; store as float
        self.soft_negative_weight = soft_negative_weight
        self.adaptive_soft_negative_weight = adaptive_soft_negative_weight
        self.temperature = nn.Parameter(
            torch.tensor(temperature).log(), requires_grad=learnable_temperature
        )
        self.bias = nn.Parameter(
            torch.tensor(bias), requires_grad=learnable_temperature
        )
        self.num_classes = num_classes

        # Pre-build metric aggregators (more efficient)
        self.batch_metrics = self.build_batch_metric_aggregators()
        self.epoch_metrics = self.build_epoch_metric_aggregators()
        self.use_all_gather = (
            torch.distributed.is_initialized()
            and self.all_gather_fn is not None
        ) or force_all_gather

    def build_batch_metric_aggregators(
        self,
    ) -> Mapping[str, Metric | MetricCollection]:
        """Build metric collectors for batch metrics."""

        def make_metric_group(
            average: Literal["micro", "macro", "weighted", "none"],
        ) -> MetricCollection:
            metrics = {
                "recall": Recall(
                    task="multiclass",
                    num_classes=self.num_classes,
                    average=average,
                    ignore_index=-100,
                ),
                "precision": Precision(
                    task="multiclass",
                    num_classes=self.num_classes,
                    average=average,
                    ignore_index=-100,
                ),
                "f1": F1Score(
                    task="multiclass",
                    num_classes=self.num_classes,
                    average=average,
                    ignore_index=-100,
                ),
            }
            if average == "weighted":
                metrics["accuracy"] = Accuracy(
                    task="multiclass",
                    num_classes=self.num_classes,
                    average=average,
                    ignore_index=-100,
                )
            return MetricCollection(metrics, prefix=f"{average}_")

        return {
            "classification": MetricCollection(
                [
                    make_metric_group("macro"),
                    make_metric_group("weighted"),
                    make_metric_group("none"),
                ]
            ),
            "loss": MeanMetric(),
        }

    def build_epoch_metric_aggregators(
        self,
    ) -> Mapping[str, Metric | MetricCollection]:
        """Build run-level metric aggregators."""
        return {
            f"best_{metric_type}_{metric}": MaxMetric()
            for metric_type in ["macro", "weighted"]
            + [f"class_{i}" for i in range(self.num_classes)]
            for metric in ["f1", "recall", "precision", "accuracy"]
        } | {"best_loss": MinMetric()}

    def _compute_similarity(
        self, graph_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Computes the scaled similarity matrix."""
        # Normalize once, before the matmul (more efficient)
        normalized_embeddings = F.normalize(graph_embeddings, p=2, dim=-1)
        sim = torch.matmul(
            normalized_embeddings, normalized_embeddings.transpose(0, 1)
        )
        return sim * self.temperature.exp() + self.bias

    def _compute_target_matrices(
        self, targets: torch.Tensor, hard_targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes target, hard target, and soft label matrices."""
        padding = targets == -100
        padding_mask = padding[:, None] | padding  # Use broadcasting

        target_matrix = targets[:, None] == targets  # More concise
        target_matrix.fill_diagonal_(False)  # More efficient than -100, then 0
        target_matrix[padding_mask] = False  # Treat padding as False

        hard_target_matrix = hard_targets[:, None] == targets
        hard_target_matrix[padding_mask] = False

        soft_labels = ~(
            target_matrix | hard_target_matrix
        )  # Logical operations

        return target_matrix, hard_target_matrix, soft_labels

    def _compute_loss_weights(
        self,
        soft_labels: torch.Tensor,
        target_matrix: torch.Tensor,
        hard_target_matrix: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the weights for the contrastive loss."""

        if self.adaptive_soft_negative_weight:
            extra_weight = 1.0 / torch.clamp(
                soft_labels.sum(dim=1, keepdim=True), min=1
            )  # keepdim for broadcasting
        else:
            extra_weight = self.soft_negative_weight

        soft_matrix = torch.where(soft_labels, extra_weight, 1.0)
        soft_matrix.fill_diagonal_(0.0)  # Remove self-similarity
        soft_matrix[padding_mask] = 0.0

        # Normalize weights efficiently
        soft_matrix = F.normalize(soft_matrix, p=1, dim=1)

        # Calculate num_valid_labels based on soft_matrix (more consistent)
        num_valid_labels = torch.clamp((soft_matrix > 0).sum(dim=1), min=1)
        soft_matrix = (
            soft_matrix * num_valid_labels[:, None]
        )  # Scale by num_valid_labels
        return soft_matrix

    def forward(
        self,
        node_embeddings: torch.Tensor | None,
        graph_embeddings: torch.Tensor,
        ys: Mapping[ContrastiveLabels, torch.Tensor],
        batch_metrics: (
            dict[str, Metric | MetricCollection] | None
        ) = None,  # Keep for compatibility
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the contrastive pretraining loss.

        Args:
            node_embeddings: Unused.
            graph_embeddings: Embeddings for each graph (B, D).
            ys: Labels, including ContrastiveLabels.Ys and .HardYs.
            batch_metrics: Optional dict of metrics (unused, kept for
                compatibility).

        Returns:
            Loss and metrics.
        """
        del node_embeddings  # Explicitly indicate it's unused.
        if batch_metrics is not None:
            print(
                "Warning: batch_metrics argument is deprecated and will be"
                " ignored."
            )

        device_targets = ys[ContrastiveLabels.Ys]
        device_hard_targets = ys[ContrastiveLabels.HardYs]

        # all_gather is crucial for distributed training, but adds complexity.
        # If *not* using distributed training, skip it for efficiency.
        if self.use_all_gather:  # type: ignore
            all_targets, all_hard_targets = self.all_gather_fn(  # type: ignore
                (device_targets, device_hard_targets)
            )
            all_graph_embeddings = self.all_gather_fn(
                graph_embeddings  # type: ignore
            )
            targets = all_targets.reshape(-1)
            hard_targets = all_hard_targets.reshape(-1)
            graph_embeddings = all_graph_embeddings.reshape(
                -1, graph_embeddings.shape[-1]
            )
        else:
            targets = device_targets
            hard_targets = device_hard_targets
            # No reshaping needed if not using all_gather

        # Similarity
        sim = self._compute_similarity(graph_embeddings)

        # Target matrices
        padding = targets == -100
        padding_mask = padding[:, None] | padding
        target_matrix, hard_target_matrix, soft_labels = (
            self._compute_target_matrices(targets, hard_targets)
        )

        # Loss weights
        soft_matrix = self._compute_loss_weights(
            soft_labels, target_matrix, hard_target_matrix, padding_mask
        )

        # Loss calculation (optimized)
        target_matrix_signed = (
            target_matrix.float() * 2 - 1
        )  # Convert bool to [-1, 1]
        loss = torch.einsum(
            "ij,ij->i", -F.logsigmoid(sim * target_matrix_signed), soft_matrix
        )

        # Handle the case where an entire batch might be padding
        num_valid_for_loss = torch.clamp((~padding).sum(), min=1)
        loss = loss.sum() / num_valid_for_loss

        # --- Metrics ---
        with torch.no_grad():
            metric_sim = sim.detach()
            metric_sim[padding_mask] = -1e9
            metric_sim = metric_sim.fill_diagonal_(-1e9)
            metrics = self.compute_batch_metrics(metric_sim, targets, loss)

        return loss, metrics

    def compute_batch_metrics(
        self, logits: torch.Tensor, targets: torch.Tensor, loss: torch.Tensor
    ) -> Mapping[str, torch.Tensor]:
        """Update metric objects with new batch."""
        preds = targets.clone()  # avoid modifying the original targets

        preds = preds[logits.argmax(dim=1)]  # Now correct
        padding_mask = targets == -100
        preds[padding_mask] = -100

        self.batch_metrics["classification"].to(preds.device).update(
            preds, targets
        )
        self.batch_metrics["loss"].to(loss.device).update(loss)

        effective_batch_size = (~padding_mask).sum()
        return {
            "loss": loss.detach(),
            "weight": effective_batch_size,  # No .item() needed
            "bias": self.bias.detach(),
            "temperature": self.temperature.exp().detach(),
        }

    def compute_epoch_metrics(
        self,
        batch_metrics: (
            dict[str, Metric | MetricCollection] | None
        ) = None,  # Keep for compatibility
        epoch_metrics: (
            dict[str, Metric | MetricCollection] | None
        ) = None,  # Keep for compatibility
    ) -> Mapping[str, torch.Tensor]:
        """Update run-level metric aggregator with epoch metrics."""

        if batch_metrics is not None:
            print(
                "Warning: batch_metrics argument is deprecated and will be"
                " ignored. Use self.batch_metrics"
            )
        if epoch_metrics is not None:
            print(
                "Warning: epoch_metrics argument is deprecated and will be"
                " ignored. Use self.epoch_metrics"
            )

        ret_metrics = {}
        device = self.batch_metrics["classification"]["macro_f1"].device
        metrics = self.batch_metrics["classification"].compute()

        for metric_type in ["macro", "weighted"]:
            metric_ids = ["f1", "recall", "precision"]
            if metric_type == "weighted":
                metric_ids.append("accuracy")
            for metric in metric_ids:
                key = f"best_{metric_type}_{metric}"
                metric_value = metrics[f"{metric_type}_{metric}"]
                ret_metrics[f"{metric_type}_{metric}"] = metric_value
                self.epoch_metrics[key].to(device).update(metric_value)
                ret_metrics[key] = self.epoch_metrics[key].compute()

        for metric in ["f1", "recall", "precision"]:
            metric_values = metrics["none_" + metric]
            for class_id in range(self.num_classes):
                key = f"best_class_{class_id}_{metric}"
                ret_metrics[f"class_{class_id}_{metric}"] = metric_values[
                    class_id
                ]
                self.epoch_metrics[key].to(device).update(
                    metric_values[class_id]
                )
                ret_metrics[key] = self.epoch_metrics[key].compute()

        self.epoch_metrics["best_loss"].to(device).update(
            self.batch_metrics["loss"].compute()
        )
        ret_metrics["best_loss"] = self.epoch_metrics["best_loss"].compute()

        # Reset batch metrics *after* computing epoch metrics
        for metric_obj in self.batch_metrics.values():
            metric_obj.reset()

        return ret_metrics
