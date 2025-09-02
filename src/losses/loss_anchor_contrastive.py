"""Contrastive loss function for pretraining with contrastive learning."""

import torch.nn as nn
import torch
from losses.loss_abstract import Loss
from torchmetrics.metric import Metric
from data.types import ContrastiveLabels
import torch.nn.functional as F
from torchmetrics import MetricCollection
from torchmetrics import Accuracy
from torchmetrics import F1Score
from torchmetrics import Precision
from torchmetrics import Recall
from typing import Literal, Mapping
from utils import RankedLogger

logger = RankedLogger(__name__)


class AnchorContrastiveLoss(Loss):
    """Contrastive loss function between the roi and candidate embeddings using
    in-batch negatives.

    This is done by aligning the positive roi regions to the positive candidate
    embedding, and the treating all other candidate embeddings as negatives. The
    implementation is similar to a cross entropy loss, where the "probability
    logits" of each class (candidates) is the cosine similarity score between the
    roi and candidate embeddings.

    This implementation is as proposed by InfoNCE, but with a modification that
    handles duplicate positive pairs.

    Since we use in-batch negatives, it is possible that multiple items within
    the same batch have the same positive candidate. However, InfoNCE only works
    with a single positive class (due to cross entropy loss). To handle this, we
    have an optional parameter ("remove_duplicates") that will check for and then
    remove duplicate positive and negative pairs.

    See: https://paperswithcode.com/method/infonce for more details.
    """

    cosine_similarity: nn.CosineSimilarity = torch.nn.CosineSimilarity(dim=2)
    soft_negative_weight: torch.Tensor
    adaptive_soft_negative_weight: bool
    temperature: nn.Parameter
    bias: nn.Parameter

    def __init__(
        self,
        num_classes: int,
        dim: int = 768,
        temperature: float = 10.0,
        learnable_temperature: bool = True,
    ):
        """Initializes the contrastive loss.

        Args:
            num_classes (int): The number of classes in the dataset,
                used for the classification metrics.
            dim (int): The dimension of the embeddings. Defaults to 768.
            temperature (float, optional): The temperature to use for the softmax
                function. A higher value will make the distribution more uniform,
                while a lower value will make the distribution more peaky.
                Defaults to 10.
            learnable_temperature (bool, optional): Whether to learn the
                temperature parameter. The initial value of the temperature will
                be `temperature`. Defaults to False.
        """
        super().__init__()

        self.temperature = nn.Parameter(
            torch.tensor([temperature]),
            requires_grad=learnable_temperature,
        )
        self.cluster_prototypes = nn.Parameter(
            torch.zeros(num_classes, dim, dtype=torch.float).normal_(
                mean=0.0, std=0.02
            ),
        )
        self.num_classes = num_classes

    @torch.compiler.disable
    def forward(
        self,
        node_embeddings: torch.Tensor | None,
        graph_embeddings: torch.Tensor,
        ys: Mapping[ContrastiveLabels, torch.Tensor],
        batch_metrics: dict[str, Metric | MetricCollection] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the contrastive pretraining loss.

        Args:
            node_embeddings: The embedding for each node in the batch. Shape
                (B, C, D), where C is the maximum number of comments in a
                discussion. Unused.
            graph_embeddings: The embedding for each graph in the batch. Shape
                (B, D).
            ys: Dictionary for the labels in the batch. Within that dictionary,
                this loss uses
                - y: The true positive label for each graph. Shape (B,). Nodes
                    with a label of -100 are ignored in the loss.
                - hard_y: The true hard negative label for each graph. Shape (B,).
            batch_metrics:
                A dictionary of metric objects to update with the batch metrics.
                This should be called with "compute_batch_metrics". If None, no
                metrics will be computed and only the loss will be returned.

        Returns:
            The cross-entropy loss value.
        """
        del node_embeddings

        # compute similarity matrix for contrastive loss
        graph_embeddings = graph_embeddings.to(torch.float32)
        normalized_A = F.normalize(graph_embeddings, p=2, dim=-1)
        cluster_protos = F.normalize(self.cluster_prototypes, p=2, dim=-1)

        targets = ys[ContrastiveLabels.Ys].to(torch.long)
        hard_targets = ys[ContrastiveLabels.HardYs]

        # scaling factor
        sim = torch.matmul(normalized_A, cluster_protos.t())
        sim = sim / self.temperature

        loss_graph = F.cross_entropy(
            sim, targets, ignore_index=-100, reduction="mean"
        )

        sim_anchor = (
            torch.matmul(cluster_protos, cluster_protos.t()) / self.temperature
        )  # (n, n) similarity matrix
        loss_anchor = F.cross_entropy(
            sim_anchor,
            torch.eye(self.num_classes, device=sim_anchor.device),
            reduction="mean",
        )

        loss = loss_graph + loss_anchor

        # add bias

        if batch_metrics is not None:
            with torch.no_grad():
                metrics = self.compute_batch_metrics(
                    sim,
                    targets.detach(),
                    loss.detach(),
                    batch_metrics,
                )
        else:
            metrics = {}
        return loss, metrics


class AnchorContrastiveLossWithMetrics(AnchorContrastiveLoss):
    """
    Contrastive loss function augmented with metrics. Trainers should use this
    class to be compatible with the Loss abstract class.
    """

    def build_batch_metric_aggregators(
        self,
    ) -> Mapping[str, MetricCollection | Metric]:
        """Build metric collectors for batch metrics.

        TODO(liamhebert): Write more docs here
        """

        def make_metric_group(
            average: Literal["micro", "macro", "weighted"],
        ) -> MetricCollection:
            return MetricCollection(
                (
                    {  # type: ignore
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
                    | (
                        {  # type: ignore
                            "accuracy": Accuracy(
                                task="multiclass",
                                num_classes=self.num_classes,
                                average=average,
                                ignore_index=-100,
                            )
                        }
                        if average == "weighted"
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
                    make_metric_group("micro"),  # type: ignore
                    # make_metric_group("none"),  # type: ignore
                ]
            )
            # "loss": MeanMetric(),
        }

    # @torch.compiler.disable
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
        batch_size, _ = logits.shape
        assert targets.shape == (
            batch_size,
        ), f"Unexpected shape: {targets.shape=}, {batch_size=}"

        # Don't allow self similarity
        preds = logits.argmax(dim=1)

        return_metrics = {}

        metrics["classification"].update(preds, targets)
        # metrics["loss"].update(loss)
        return_metrics["loss"] = loss

        effective_batch_size = (targets != -100).float().sum()
        return_metrics["weight"] = effective_batch_size
        return_metrics["temperature"] = self.temperature

        return return_metrics
