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


class SupContrastiveLoss(Loss):
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
        temperature: float = 0.05,
        learnable_temperature: bool = True,
        force_all_gather: bool = False,
        aux_cos_loss_weight: float = 1.0,
        aux_norm_loss_weight: float = 1.0,
    ):
        """Initializes the contrastive loss.

        Args:
            num_classes (int): The number of classes in the dataset,
                used for the classification metrics.
            temperature (float, optional): The temperature to use for the softmax
                function. A higher value will make the distribution more uniform,
                while a lower value will make the distribution more peaky.
                Defaults to 0.05.
            learnable_temperature (bool, optional): Whether to learn the
                temperature parameter. The initial value of the temperature will
                be `temperature`. Defaults to False.
        """
        super().__init__()

        self.temperature = nn.Parameter(
            torch.log(torch.tensor([temperature])),
            requires_grad=learnable_temperature,
        )
        self.num_classes = num_classes
        self.force_all_gather = force_all_gather
        self.aux_cos_loss_weight = aux_cos_loss_weight
        self.aux_norm_loss_weight = aux_norm_loss_weight

    @torch.compiler.disable
    def forward(
        self,
        node_embeddings: torch.Tensor | None,
        graph_embeddings: torch.Tensor,
        ys: Mapping[ContrastiveLabels, torch.Tensor],
        batch_metrics: dict[str, Metric | MetricCollection] | None = None,
        aux_loss: dict[str, torch.Tensor] | None = None,
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
        graph_embeddings = graph_embeddings
        normalized_embeddings = F.normalize(graph_embeddings, p=2, dim=-1)

        device_targets = ys[ContrastiveLabels.Ys]
        use_all_gather = (
            self.force_all_gather or torch.distributed.is_initialized()
        )

        if use_all_gather:
            all_targets, all_graph_embeddings = (
                self.all_gather(x)
                for x in (
                    device_targets,
                    normalized_embeddings,
                )
            )
            assert isinstance(all_graph_embeddings, torch.Tensor)
            graph_shape = graph_embeddings.shape
            targets, graph_embeddings = (
                all_targets.reshape(-1),
                all_graph_embeddings.reshape(-1, graph_shape[-1]),
            )
        else:
            targets = device_targets
            graph_embeddings = normalized_embeddings

        padding = targets == -100
        padding_mask = padding.repeat(targets.shape[0], 1)
        padding_mask = padding_mask | padding_mask.t()
        padding_mask = padding_mask.fill_diagonal_(True)

        target_matrix = (targets.unsqueeze(0) == targets.unsqueeze(1)).float()
        target_matrix = target_matrix * (~padding_mask).float()

        sim = torch.matmul(graph_embeddings, graph_embeddings.t())
        tau = self.temperature.exp().clamp_min(1e-6)
        sim = sim / tau

        logits_max, _ = torch.max(sim, dim=1, keepdim=True)
        logits = sim - logits_max.detach()

        exp_logits = torch.exp(logits) * (~padding_mask)
        denom = exp_logits.sum(1, keepdim=True).clamp_min(1e-12)
        log_prob = logits - denom.log()

        pos_counts = target_matrix.sum(1)
        mean_log_prob_pos = (target_matrix * log_prob).sum(
            1
        ) / pos_counts.clamp_min(1.0)
        valid = pos_counts > 0
        if valid.any():
            loss = -mean_log_prob_pos[valid].mean()
        else:
            loss = torch.zeros(
                (), device=graph_embeddings.device, dtype=graph_embeddings.dtype
            )

        if aux_loss is not None:
            total_loss = (
                loss
                + self.aux_norm_loss_weight * aux_loss.get("norm_loss", 0.0)
                + self.aux_cos_loss_weight * aux_loss.get("cos_loss", 0.0)
            )
        else:
            total_loss = loss

        if batch_metrics is not None:
            with torch.no_grad():
                metric_sim = sim.detach()
                metric_sim = metric_sim.fill_diagonal_(-1e9)
                metric_sim[padding_mask] = -1e9
                raw_metrics = self.compute_batch_metrics(
                    metric_sim, targets.detach(), loss.detach(), batch_metrics
                )
                metrics: dict[str, torch.Tensor] = {
                    k: v
                    for k, v in raw_metrics.items()
                    if isinstance(v, torch.Tensor)
                }
                metrics["original_loss"] = loss.detach()
                if aux_loss is not None:
                    for k, v in aux_loss.items():
                        metrics[f"aux_{k}"] = v.detach()
        else:
            metrics = {}
        return total_loss, metrics

    # ------------------------------------------------------------------
    # Basic metrics (only loss/weight unless subclass overrides)
    # ------------------------------------------------------------------
    def compute_batch_metrics(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        loss: torch.Tensor,
        metrics: dict[str, Metric | MetricCollection],
    ) -> Mapping[str, torch.Tensor | Metric]:
        batch_size, _ = logits.shape
        assert targets.shape == (
            batch_size,
        ), f"Unexpected shape: {targets.shape=} vs (B,)"

        # Derive pseudo-preds by nearest neighbor (argmax similarity) excluding
        # self sims already masked
        preds = (
            targets[logits.argmax(dim=1)] if targets.numel() > 0 else targets
        )

        out: dict[str, torch.Tensor | Metric] = {"loss": loss.detach()}
        if "classification" in metrics:
            metrics["classification"].update(preds, targets)
        effective_batch_size = (targets != -100).float().sum()
        out["weight"] = effective_batch_size
        out["temperature"] = self.temperature.exp().detach()
        return out


class SupContrastiveLossWithMetrics(SupContrastiveLoss):
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
            # ),
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
        preds = targets[logits.argmax(dim=1)]

        return_metrics = {}

        metrics["classification"].update(preds, targets)
        # metrics["loss"].update(loss)
        return_metrics["loss"] = loss

        effective_batch_size = (targets != -100).float().sum()
        return_metrics["weight"] = effective_batch_size
        return_metrics["temperature"] = self.temperature

        return return_metrics
