"""SigLIP-style intra-modality contrastive loss.

This module implements a pairwise BCE-with-logits objective over the full
in-batch similarity matrix (SigLIP-style). It supports an additive bias term
and temperature scaling of the similarities. Because this project performs
intra-modality contrastive learning, the diagonal entries are removed from the
loss (a sample isn't contrasted with itself), while all off-diagonal pairs are
included as negatives by default. Optionally, hard negatives can be passed to
adjust soft-negative weights.

Key features
- Pairwise BCE-with-logits over similarity matrix with learned log-temperature
    and bias.
- Diagonal removed for intra-modality setup; all other pairs are used.
- Optional weighting of soft negatives; adaptive or fixed.
- Optional symmetric reduction (row-wise and column-wise) or row-wise only to
    match SigLIP Algorithm 1 exactly.
"""

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


class ContrastiveLoss(Loss):
    """SigLIP-style contrastive loss with in-batch negatives.

    The loss is computed on the full similarity matrix S = T * cos(A @ A^T) + b
    where A are normalized graph embeddings, T is the temperature (learnable by
    default; stored in log-space), and b is an additive bias (learnable if
    desired). Labels identify positives; all other pairs are treated as
    negatives. For intra-modality training, the diagonal is removed. Optional
    hard negatives are supported via ``hard_y`` to reweight soft negatives.

    Reduction can be row-wise only (to match SigLIP Algorithm 1) or symmetric
    row/column-wise by setting ``symmetric=True``.
    """

    cosine_similarity: nn.CosineSimilarity = torch.nn.CosineSimilarity(dim=2)
    soft_negative_weight: torch.Tensor
    adaptive_soft_negative_weight: bool
    temperature: nn.Parameter
    bias: nn.Parameter

    def __init__(
        self,
        num_classes: int,
        soft_negative_weight: float = 1.0,
        adaptive_soft_negative_weight: bool = False,
        temperature: float = 10.0,
        bias: float = -10.0,
        learnable_temperature: bool = True,
        force_all_gather: bool = False,
        symmetric: bool = True,
        aux_cos_loss_weight: float = 1.0,
        aux_norm_loss_weight: float = 1.0,
    ):
        """Initializes the SigLIP-style contrastive loss.

        Args:
            num_classes (int): The number of classes in the dataset,
                used for the classification metrics.
            soft_negative_weight (float, optional): Weight to associate to soft
                negative pairs in the contrastive loss. Flag is exclusive against
                adaptive_soft_negative_weight
            adaptive_soft_negative_weight (bool, optional): Whether to adapt
                the soft negative weight based on the number of positive pairs
                and negative pairs. Flag is exclusive against soft_negative_weight
            temperature (float, optional): Initial temperature; stored as log(T)
                and exponentiated during the forward pass. Higher T smooths
                similarities; lower T sharpens them. Defaults to 10.0.
            bias (float): Additive bias applied to all pairwise similarities.
                Defaults to -10.0.
            learnable_temperature (bool, optional): Whether to learn the
                temperature and bias parameters. Defaults to True.
            force_all_gather (bool, optional): If True or if torch.distributed is
                initialized, embeddings and labels are gathered across workers
                before loss computation. Defaults to False.
            symmetric (bool, optional): If True, compute both row-wise and
                column-wise reductions and average them; if False, compute
                row-wise only (closer to SigLIP Algorithm 1). Defaults to True.
        """
        super().__init__()

        assert isinstance(soft_negative_weight, float)
        self.soft_negative_weight = nn.Parameter(
            torch.tensor([soft_negative_weight]).float(), requires_grad=False
        )
        self.adaptive_soft_negative_weight = adaptive_soft_negative_weight
        self.temperature = nn.Parameter(
            torch.tensor([temperature]).log(),
            requires_grad=learnable_temperature,
        )
        self.bias = nn.Parameter(
            torch.tensor([bias]).float(), requires_grad=learnable_temperature
        )
        self.num_classes = int(num_classes)
        self.force_all_gather = bool(force_all_gather)
        # SigLIP-style options
        self.symmetric = bool(symmetric)

        self.aux_cos_loss_weight = aux_cos_loss_weight
        self.aux_norm_loss_weight = aux_norm_loss_weight

    @torch.compiler.disable
    def forward(
        self,
        node_embeddings: torch.Tensor | None,
        graph_embeddings: torch.Tensor,
        ys: Mapping[str, torch.Tensor],
        batch_metrics: dict[str, Metric | MetricCollection] | None = None,
        aux_loss: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the SigLIP-style intra-modality contrastive loss.

        Args:
            node_embeddings: The embedding for each node in the batch. Shape
                (B, C, D), where C is the maximum number of comments in a
                discussion. Unused.
            graph_embeddings: The embedding for each graph in the batch. Shape
                (B, D).
            ys: Dictionary for the labels in the batch. Within that dictionary,
                this loss uses
                - y: Integer class label per graph. Shape (B,). Entries with
                  value -100 are ignored (padding) and masked out of the loss.
                - hard_y: Optional hard-negative label per graph. Shape (B,).
                  Used to down-weight soft negatives; can be omitted.
            batch_metrics:
                A dictionary of metric objects to update with the batch metrics.
                This should be called with "compute_batch_metrics". If None, no
                metrics will be computed and only the loss will be returned.

        Returns:
            A tuple of (loss, metrics), where loss is a scalar tensor and
            metrics is a dict[str, Tensor] containing per-batch values such as
            loss, weight, bias, temperature, and classification metrics.
        """
        del node_embeddings

        # compute similarity matrix for contrastive loss
        graph_embeddings = graph_embeddings.to(torch.float32)
        normalized_A = F.normalize(graph_embeddings, p=2, dim=-1)

        # Support enum or string keys for labels
        device_targets = (
            ys.get(ContrastiveLabels.Ys)  # type: ignore[arg-type]
            if hasattr(ys, "get")
            else None
        )
        if device_targets is None:
            device_targets = ys.get("y")  # type: ignore[index]
        if device_targets is None:
            raise ValueError("Targets 'y' not found in ys mapping.")

        device_hard_targets = (
            ys.get(ContrastiveLabels.HardYs)  # type: ignore[arg-type]
            if hasattr(ys, "get")
            else None
        )
        if device_hard_targets is None:
            device_hard_targets = ys.get("hard_y")  # type: ignore[index]
        if device_hard_targets is None:
            device_hard_targets = torch.full_like(device_targets, -100)
        use_all_gather = (
            self.force_all_gather or torch.distributed.is_initialized()
        )
        # use_all_gather = False

        if use_all_gather:
            all_targets, all_hard_targets, all_graph_embeddings = (
                self.all_gather(x)
                for x in (device_targets, device_hard_targets, normalized_A)
            )
            assert isinstance(all_graph_embeddings, torch.Tensor)
            graph_shape = graph_embeddings.shape
            targets, hard_targets, graph_embeddings = (
                all_targets.reshape(-1),
                all_hard_targets.reshape(-1),
                all_graph_embeddings.reshape(-1, graph_shape[-1]),
            )
        else:
            targets, hard_targets = device_targets, device_hard_targets
            graph_embeddings = normalized_A

        # scaling factor
        sim = torch.matmul(graph_embeddings, graph_embeddings.t())
        sim = sim * self.temperature.exp()  # + self.bias

        # Targets is an array of int labels, discussions sharing the same label
        # are from the same community/topic

        padding = targets == -100

        padding_mask = padding.repeat(targets.shape[0], 1)
        padding_mask = padding_mask | padding_mask.t()
        # Format y into a n x n matrix where n is the number of graphs and each
        # row has 1 for the correct label and 0 for the rest

        target_matrix = targets.unsqueeze(1).eq(targets).float()

        # Remove padding from the loss, and any reweighting
        target_matrix[padding_mask] = -1
        target_matrix = target_matrix.fill_diagonal_(-1)

        # Same as targets, but for hard negatives
        hard_target_matrix = hard_targets.unsqueeze(1).eq(targets).float()
        hard_target_matrix[padding_mask] = -1

        soft_labels = torch.logical_and(
            target_matrix.eq(0), hard_target_matrix.eq(0)
        )

        num_hard_labels = (
            torch.logical_or(target_matrix.eq(1), hard_target_matrix.eq(1))
        ).sum(dim=1)
        num_hard_labels = torch.clamp(num_hard_labels, min=1)

        if self.adaptive_soft_negative_weight:
            # soft_negs are proportionally weighted to the number of hard_negs
            # and hard_pos
            extra_weight = num_hard_labels / torch.clamp(
                soft_labels.sum(dim=1), min=1
            )
            extra_weight = extra_weight.reshape(-1, 1)
        else:
            extra_weight = self.soft_negative_weight

        # compute loss weights. Hard labels are given 1 weight, soft labels
        # are given extra_weight

        soft_matrix = torch.where(soft_labels, extra_weight, 1.0)
        # Since we do intra-modality contrastive loss, remove diagonal from
        # loss matrix. We don't want to include itself in the loss

        soft_matrix = soft_matrix.fill_diagonal_(0)
        # Set the weight for padding entries to 0
        soft_matrix[padding_mask] = 0
        # For numerical stability, set this to 0
        target_matrix[padding_mask] = 0
        target_matrix = target_matrix.fill_diagonal_(0)

        # Map 0, 1 labels to -1, 1 so we can use logsigmoid for both pos/neg
        target_pm = (target_matrix * 2) - 1
        pair_log_probs = F.logsigmoid(sim * target_pm)

        # Pairwise weighted BCE-style objective. Directional reductions ensure
        # row-wise and column-wise terms are not redundant.
        pair_loss = -pair_log_probs  # shape [N, N]
        weights = soft_matrix  # shape [N, N]

        # Row-wise mean: for each i, mean over j of loss[i, j] weighted by w[i, j]
        row_weight_sums = weights.sum(dim=1)
        row_denoms = torch.clamp(row_weight_sums, min=1.0)
        row_means = (pair_loss * weights).sum(dim=1) / row_denoms
        valid_rows = row_weight_sums > 0
        loss_rows = (
            row_means[valid_rows].mean()
            if valid_rows.any()
            else row_means.mean()
        )

        if self.symmetric:
            # Column-wise mean: for each j,
            # mean over i of loss[i, j] weighted by w[i, j]
            col_weight_sums = weights.sum(dim=0)
            col_denoms = torch.clamp(col_weight_sums, min=1.0)
            col_means = (pair_loss * weights).sum(dim=0) / col_denoms
            valid_cols = col_weight_sums > 0
            loss_cols = (
                col_means[valid_cols].mean()
                if valid_cols.any()
                else col_means.mean()
            )
            loss = 0.5 * (loss_rows + loss_cols)
        else:
            loss = loss_rows

        if aux_loss is not None:
            norm_loss = self.aux_norm_loss_weight * aux_loss.get(
                "norm_loss", 0.0
            )
            cos_loss = self.aux_cos_loss_weight * aux_loss.get("cos_loss", 0.0)
            total_loss = loss + norm_loss + cos_loss
            # logger.warning(
            #     f"TOTAL LOSS: {loss=} {norm_loss=} {cos_loss=} "
            #     f" ** {total_loss=} **"
            # )
        else:
            total_loss = loss

        # Apply padding mask to only include non-padded examples

        # Final scaling for multi-GPU training
        # loss = loss / num_gpus  # To control gradients

        if batch_metrics is not None:
            with torch.no_grad():
                metric_sim = sim.detach()
                metric_sim = metric_sim.fill_diagonal_(-1e9)

                metric_sim[padding_mask] = -1e9

                batch_metrics_out = self.compute_batch_metrics(
                    metric_sim,
                    targets.detach(),
                    total_loss.detach(),
                    batch_metrics,
                )
                # Ensure return type is dict[str, Tensor]
                metrics = {
                    k: v
                    for k, v in batch_metrics_out.items()
                    if isinstance(v, torch.Tensor)
                }
                metrics["original_loss"] = loss.detach()
                if aux_loss is not None:
                    for k, v in aux_loss.items():
                        metrics[f"aux_{k}"] = v.detach()
        else:
            metrics = {}
        return total_loss, metrics


class ContrastiveLossWithMetrics(ContrastiveLoss):
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
        return_metrics["bias"] = self.bias
        return_metrics["temperature"] = self.temperature.exp()

        return return_metrics
