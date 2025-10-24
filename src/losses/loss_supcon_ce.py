"""Combined supervised cross-entropy + supervised contrastive (InfoNCE) loss.

This loss wraps an encoder's graph embeddings with a small MLP classifier
and applies:
  1. Primary supervised classification loss (CrossEntropy ignoring -100)
  2. Auxiliary supervised contrastive loss (multi-positive InfoNCE) using
     the `SupContrastiveLoss` logic (temperatured cosine similarities).

Total:  L = L_ce + lambda_contrast * L_supcon

Intended usage: strengthen intra-class compactness and inter-class
separation while preserving a directly optimized classifier head.
"""

from __future__ import annotations

from typing import Mapping, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import (
    Metric,
    MetricCollection,
    Accuracy,
    Precision,
    Recall,
    F1Score,
)

from losses.loss_abstract import Loss
from losses.loss_sup_contrastive import SupContrastiveLossWithMetrics
from data.types import ContrastiveLabels


class SupConWithCELoss(Loss):
    """Supervised classification + supervised contrastive auxiliary loss.

    Args:
        embedding_dim: Dimension of input embeddings.
        num_classes: Number of target classes.
        contrastive_temperature: Initial temperature for SupCon branch.
        lambda_contrastive: Weight on auxiliary contrastive loss.
        projection_dim: Optional projection head output dim for contrastive;
            if None uses embedding directly.
        learnable_temperature: Whether SupCon temperature is learnable.
        force_all_gather: Force DDP all_gather for embeddings/labels.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        temperature: float = 0.05,
        lambda_contrastive: float = 0.1,
        projection_dim: int | None = None,
        learnable_temperature: bool = True,
        force_all_gather: bool = False,
        aux_cos_loss_weight: float = 1.0,
        aux_norm_loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.lambda_contrastive = lambda_contrastive
        self.force_all_gather = force_all_gather

        # Projection head (optional) for contrastive branch
        if projection_dim is not None:
            self.projection = nn.Sequential(
                nn.Linear(embedding_dim, projection_dim, bias=True),
                nn.GELU(),
                nn.Linear(projection_dim, projection_dim, bias=False),
            )
            self.classifier = nn.Sequential(
                nn.Linear(embedding_dim, projection_dim, bias=True),
                nn.GELU(),
                nn.Linear(projection_dim, num_classes, bias=True),
            )
        else:
            self.projection = nn.Identity()
            self.classifier = nn.Linear(embedding_dim, num_classes)

        # Reuse SupContrastiveLoss logic (temperature parameterization & masking)
        self.supcon = SupContrastiveLossWithMetrics(
            num_classes=num_classes,
            temperature=temperature,
            learnable_temperature=learnable_temperature,
            force_all_gather=force_all_gather,
        )

        self.aux_cos_loss_weight = aux_cos_loss_weight
        self.aux_norm_loss_weight = aux_norm_loss_weight

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def build_batch_metric_aggregators(
        self,
    ) -> Mapping[str, Metric | MetricCollection]:
        def make_metric_group(
            avg: Literal["micro", "macro", "weighted"],
        ) -> MetricCollection:
            metrics: dict[str, Metric] = {
                "recall": Recall(
                    task="multiclass",
                    num_classes=self.num_classes,
                    average=avg,
                    ignore_index=-100,
                ),
                "precision": Precision(
                    task="multiclass",
                    num_classes=self.num_classes,
                    average=avg,
                    ignore_index=-100,
                ),
                "f1": F1Score(
                    task="multiclass",
                    num_classes=self.num_classes,
                    average=avg,
                    ignore_index=-100,
                ),
            }
            if avg == "weighted":
                metrics["accuracy"] = Accuracy(
                    task="multiclass",
                    num_classes=self.num_classes,
                    average=avg,
                    ignore_index=-100,
                )
            return MetricCollection(metrics, prefix=f"{avg}_")

        classification = MetricCollection(
            [
                make_metric_group("macro"),  # type: ignore
                make_metric_group("weighted"),  # type: ignore
                make_metric_group("micro"),  # type: ignore
            ]
        )
        return {"classification": classification}

    def compute_batch_metrics(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        loss: torch.Tensor,
        metrics: dict[str, Metric | MetricCollection],
    ) -> Mapping[str, torch.Tensor | Metric]:
        batch_size, _ = logits.shape
        assert targets.shape == (batch_size,), "Targets shape mismatch"
        preds = logits.argmax(dim=1)
        metrics["classification"].update(preds, targets)
        effective = (targets != -100).float().sum()
        return {"loss": loss.detach(), "weight": effective}

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        node_embeddings: torch.Tensor | None,
        graph_embeddings: torch.Tensor,
        ys: Mapping[str, torch.Tensor],
        batch_metrics: dict[str, Metric | MetricCollection] | None = None,
        aux_loss: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del node_embeddings

        targets = ys.get(ContrastiveLabels.Ys)
        if targets is None:
            raise ValueError(
                "'y' key with class labels required for SupConWithCELoss"
            )

        # DDP gather if requested
        use_all_gather = (
            self.force_all_gather or torch.distributed.is_initialized()
        )
        emb = graph_embeddings
        if use_all_gather:
            gathered_emb = self.all_gather(emb)
            gathered_tgt = self.all_gather(targets)
            assert isinstance(gathered_emb, torch.Tensor)
            emb_shape = emb.shape
            emb = gathered_emb.reshape(-1, emb_shape[-1])
            targets = gathered_tgt.reshape(-1)

        # Classifier forward (detach not applied so gradients flow from CE)
        logits = self.classifier(emb)
        ce_loss = F.cross_entropy(logits, targets.long(), ignore_index=-100)

        # Contrastive branch (optionally projection)
        z_con = self.projection(emb)
        # Translate plain label key to contrastive label enum mapping expected
        # by SupContrastiveLoss
        supcon_loss, _ = self.supcon(None, z_con, ys, None)

        loss = ce_loss + self.lambda_contrastive * supcon_loss
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
                raw = self.compute_batch_metrics(
                    logits.detach(),
                    targets.detach(),
                    total_loss.detach(),
                    batch_metrics,
                )
                metrics_out: dict[str, torch.Tensor] = {
                    k: v for k, v in raw.items() if isinstance(v, torch.Tensor)
                }
                metrics_out["ce_loss"] = ce_loss.detach()
                metrics_out["supcon_loss"] = supcon_loss.detach()
                metrics_out["original_loss"] = loss.detach()
                if aux_loss is not None:
                    for k, v in aux_loss.items():
                        metrics_out[f"aux_{k}"] = v.detach()
        else:
            metrics_out = {}
        return total_loss, metrics_out

    @torch.no_grad()
    def predict(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.classifier(embeddings).argmax(dim=1)
