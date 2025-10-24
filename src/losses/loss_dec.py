"""Deep Embedding Clustering (DEC) loss.

Implementation follows the original DEC paper:
  Xie, Girshick, Farhadi. "Unsupervised Deep Embedding for Clustering Analysis"
  (ICML 2016).

Given latent embeddings z_i and learnable cluster centers mu_j, we compute
soft assignments with a (normalized) Student's t-distribution (alpha defaults
to 1.0) and minimize the KL divergence between an auxiliary target distribution
and the current soft assignments, sharpening cluster assignments over time.

Loss:  L = KL(P || Q) = sum_i sum_j p_ij * log(p_ij / q_ij)

Where:
  q_ij = (1 + ||z_i - mu_j||^2 / alpha)^(-(alpha+1)/2) normalized over j
  p_ij = (q_ij^2 / f_j) / sum_k (q_ik^2 / f_k) with f_j = sum_i q_ij

This module optionally computes supervised classification metrics when ground
truth labels (key 'y' in ys) are provided (semi-supervised evaluation), but the
loss itself is unsupervised.
"""

from __future__ import annotations

from typing import Mapping, Literal, Optional

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

from losses.loss_abstract import Loss, LastValueMetric
from utils import RankedLogger

logger = RankedLogger(__name__)


class DeclusteringLoss(Loss):
    """Deep Embedding Clustering (DEC) loss.

    Args:
        num_clusters: Number of clusters (K).
        embedding_dim: Dimension of each embedding feature (D).
        alpha: Degrees of freedom for Student's t-distribution (usually 1.0).
        num_classes: (Optional) Number of ground-truth classes. If provided and
            labels ('y') appear in `ys`, classification metrics will be logged.
        force_all_gather: Force usage of all_gather even if torch.distributed
            is not initialized (mirrors other loss implementations).

    Notes:
        - Cluster centers are randomly initialized from N(0,1). You can call
          `update_centers(tensor)` (e.g., after a KMeans init) before training.
        - The loss is unsupervised; labels are only used for metric reporting.
    """

    def __init__(
        self,
        num_clusters: int,
        embedding_dim: int,
        alpha: float = 1.0,
        num_classes: Optional[int] = None,
        force_all_gather: bool = False,
    ) -> None:
        super().__init__()
        self.num_clusters = num_clusters
        self.embedding_dim = embedding_dim
        self.alpha = alpha
        self.force_all_gather = force_all_gather
        self.num_classes = num_classes

        # Learnable cluster centers μ_j
        self.cluster_centers = nn.Parameter(
            torch.randn(num_clusters, embedding_dim)
        )

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def update_centers(self, centers: torch.Tensor) -> None:
        """Replace the current cluster centers.

        Args:
            centers: Tensor of shape (K, D)
        """
        assert centers.shape == self.cluster_centers.shape, (
            f"Expected centers shape {self.cluster_centers.shape}, got"
            f" {centers.shape}"
        )
        self.cluster_centers.data.copy_(centers.to(self.cluster_centers.device))

    # ---------------------------------------------------------------------
    # Core DEC computations
    # ---------------------------------------------------------------------
    def _soft_assign(self, z: torch.Tensor) -> torch.Tensor:
        """Compute soft assignments q_ij using Student's t-distribution.

        Args:
            z: Embeddings (B, D)
        Returns:
            q: Soft assignments (B, K)
        """
        # Squared Euclidean distance between embeddings and cluster centers
        # (B, K)
        dist_sq = torch.cdist(z, self.cluster_centers) ** 2
        # Student t-kernel (α defaults to 1): (1 + d^2/α)^(-(α+1)/2)
        numerator = (1.0 + dist_sq / self.alpha) ** (-(self.alpha + 1.0) / 2.0)
        q = numerator / numerator.sum(dim=1, keepdim=True)
        return q.clamp_min(1e-10)

    def _target_distribution(self, q: torch.Tensor) -> torch.Tensor:
        """Compute DEC target distribution p."""

        p_num = (q**2) / q.sum(dim=0, keepdim=True)
        p = p_num / p_num.sum(dim=1, keepdim=True)
        return p.clamp_min(1e-10)

    # ---------------------------------------------------------------------
    # Metrics interface
    # ---------------------------------------------------------------------
    def build_batch_metric_aggregators(
        self,
    ) -> Mapping[str, Metric | MetricCollection]:
        """Build metric collectors. Includes classification metrics if
        `num_classes` is provided (semi-supervised evaluation)."""

        metrics: dict[str, Metric | MetricCollection] = {
            "loss": LastValueMetric()
        }

        if self.num_classes is not None:

            def make_metric_group(
                average: Literal["micro", "macro", "weighted"],
            ) -> MetricCollection:
                return MetricCollection(
                    (
                        {
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
                            {
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

            metrics["classification"] = MetricCollection(
                [
                    make_metric_group("macro"),  # type: ignore
                    make_metric_group("weighted"),  # type: ignore
                    make_metric_group("micro"),  # type: ignore
                ]
            )
        return metrics

    def compute_batch_metrics(
        self,
        logits: torch.Tensor,  # (B, K) soft assignments q
        targets: torch.Tensor,  # (B,) ground truth labels (optional / -100)
        loss: torch.Tensor,
        metrics: dict[str, Metric | MetricCollection],
    ) -> Mapping[str, torch.Tensor | Metric]:
        batch_size, _ = logits.shape
        assert targets.shape == (
            batch_size,
        ), f"Unexpected shape: {targets.shape=} vs (B,)"

        preds = logits.argmax(dim=1)

        return_metrics: dict[str, torch.Tensor | Metric] = {}

        # Update supervised metrics only if classification metrics available
        if "classification" in metrics:
            metrics["classification"].update(preds, targets)
        metrics["loss"].update(loss)

        return_metrics["loss"] = loss.detach()
        effective_batch_size = (targets != -100).float().sum()
        return_metrics["weight"] = effective_batch_size
        return return_metrics

    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------
    def forward(
        self,
        node_embeddings: torch.Tensor | None,
        graph_embeddings: torch.Tensor,
        ys: Mapping[str, torch.Tensor],
        batch_metrics: dict[str, Metric | MetricCollection] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute DEC loss over a batch of graph (or instance) embeddings.

        Args:
            node_embeddings: Optional per-node embeddings (unused).
            graph_embeddings: Tensor (B, D) latent embeddings.
            ys: Mapping potentially containing a key 'y' with ground-truth
                labels (shape (B,)) used only for metrics (ignore_index=-100).
            batch_metrics: Metric aggregators (optional). If provided, metrics
                are updated and returned.
        Returns:
            loss (scalar), metrics dict
        """
        del node_embeddings

        z = graph_embeddings
        use_all_gather = (
            self.force_all_gather or torch.distributed.is_initialized()
        )

        if use_all_gather:
            all_embeddings = self.all_gather(z)
            assert isinstance(all_embeddings, torch.Tensor)
            z_shape = z.shape
            z = all_embeddings.reshape(-1, z_shape[-1])

            if "y" in ys:
                all_targets = self.all_gather(ys["y"])
                targets = all_targets.reshape(-1)
            else:
                targets = torch.full(
                    (z.shape[0],), -100, dtype=torch.long, device=z.device
                )
        else:
            targets = (
                ys.get("y")
                if "y" in ys
                else torch.full(
                    (z.shape[0],), -100, dtype=torch.long, device=z.device
                )
            )

        # Soft assignments Q
        q = self._soft_assign(z)
        # Target distribution P
        p = self._target_distribution(q)

        # KL(P || Q)
        log_q = torch.log(q)
        loss = F.kl_div(log_q, p, reduction="batchmean")

        if batch_metrics is not None:
            with torch.no_grad():
                metrics_out = self.compute_batch_metrics(
                    q.detach(), targets.detach(), loss.detach(), batch_metrics
                )
        else:
            metrics_out = {}

        return loss, metrics_out

    # ------------------------------------------------------------------
    # Convenience utilities
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_clusters(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Return hard cluster assignments (argmax over soft assignments)."""
        q = self._soft_assign(embeddings)
        return q.argmax(dim=1)
