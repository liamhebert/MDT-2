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
from torchmetrics import MaxMetric
from torchmetrics import MeanMetric
from torchmetrics import MinMetric
from torchmetrics import Precision
from torchmetrics import Recall
from typing import Literal, Mapping


class ContrastiveLoss(Loss):
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
        soft_negative_weight: float = 0.0,
        adaptive_soft_negative_weight: bool = False,
        temperature: float = 10.0,
        bias: float = -10.0,
        learnable_temperature: bool = True,
        force_all_gather: bool = False,
    ):
        """Initializes the contrastive loss.

        Args:
            num_classes (int): The number of classes in the dataset,
                used for the classification metrics.
            soft_negative_weight (float, optional): Weight to associate to soft
                negative pairs in the contrastive loss. Flag is exclusive against
                adaptive_soft_negative_weight
            adaptive_soft_negative_weight (bool, optional): Whether to adapt
                the soft negative weight based on the number of positive pairs
                and negative pairs. Flag is exclusive against soft_negative_weight
            temperature (float, optional): The temperature to use for the softmax
                function. A higher value will make the distribution more uniform,
                while a lower value will make the distribution more peaky.
                Defaults to 10.
            bias (float): The bias to add to the similarity matrix. Defaults to
                -10.
            learnable_temperature (bool, optional): Whether to learn the
                temperature parameter. The initial value of the temperature will
                be `temperature`. Defaults to False.
        """
        super().__init__()

        assert isinstance(soft_negative_weight, float)
        self.soft_negative_weight = torch.Tensor([soft_negative_weight])
        self.adaptive_soft_negative_weight = adaptive_soft_negative_weight
        self.temperature = nn.Parameter(
            torch.tensor([temperature]).log(),
            requires_grad=learnable_temperature,
        )
        self.bias = nn.Parameter(
            torch.tensor([bias]).float(), requires_grad=learnable_temperature
        )
        self.num_classes = num_classes
        self.use_all_gather = (
            torch.distributed.is_initialized() or force_all_gather
        )
        print("USE ALL GATHER", self.use_all_gather)

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
        normalized_A = F.normalize(graph_embeddings, p=2, dim=-1)

        device_targets = ys[ContrastiveLabels.Ys]
        device_hard_targets = ys[ContrastiveLabels.HardYs]

        if self.use_all_gather:
            print("ALL GATHER", flush=True)
            print(
                device_targets.shape,
                device_hard_targets.shape,
                normalized_A.shape,
                flush=True,
            )

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
        sim = sim * self.temperature.exp() + self.bias

        # Targets is an array of int labels, discussions sharing the same label
        # are from the same community/topic

        padding = targets == -100

        padding_mask = padding.repeat(targets.shape[0], 1)
        padding_mask = padding_mask | padding_mask.t()
        # Format y into a n x n matrix where n is the number of graphs and each
        # row has 1 for the correct label and 0 for the rest

        target_matrix = targets.unsqueeze(1).eq(targets).float()

        # Remove padding from the loss, and any reweighting
        target_matrix[padding_mask] = -100
        target_matrix = target_matrix.fill_diagonal_(-100)

        # Same as targets, but for hard negatives
        hard_target_metrix = hard_targets.unsqueeze(1).eq(targets).float()
        hard_target_metrix[padding_mask] = -100

        soft_labels = torch.logical_and(
            target_matrix.eq(0), hard_target_metrix.eq(0)
        )

        if self.adaptive_soft_negative_weight:
            # soft_negs are proportionally weighted to the number of hard_negs
            # and hard_pos
            # num_hard_labels = (
            #     torch.logical_or(target_matrix.eq(1), hard_target_metrix.eq(1))
            # ).sum(dim=1)
            # print("NUM_HARD_LABELS", num_hard_labels)
            # num_hard_labels = torch.clamp(num_hard_labels, min=1)
            extra_weight = 1 / torch.clamp(soft_labels.sum(dim=1), min=1)
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

        # Normalize the weights to sum to the number of valid labels
        soft_matrix = F.normalize(soft_matrix, p=1, dim=1)

        if (
            self.adaptive_soft_negative_weight
            or (self.soft_negative_weight != 0).all()
        ):
            num_valid_labels = torch.clamp((~padding).sum(), min=1)
            num_valid_labels = num_valid_labels - 1
        else:
            num_valid_labels = torch.clamp(
                (soft_matrix != 0).int().sum(dim=0), min=1
            )

        soft_matrix = soft_matrix * (num_valid_labels)

        # compute loss
        # Map 0, 1 labels to -1, 1
        target_matrix = (target_matrix * 2) - 1

        loss = torch.einsum(
            "ij, ij -> i", -F.logsigmoid(sim * target_matrix), soft_matrix
        )

        if (
            self.adaptive_soft_negative_weight
            or (self.soft_negative_weight != 0).all()
        ):
            loss = loss.sum() / (num_valid_labels + 1)
        else:
            loss = loss / num_valid_labels

            loss = loss.sum()

        # Since the loss will eventually be all_gathered summed anyway.
        loss = loss

        if batch_metrics is not None:
            with torch.no_grad():
                metric_sim = sim.clone().detach()
                pred_sim = metric_sim.fill_diagonal_(-1e9)

                pred_sim[padding_mask] = -1e9

                metrics = self.compute_batch_metrics(
                    metric_sim,
                    targets.clone().detach(),
                    loss.clone().detach(),
                    batch_metrics,
                )
            # metrics = {"weight": sim.shape[0]}
        else:
            metrics = {}
        print(loss)
        return loss, metrics


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
            average: Literal["micro", "macro", "weighted", "none"],
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
                    make_metric_group("none"),  # type: ignore
                ]
            ),
            "loss": MeanMetric(),
        }

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
            for metric_type in ["macro", "weighted"]
            + [f"class_{i}" for i in range(self.num_classes)]
            for metric in ["f1", "recall", "precision", "accuracy"]
        } | {
            "best_loss": MinMetric(),
        }

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

        metrics["classification"].to(preds.device).update(preds, targets)
        metrics["loss"].to(loss.device).update(loss)

        return_metrics["loss"] = loss.item()

        effective_batch_size = (targets != -100).sum()
        return_metrics["weight"] = effective_batch_size.item()
        return_metrics["bias"] = self.bias.item()
        return_metrics["temperature"] = self.temperature.exp().item()

        return return_metrics

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

        ret_metrics = {}

        device = batch_metrics["classification"]["macro_f1"].device
        assert isinstance(device, torch.device)
        metrics = batch_metrics["classification"].to(device).compute()
        for metric_type in ["macro", "weighted"]:
            metric_ids = ["f1", "recall", "precision"]
            if metric_type == "weighted":
                metric_ids.append("accuracy")
            for metric in metric_ids:
                key = f"best_{metric_type}_{metric}"

                metric_value = metrics[f"{metric_type}_{metric}"]
                ret_metrics[f"{metric_type}_{metric}"] = metric_value
                ret_metrics[key] = (
                    epoch_metrics[key].to(device).forward(metric_value)
                )

        for metric in ["f1", "recall", "precision"]:
            metric_values = metrics["none_" + metric]
            for class_id in range(self.num_classes):
                key = f"best_class_{class_id}_{metric}"
                ret_metrics[f"class_{class_id}_{metric}"] = metric_values[
                    class_id
                ]
                ret_metrics[key] = (
                    epoch_metrics[key]
                    .to(device)
                    .forward(metric_values[class_id])
                )

        ret_metrics["best_loss"] = (
            epoch_metrics["best_loss"]
            .to(device)
            .forward(batch_metrics["loss"].compute())
        )

        for metric_obj in batch_metrics.values():
            metric_obj.reset()

        return ret_metrics
