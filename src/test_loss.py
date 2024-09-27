"""Tests for the loss functions."""
import pytest
import torch

from loss import NodeCrossEntropyLoss


class TestCrossEntropyLoss:
    """Tests for the cross-entropy loss function."""

    def test_compute_batch_metrics(self):
        """Test to check if we can comput e."""
        loss = NodeCrossEntropyLoss(1, 1)

        metric_agg = loss.build_batch_metric_aggregators()

        logits = torch.tensor([[0.1, 0.9], [0.9, 0.1]])
        targets = torch.tensor([1, 0])

        loss.compute_batch_metrics(logits, targets, metric_agg)

        for key, metric in metric_agg:
            print(key, metric.compute())
