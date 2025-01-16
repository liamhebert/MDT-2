"""
Tests for the loss functions.
"""

import pytest
import torch
from torch.nn import functional as F
import torch.nn as nn
import torch.testing as test

from components.output_head import SimpleOutputHead
from data.types import Labels
from losses.loss_cross import NodeCrossEntropyLoss


class IdentityOutputHead(SimpleOutputHead):
    """An output head that returns the input as the output.

    Useful for deterministic testing.
    """

    def __init__(self, input_dim: int, output_dim: int, **args):
        """
        Construct an `IdentityOutputHead`, swapping the model to identity.
        """
        super().__init__(input_dim, output_dim, **args)
        self.model = nn.Identity()
        self.input_dim = input_dim
        self.output_dim = output_dim


class TestCrossEntropyLoss:
    """
    Tests for the cross-entropy loss function.
    """

    def test_compute_batch_metrics(self):
        """
        Test to check if we can compute.
        """
        loss = NodeCrossEntropyLoss([1, 1], IdentityOutputHead(2, 2))

        metric_agg = loss.build_batch_metric_aggregators()

        logits = torch.tensor([[0.1, 0.9], [0.1, 0.9], [0.1, 0.9], [0.5, 0.5]])
        targets = torch.tensor([1, 0, 1, -100])
        loss_val = F.cross_entropy(
            logits,
            targets,
            weight=torch.Tensor([1.0, 1.0]),
            reduction="none",
            ignore_index=-100,
        )

        returned_metrics = loss.compute_batch_metrics(
            logits, targets, loss_val, metric_agg
        )

        test.assert_close(returned_metrics["loss"], loss_val)
        del returned_metrics["loss"]

        test.assert_close(metric_agg["loss"].compute(), loss_val.mean())

        # Check that the metrics are updated correctly
        metrics = metric_agg["classification"].compute()
        metrics["weight"] = 3

        test.assert_close(metrics["none_recall"], torch.Tensor([0.0, 1.0]))
        test.assert_close(
            metrics["none_precision"], torch.Tensor([0.0, 0.66667])
        )
        test.assert_close(metrics["none_f1"], torch.Tensor([0.0, 0.8]))

        test.assert_close(metrics["macro_recall"], torch.tensor(0.50))
        test.assert_close(metrics["macro_precision"], torch.tensor(0.33333))
        test.assert_close(metrics["macro_f1"], torch.tensor(0.40))
        test.assert_close(metrics["macro_accuracy"], torch.tensor(0.5))

        test.assert_close(metrics["weighted_recall"], torch.tensor(0.666667))
        test.assert_close(metrics["weighted_precision"], torch.tensor(0.444444))
        test.assert_close(metrics["weighted_f1"], torch.tensor(0.53333))
        test.assert_close(metrics["weighted_accuracy"], torch.tensor(0.66667))

        # Formatting the metrics to match each other
        for metric in ["f1", "precision", "recall"]:
            metrics["class_0_" + metric] = metrics["none_" + metric][0]
            metrics["class_1_" + metric] = metrics["none_" + metric][1]
            del metrics["none_" + metric]

        # Test to ensure that the logged metrics match the computed metrics and
        # that all expected metrics are there.
        assert metrics == returned_metrics

    def test_compute_epoch_metrics(self):
        """
        Test to check if we can correctly aggregate compute epoch metrics.
        """
        loss = NodeCrossEntropyLoss([1, 1], IdentityOutputHead(2, 2))

        batch_agg = loss.build_batch_metric_aggregators()
        epoch_agg = loss.build_epoch_metric_aggregators()

        # Best performing batch
        logits = torch.tensor([[0.1, 0.9], [0.1, 0.9], [0.1, 0.9], [0.5, 0.5]])
        targets = torch.tensor([1, 0, 1, -100])
        good_loss = F.cross_entropy(
            logits,
            targets,
            weight=torch.Tensor([1.0, 1.0]),
            reduction="none",
            ignore_index=-100,
        )

        # Bad batch
        logits_bad = torch.tensor([[0.1, 0.9], [0.1, 0.9], [0.1, 0.9]])
        targets_bad = torch.tensor([0, 1, 0])
        bad_loss = F.cross_entropy(
            logits_bad,
            targets_bad,
            weight=torch.Tensor([1.0, 1.0]),
            reduction="none",
        )

        for _ in range(10):
            # Simulating 3 batch epoch
            for _ in range(3):
                loss.compute_batch_metrics(
                    logits_bad, targets_bad, bad_loss, batch_agg
                )
            loss.compute_epoch_metrics(batch_agg, epoch_agg)

        for _ in range(3):
            loss.compute_batch_metrics(logits, targets, good_loss, batch_agg)
        loss.compute_epoch_metrics(batch_agg, epoch_agg)

        test.assert_close(epoch_agg["best_loss"].compute(), good_loss.mean())

        test.assert_close(
            epoch_agg["best_class_0_recall"].compute(), torch.tensor(0.0)
        )
        test.assert_close(
            epoch_agg["best_class_0_precision"].compute(), torch.tensor(0.0)
        )
        test.assert_close(
            epoch_agg["best_class_0_f1"].compute(), torch.tensor(0.0)
        )

        test.assert_close(
            epoch_agg["best_class_1_recall"].compute(), torch.tensor(1.0)
        )
        test.assert_close(
            epoch_agg["best_class_1_precision"].compute(), torch.tensor(0.66667)
        )
        test.assert_close(
            epoch_agg["best_class_1_f1"].compute(), torch.tensor(0.8)
        )

        test.assert_close(
            epoch_agg["best_macro_recall"].compute(), torch.tensor(0.50)
        )
        test.assert_close(
            epoch_agg["best_macro_precision"].compute(), torch.tensor(0.33333)
        )
        test.assert_close(
            epoch_agg["best_macro_f1"].compute(), torch.tensor(0.40)
        )

        test.assert_close(
            epoch_agg["best_weighted_recall"].compute(), torch.tensor(0.666667)
        )
        test.assert_close(
            epoch_agg["best_weighted_precision"].compute(),
            torch.tensor(0.444444),
        )
        test.assert_close(
            epoch_agg["best_weighted_f1"].compute(), torch.tensor(0.53333)
        )

    @pytest.mark.parametrize("weights", [[1.0, 1.0], [0.5, 0.3]])
    def test_call(self, weights: list[float]):
        """
        Test to check if we can compute the loss.
        """
        loss = NodeCrossEntropyLoss(weights, IdentityOutputHead(2, 2))

        logits = torch.tensor([[0.1, 0.9], [0.1, 0.9], [0.1, 0.9], [0.5, 0.5]])
        targets = torch.tensor([1, 0, 1, -100])

        formatted_targets = {
            Labels.Ys: targets,
        }

        loss_val, metrics = loss(logits, None, formatted_targets)

        logits = logits.reshape((-1, 2))[:-1, :]
        targets = targets.flatten()[:-1]
        expected_loss = F.cross_entropy(
            logits,
            targets,
            weight=torch.tensor(weights, requires_grad=False),
            reduction="mean",
        )
        test.assert_close(loss_val, expected_loss)
