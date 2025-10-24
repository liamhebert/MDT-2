"""Tests for the the contrastive loss function."""

from losses import ContrastiveLoss

# from losses.new_loss_contrastive import (
#     ContrastiveLossWithMetrics as ContrastiveLoss,
# )
import torch
from data.types import ContrastiveLabels
import torch.nn.functional as F
import torch.testing as test
import pytest


def test_smoke_test():
    """Simple smoke test with duplicates.

    This test ensures that we can use a larger set of candidate embeddings,
    matching the batch size, and trigger the remove duplicates stream
    successfully.
    """
    loss = ContrastiveLoss(temperature=1.0, num_classes=10, bias=0.0)
    batch_metrics = loss.build_batch_metric_aggregators()

    node_x = torch.rand(10, 256)
    graph_x = torch.rand(10, 256)
    y_true = {
        ContrastiveLabels.Ys: torch.randint(0, 10, (10,)),
        ContrastiveLabels.HardYs: torch.randint(0, 10, (10,)),
    }
    loss_value, _ = loss(node_x, graph_x, y_true, batch_metrics)
    assert loss_value.shape == ()


def test_temperature():
    """Simple test to check the temperature parameter.

    This test ensures that we can use a temperature parameter, and that it
    changes the output of the loss function.
    """
    loss = ContrastiveLoss(
        temperature=2.0, learnable_temperature=False, num_classes=10, bias=0.0
    )
    batch_metrics = loss.build_batch_metric_aggregators()

    node_x = torch.rand(10, 256)
    graph_x = torch.rand(10, 256)
    y_true = {
        ContrastiveLabels.Ys: torch.randint(0, 10, (10,)),
        ContrastiveLabels.HardYs: torch.randint(0, 10, (10,)),
    }
    loss_value_high_temp, _ = loss(node_x, graph_x, y_true, batch_metrics)

    loss = ContrastiveLoss(
        temperature=0.5, learnable_temperature=False, num_classes=10, bias=0.0
    )
    node_x = torch.rand(10, 256)
    graph_x = torch.rand(10, 256)
    y_true = {
        ContrastiveLabels.Ys: torch.randint(0, 10, (10,)),
        ContrastiveLabels.HardYs: torch.randint(0, 10, (10,)),
    }
    loss_value_low_temp, _ = loss(node_x, graph_x, y_true, batch_metrics)

    assert loss_value_high_temp != loss_value_low_temp


def test_learnable_temperature():
    """Test to check the learnable temperature parameter.

    This test ensures that the temperature parameter is updated during training.
    """
    loss = ContrastiveLoss(
        temperature=1.0, learnable_temperature=True, num_classes=10, bias=1.0
    )
    batch_metrics = loss.build_batch_metric_aggregators()

    optimizer = torch.optim.SGD(loss.parameters(), lr=0.01)
    node_x = torch.rand(10, 256)
    graph_x = torch.rand(10, 256)
    y_true = {
        ContrastiveLabels.Ys: torch.randint(0, 10, (10,)),
        ContrastiveLabels.HardYs: torch.randint(0, 10, (10,)),
    }

    initial_temperature = loss.temperature.item()
    initial_bias = loss.bias.item()

    for _ in range(10):
        optimizer.zero_grad()
        loss_value, _ = loss(node_x, graph_x, y_true, batch_metrics)
        loss_value.backward()
        optimizer.step()

    updated_temperature = loss.temperature.item()
    updated_bias = loss.bias.item()
    assert initial_temperature != updated_temperature
    assert updated_bias != initial_bias


@pytest.mark.parametrize("weight", ["fixed", "adaptive", "none"])
def test_soft_negative_weight(weight):
    """Test to ensure the loss is calculated accurately without duplicates."""
    loss = ContrastiveLoss(
        temperature=1,
        learnable_temperature=False,
        num_classes=5,
        bias=0.0,
        adaptive_soft_negative_weight=(weight == "adaptive"),
        soft_negative_weight=0.3 if weight == "fixed" else 0.0,
        symmetric=False,
    )
    batch_metrics = loss.build_batch_metric_aggregators()
    node_x = None
    graph_x = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]
    )
    y_true = {
        ContrastiveLabels.Ys: torch.tensor([0, 1, 1.0, 3, 4, -100]),
        ContrastiveLabels.HardYs: torch.tensor([1, 0, 0, 2, 5, -100]),
    }

    loss_value, returned_metrics = loss(node_x, graph_x, y_true, batch_metrics)

    graph_x = F.normalize(graph_x[:-1], p=2, dim=1)
    expect_sim = torch.matmul(graph_x, graph_x.T)

    expect_ys = y_true[ContrastiveLabels.Ys][:-1]
    expect_labels = expect_ys.unsqueeze(1).eq(expect_ys).float()
    expect_labels = (expect_labels * 2) - 1

    # Each row must sum to the number of labels, therefore, 1.5 each.
    if weight == "adaptive":
        # Weights are 1 / num_soft_negatives
        weight_matrix = torch.tensor(
            [
                [0, 1.0, 1.0, 1.0, 1.0],  # 2 / 2 soft = 1
                [1.0, 0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 0, 1.0, 1.0],
                [0.25, 0.25, 0.25, 0.0, 0.25],  # 1 / 4 soft = 0.25
                [0.25, 0.25, 0.25, 0.25, 0.0],
            ]
        )
    elif weight == "fixed":
        # All soft negatives have a weight of 0.3
        weight_matrix = torch.tensor(
            [
                [0, 1.0, 1.0, 0.3, 0.3],
                [1.0, 0, 1.0, 0.3, 0.3],
                [1.0, 1.0, 0, 0.3, 0.3],
                [0.3, 0.3, 0.3, 0.0, 0.3],
                [0.3, 0.3, 0.3, 0.3, 0.0],
            ]
        )
    else:
        # No weight for soft negatives
        weight_matrix = torch.tensor(
            [
                [0, 1.0, 1.0, 0, 0],
                [1.0, 0, 1.0, 0, 0],
                [1.0, 1.0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )

    # Row-wise normalized means then mean over valid rows (matches implementation)
    pair_loss = -F.logsigmoid(expect_sim * expect_labels)
    row_weight_sums = weight_matrix.sum(dim=1)
    denoms = torch.clamp(row_weight_sums, min=1.0)
    row_means = (pair_loss * weight_matrix).sum(dim=1) / denoms
    valid_rows = row_weight_sums > 0
    expected_loss = row_means[valid_rows].mean()

    test.assert_close(loss_value, expected_loss)


def test_contrastive_loss_value():
    """Test to ensure the loss is calculated accurately without duplicates."""
    loss = ContrastiveLoss(
        temperature=1,
        learnable_temperature=False,
        num_classes=4,
        bias=0.0,
        adaptive_soft_negative_weight=False,
        soft_negative_weight=0.0,
        symmetric=False,
    )
    batch_metrics = loss.build_batch_metric_aggregators()

    # Here, the cosine similarity between the positive pair is perfect and the
    # negative pair is perpendicular ([0, 1] and [1, 0]).
    # We have 2 perfect matches, and one mismatch for class 0
    # Pred: [0, 1, 0]
    # True: [0, 1, 1]
    # Expected metrics:
    # Class 0: tp: 1, fp: 1, fn: 0, tn: 1
    # Precision: 1/2, Recall: 1, F1: 0.66667, Accuracy: 1.0
    # Class 1: tp: 1, fp: 0, fn: 1, tn: 1
    # Precision: 1, Recall: 0.5, F1: 0.66667, Accuracy: 0.5

    # Since soft_negative_weight is 0, the extra (3, 2) item should be ignored
    node_x = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [0.5, 0.5]]
    )
    graph_x = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [0.5, 0.5]]
    )
    y_true = {
        ContrastiveLabels.Ys: torch.tensor([0, 1, 1.0, 3, -100]),
        ContrastiveLabels.HardYs: torch.tensor([1, 0, 0, 2, -100]),
    }

    loss_value, returned_metrics = loss(node_x, graph_x, y_true, batch_metrics)

    normed_x = F.normalize(graph_x[:-2], p=2, dim=1)
    expect_sim = torch.matmul(normed_x, normed_x.T)

    expect_labels = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0]]
    )
    expect_labels = (expect_labels * 2) - 1

    # Row-wise normalization across non-diagonal entries
    weight_matrix = torch.ones((3, 3)).fill_diagonal_(0)
    pair_loss = -F.logsigmoid(expect_sim * expect_labels)
    row_weight_sums = weight_matrix.sum(dim=1)
    denoms = torch.clamp(row_weight_sums, min=1.0)
    row_means = (pair_loss * weight_matrix).sum(dim=1) / denoms
    expected_loss = row_means.mean()

    test.assert_close(loss_value, expected_loss)

    metrics = batch_metrics["classification"].compute()
    metrics["weight"] = 3

    # print("METRICS", metrics)
    # TODO(liamhebert): Add metrics back in, since they currently do not work with
    # our test examples

    # test.assert_close(metrics["none_recall"], torch.Tensor([1.0, 0.5, 0, 0]))
    # test.assert_close(metrics["none_precision"], torch.Tensor([0.5, 1.0, 0, 0]))

    # test.assert_close(
    #     metrics["none_f1"],
    #     torch.Tensor([0.66667, 0.66667, 0.0, 0.0]),
    # )

    # test.assert_close(metrics["macro_recall"], torch.tensor(0.75))
    # test.assert_close(metrics["macro_precision"], torch.tensor(0.75))
    # test.assert_close(metrics["macro_f1"], torch.tensor(0.66667))

    # test.assert_close(metrics["weighted_recall"], torch.tensor(0.66667))
    # test.assert_close(metrics["weighted_precision"], torch.tensor(0.83333))
    # test.assert_close(metrics["weighted_f1"], torch.tensor(0.66667))
    # test.assert_close(metrics["weighted_accuracy"], torch.tensor(0.66667))

    test.assert_close(returned_metrics["loss"], expected_loss)
    del returned_metrics["loss"]

    # # Formatting the metrics to match each other
    # for metric in ["f1", "precision", "recall"]:
    #     for class_id in range(4):
    #         metrics[f"class_{class_id}_" + metric] = metrics["none_" + metric][
    #             class_id
    #         ]
    #     del metrics["none_" + metric]

    # # Test to ensure that the logged metrics match the computed metrics and
    # # that all expected metrics are there.
    # assert metrics == returned_metrics

    # Now we do it again but flip the classes
    node_x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
    graph_x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
    y_true = {
        ContrastiveLabels.Ys: torch.tensor([2, 3.0, 3.0, -100]),
        ContrastiveLabels.HardYs: torch.tensor([3.0, 2.0, 2.0, -100]),
    }

    loss_value_2, returned_metrics = loss(
        node_x, graph_x, y_true, batch_metrics
    )

    expect_sim = torch.tensor(
        [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0]]
    )
    expect_labels = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0]]
    )
    expect_labels = (expect_labels * 2) - 1
    weight_matrix = torch.ones((3, 3)).fill_diagonal_(0)
    pair_loss = -F.logsigmoid(expect_sim * expect_labels)
    row_weight_sums = weight_matrix.sum(dim=1)
    denoms = torch.clamp(row_weight_sums, min=1.0)
    row_means = (pair_loss * weight_matrix).sum(dim=1) / denoms
    expected_loss = row_means.mean()

    test.assert_close(loss_value_2, expected_loss)

    metrics = batch_metrics["classification"].compute()
    metrics["weight"] = 3

    # test.assert_close(
    #     metrics["none_recall"], torch.Tensor([1.0, 0.5, 1.0, 0.5])
    # )
    # test.assert_close(
    #     metrics["none_precision"], torch.Tensor([0.5, 1.0, 0.5, 1.0])
    # )

    # test.assert_close(
    #     metrics["none_f1"],
    #     torch.Tensor([0.66667, 0.66667, 0.66667, 0.66667]),
    # )

    # test.assert_close(metrics["macro_recall"], torch.tensor(0.75))
    # test.assert_close(metrics["macro_precision"], torch.tensor(0.75))
    # test.assert_close(metrics["macro_f1"], torch.tensor(0.66667))

    # test.assert_close(metrics["weighted_recall"], torch.tensor(0.66667))
    # test.assert_close(metrics["weighted_precision"], torch.tensor(0.83333))
    # test.assert_close(metrics["weighted_f1"], torch.tensor(0.66667))
    # test.assert_close(metrics["weighted_accuracy"], torch.tensor(0.66667))

    test.assert_close(returned_metrics["loss"], expected_loss)
    del returned_metrics["loss"]


@pytest.mark.parametrize("num_gpus", [2, 4])
def test_distributed_contrastive_loss_value(num_gpus: int):
    """Test to ensure the loss is calculated accurately without duplicates."""

    class FakeDistContrastiveLoss(ContrastiveLoss):
        def all_gather(self, x: torch.Tensor) -> torch.Tensor:
            if num_gpus == 1:
                return x
            if isinstance(x, torch.Tensor):
                return x.unsqueeze(0).repeat_interleave(num_gpus, dim=0)

    loss = FakeDistContrastiveLoss(
        temperature=1,
        learnable_temperature=False,
        num_classes=4,
        bias=0.0,
        force_all_gather=True,
        symmetric=False,
    )

    batch_metrics = loss.build_batch_metric_aggregators()

    # Here, the cosine similarity between the positive pair is perfect and the
    # negative pair is perpendicular ([0, 1] and [1, 0]).
    # We have 2 perfect matches, and one mismatch for class 0
    # Pred: [0, 1, 0]
    # True: [0, 1, 1]
    # Expected metrics:
    # Class 0: tp: 1, fp: 1, fn: 0, tn: 1
    # Precision: 1/2, Recall: 1, F1: 0.66667, Accuracy: 1.0
    # Class 1: tp: 1, fp: 0, fn: 1, tn: 1
    # Precision: 1, Recall: 0.5, F1: 0.66667, Accuracy: 0.5

    node_x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
    graph_x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
    y_true = {
        ContrastiveLabels.Ys: torch.tensor([0, 1, 1.0, -100]),
        ContrastiveLabels.HardYs: torch.tensor([1, 0, 0, -100]),
    }

    weight_matrix = torch.ones((3 * num_gpus, 3 * num_gpus)).fill_diagonal_(0)

    expected_logits = torch.tensor(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
        ]
    ).tile((num_gpus, num_gpus))
    expected_labels = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    ).tile((num_gpus, num_gpus))
    expected_labels = (expected_labels * 2) - 1
    num_positives = [1 * num_gpus, 2 * num_gpus, 2 * num_gpus]

    loss_value, returned_metrics = loss(node_x, graph_x, y_true, batch_metrics)
    print("expected_soft", weight_matrix)
    num_valid_labels = torch.clamp((weight_matrix.sum(dim=1) > 0).sum(), min=1)
    print("expected_pos", num_valid_labels)
    pair_loss = -F.logsigmoid(expected_logits * expected_labels)
    row_weight_sums = weight_matrix.sum(dim=1)
    denoms = torch.clamp(row_weight_sums, min=1.0)
    row_means = (pair_loss * weight_matrix).sum(dim=1) / denoms
    expected_loss = row_means[row_weight_sums > 0].mean()

    test.assert_close(loss_value, expected_loss)


def test_contrastive_loss_symmetric_reduction():
    """Symmetric=True should average row-wise and column-wise reductions.

    We build a case with asymmetric row weights (adaptive soft negatives), so
    row-wise and column-wise means differ. The symmetric loss should match the
    average of the two.
    """
    loss = ContrastiveLoss(
        temperature=1,
        learnable_temperature=False,
        num_classes=3,
        bias=0.0,
        adaptive_soft_negative_weight=True,
        symmetric=True,
    )
    batch_metrics = loss.build_batch_metric_aggregators()

    # Three samples, labels [0,1,1]; embeddings chosen to yield sim matrix
    # [[1,0,0],[0,1,1],[0,1,1]] after normalization
    node_x = None
    graph_x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    ys = {
        ContrastiveLabels.Ys: torch.tensor([0, 1, 1]),
        ContrastiveLabels.HardYs: torch.tensor([2, 2, 2]),  # no hard matches
    }

    loss_value, _ = loss(node_x, graph_x, ys, batch_metrics)

    # Manual expected using implementation details
    norm_x = F.normalize(graph_x, p=2, dim=1)
    sim = norm_x @ norm_x.T

    targets = ys[ContrastiveLabels.Ys]
    target_matrix = targets.unsqueeze(1).eq(targets).float()
    target_matrix = target_matrix.fill_diagonal_(-1)

    hard_targets = ys[ContrastiveLabels.HardYs]
    hard_target_matrix = hard_targets.unsqueeze(1).eq(targets).float()
    hard_target_matrix[target_matrix.lt(0)] = -1  # align with padding/diag

    soft_labels = torch.logical_and(
        target_matrix.eq(0), hard_target_matrix.eq(0)
    )

    num_hard_labels = (
        torch.logical_or(target_matrix.eq(1), hard_target_matrix.eq(1))
    ).sum(dim=1)
    num_hard_labels = torch.clamp(num_hard_labels, min=1)

    extra_weight = (
        num_hard_labels / torch.clamp(soft_labels.sum(dim=1), min=1)
    ).reshape(-1, 1)
    weights = torch.where(soft_labels, extra_weight, 1.0)
    weights = weights.fill_diagonal_(0)

    target_pm = (target_matrix.clamp_min(0) * 2) - 1
    pair_loss = -F.logsigmoid(sim * target_pm)

    row_ws = weights.sum(dim=1)
    row_means = (pair_loss * weights).sum(dim=1) / torch.clamp(row_ws, min=1.0)
    valid_rows = row_ws > 0
    loss_rows = row_means[valid_rows].mean()

    col_ws = weights.sum(dim=0)
    col_means = (pair_loss * weights).sum(dim=0) / torch.clamp(col_ws, min=1.0)
    valid_cols = col_ws > 0
    loss_cols = col_means[valid_cols].mean()

    expected = 0.5 * (loss_rows + loss_cols)

    test.assert_close(loss_value, expected)


def test_contrastive_loss_padding_does_not_affect():
    """Adding a padded sample (-100) must not change the loss for valid pairs."""
    base_loss = ContrastiveLoss(
        temperature=1,
        learnable_temperature=False,
        num_classes=2,
        bias=0.0,
        adaptive_soft_negative_weight=False,
        soft_negative_weight=0.0,
        symmetric=False,
    )
    batch_metrics = base_loss.build_batch_metric_aggregators()

    # Two samples, different labels; orthogonal embeddings
    node_x = None
    small_graph = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    small_y = {
        ContrastiveLabels.Ys: torch.tensor([0, 1]),
        ContrastiveLabels.HardYs: torch.tensor([2, 2]),
    }
    loss_small, _ = base_loss(node_x, small_graph, small_y, batch_metrics)

    # Add a third padded sample; loss for valid rows should be unchanged
    big_graph = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])
    big_y = {
        ContrastiveLabels.Ys: torch.tensor([0, 1, -100]),
        ContrastiveLabels.HardYs: torch.tensor([2, 2, -100]),
    }
    loss_big, _ = base_loss(node_x, big_graph, big_y, batch_metrics)

    test.assert_close(loss_small, loss_big)
