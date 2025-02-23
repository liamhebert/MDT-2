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
                [0, 1.0, 1.0, 0.5, 0.5],  # 1 / 2 soft = 0.5
                [1.0, 0, 1.0, 0.5, 0.5],
                [1.0, 1.0, 0, 0.5, 0.5],
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

    normalized_weights = F.normalize(weight_matrix, p=1, dim=1)
    if weight == "none":
        num_labels = 2
    else:
        num_labels = 4

    normalized_weights = normalized_weights * num_labels
    expected_loss = torch.einsum(
        "ij, ij -> i",
        -F.logsigmoid(expect_sim * expect_labels),
        normalized_weights,
    )

    if weight == "none":
        expected_loss = expected_loss.sum() / num_labels
    else:
        expected_loss = expected_loss.mean()

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

    expect_sim = torch.tensor(
        [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0]]
    )
    expect_labels = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0]]
    )
    expect_labels = (expect_labels * 2) - 1

    # Each row must sum to the number of labels, therefore, 1 each.
    weight_matrix = torch.ones((3, 3))
    weight_matrix = weight_matrix.fill_diagonal_(0)

    expected_loss = (
        torch.einsum(
            "ij, ij -> i",
            -F.logsigmoid(expect_sim * expect_labels),
            weight_matrix,
        ).sum()
        / 2
    )

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

    test.assert_close(returned_metrics["loss"], expected_loss.item())
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
    weight_matrix = torch.ones((3, 3))
    weight_matrix = weight_matrix.fill_diagonal_(0)

    expected_loss = (
        torch.einsum(
            "ij, ij -> i",
            -F.logsigmoid(expect_sim * expect_labels),
            weight_matrix,
        ).sum()
        / 2
    )

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

    test.assert_close(returned_metrics["loss"], expected_loss.item())
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

    weight_matrix = torch.ones((3 * num_gpus, 3 * num_gpus))
    weight_matrix = weight_matrix.fill_diagonal_(0)

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

    loss_value, returned_metrics = loss(node_x, graph_x, y_true, batch_metrics)
    expected_loss = torch.einsum(
        "ij, ij -> i",
        -F.logsigmoid(expected_logits * expected_labels),
        weight_matrix,
    ).mean()

    test.assert_close(loss_value, expected_loss)
