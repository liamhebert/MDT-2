import torch

from losses.loss_dec import DeclusteringLoss


def test_update_centers_and_shapes():
    torch.manual_seed(0)
    loss_fn = DeclusteringLoss(num_clusters=4, embedding_dim=3)
    assert loss_fn.cluster_centers.shape == (4, 3)
    new_centers = torch.randn(4, 3)
    old = loss_fn.cluster_centers.detach().clone()
    loss_fn.update_centers(new_centers)
    assert torch.allclose(loss_fn.cluster_centers, new_centers)
    assert not torch.allclose(old, loss_fn.cluster_centers)


def test_soft_assign_row_sums_and_argmax():
    torch.manual_seed(0)
    K, D, B = 3, 2, 5
    loss_fn = DeclusteringLoss(num_clusters=K, embedding_dim=D)

    # Set centers far apart for clear nearest-cluster behavior
    centers = torch.tensor([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0]])
    loss_fn.update_centers(centers)

    z = torch.tensor(
        [
            [0.1, 0.2],  # near center 0
            [99.0, 1.0],  # near center 1
            [2.0, 90.0],  # near center 2
            [1.0, 0.5],  # near center 0
            [101.0, 3.0],  # near center 1
        ],
        dtype=torch.float32,
    )
    q = loss_fn._soft_assign(z)
    assert q.shape == (B, K)
    # Rows sum to 1
    assert torch.allclose(q.sum(dim=1), torch.ones(B), atol=1e-5)
    preds = q.argmax(dim=1)
    expected = torch.tensor([0, 1, 2, 0, 1])
    assert torch.equal(preds, expected)


def test_target_distribution_properties():
    torch.manual_seed(42)
    K, D, B = 5, 4, 8
    loss_fn = DeclusteringLoss(num_clusters=K, embedding_dim=D)
    z = torch.randn(B, D)
    q = loss_fn._soft_assign(z)
    p = loss_fn._target_distribution(q)
    assert p.shape == q.shape
    # Rows sum to 1
    assert torch.allclose(p.sum(dim=1), torch.ones(B), atol=1e-6)

    # Target distribution should be more "peaked" (lower entropy) or equal
    def entropy(x):
        return -(x * torch.log(x)).sum(dim=1)

    h_q = entropy(q)
    h_p = entropy(p)
    # DEC target distribution is expected to sharpen assignments on average,
    # but individual samples can (slightly) increase in entropy due to the
    # global frequency re-weighting by f_j. Therefore we check that the mean
    # entropy decreases (or stays the same within tolerance) and that at least
    # one sample is strictly sharpened.
    assert h_p.mean() <= h_q.mean() + 1e-6
    assert (h_p < h_q).any()


def test_forward_matches_manual_kl_and_grad():
    torch.manual_seed(7)
    K, D, B = 4, 3, 6
    loss_fn = DeclusteringLoss(num_clusters=K, embedding_dim=D)
    embeddings = torch.randn(B, D, requires_grad=True)

    batch_metrics = loss_fn.build_batch_metric_aggregators()
    loss, metrics_out = loss_fn(
        node_embeddings=None,
        graph_embeddings=embeddings,
        ys={},  # unsupervised
        batch_metrics=batch_metrics,
    )
    assert loss.ndim == 0
    assert "loss" in metrics_out and "weight" in metrics_out
    assert metrics_out["weight"] == 0  # no labeled examples

    # Manual KL(P||Q)/B
    with torch.no_grad():
        q = loss_fn._soft_assign(embeddings.detach())
        p = loss_fn._target_distribution(q)
        manual_kl = (p * (torch.log(p) - torch.log(q))).sum() / B
        assert torch.allclose(loss, manual_kl, atol=1e-6)

    loss.backward()
    assert embeddings.grad is not None
    assert loss_fn.cluster_centers.grad is not None


def test_metrics_with_labels():
    torch.manual_seed(21)
    K, D, B, C = 3, 3, 10, 3
    loss_fn = DeclusteringLoss(num_clusters=K, embedding_dim=D, num_classes=C)
    z = torch.randn(B, D)
    # Provide some valid labels and some ignored
    y = torch.randint(0, C, (B,))
    y[0] = -100  # ignored

    batch_metrics = loss_fn.build_batch_metric_aggregators()
    loss, metrics_out = loss_fn(
        node_embeddings=None,
        graph_embeddings=z,
        ys={"y": y},
        batch_metrics=batch_metrics,
    )
    assert loss.ndim == 0
    assert metrics_out["loss"].ndim == 0
    assert metrics_out["weight"] == (y != -100).sum()

    # Compute classification metrics without error
    classification = batch_metrics.get("classification")
    assert classification is not None
    computed = classification.compute()
    # Spot check presence of a few keys
    expected_keys = {"macro_recall", "weighted_accuracy", "micro_f1"}
    assert expected_keys.issubset(set(computed.keys()))


def test_predict_clusters():
    torch.manual_seed(5)
    K, D, B = 4, 2, 5
    loss_fn = DeclusteringLoss(num_clusters=K, embedding_dim=D)
    centers = torch.tensor([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
    loss_fn.update_centers(centers)
    z = torch.tensor(
        [
            [1.0, 1.0],  # cluster 0
            [9.5, 0.2],  # cluster 1
            [0.3, 9.7],  # cluster 2
            [10.1, 9.9],  # cluster 3
            [11.0, -0.5],  # cluster 1
        ]
    )
    preds = loss_fn.predict_clusters(z)
    assert torch.equal(preds, torch.tensor([0, 1, 2, 3, 1]))
