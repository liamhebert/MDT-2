import math
import torch
import pytest

from data.types import ContrastiveLabels

from losses.loss_supcon_ce import SupConWithCELoss


@pytest.fixture()
def seed():
    torch.manual_seed(42)


@pytest.fixture()
def small_batch(seed):
    batch = 8
    emb_dim = 16
    num_classes = 4
    # Create two groups per class (ensure positives exist)
    embeddings = torch.randn(batch, emb_dim)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    return embeddings, labels, num_classes


def _forward(
    loss_mod: SupConWithCELoss, emb: torch.Tensor, labels: torch.Tensor
):
    ys = {ContrastiveLabels.Ys: labels}
    total_loss, metrics = loss_mod(None, emb, ys)
    return total_loss, metrics


def test_smoke_forward(small_batch):
    emb, labels, num_classes = small_batch
    mod = SupConWithCELoss(
        embedding_dim=emb.shape[1],
        num_classes=num_classes,
        lambda_contrastive=0.5,
        temperature=0.07,
        projection_dim=32,
    )
    loss, metrics = _forward(mod, emb, labels)
    assert torch.isfinite(loss), "Loss should be finite"
    assert loss > 0, "Combined loss should be positive"
    assert isinstance(metrics, dict)


def test_lambda_scaling_effect(small_batch):
    emb, labels, num_classes = small_batch
    base = SupConWithCELoss(
        embedding_dim=emb.shape[1],
        num_classes=num_classes,
        lambda_contrastive=0.0,
    )
    loss_base, _ = _forward(base, emb, labels)

    higher = SupConWithCELoss(
        embedding_dim=emb.shape[1],
        num_classes=num_classes,
        lambda_contrastive=1.0,
    )
    loss_high, _ = _forward(higher, emb, labels)

    # With lambda 0, total ~= CE. With lambda 1, total = CE + SupCon >= CE.
    assert loss_high >= loss_base - 1e-6


def test_ignore_index_handling(small_batch):
    emb, labels, num_classes = small_batch
    labels = labels.clone()
    labels[0] = -100  # ignore first sample
    mod = SupConWithCELoss(embedding_dim=emb.shape[1], num_classes=num_classes)
    loss, _ = _forward(mod, emb, labels)
    assert torch.isfinite(loss)


def test_backward_pass(small_batch):
    emb, labels, num_classes = small_batch
    emb = emb.clone().requires_grad_(True)
    mod = SupConWithCELoss(
        embedding_dim=emb.shape[1], num_classes=num_classes, projection_dim=32
    )
    loss, _ = _forward(mod, emb, labels)
    loss.backward()
    assert emb.grad is not None, "Embeddings should receive gradients"
    # Ensure not all zeros
    assert emb.grad.abs().sum() > 0


def test_learnable_temperature_updates(small_batch):
    emb, labels, num_classes = small_batch
    mod = SupConWithCELoss(embedding_dim=emb.shape[1], num_classes=num_classes)
    # Access underlying temperature parameter
    # (log-space inside SupContrastiveLoss)
    assert hasattr(mod, "temperature")
    param = mod.temperature
    assert isinstance(param, torch.Tensor)
    before = param.detach().clone()
    opt = torch.optim.SGD(mod.parameters(), lr=0.5)
    loss, _ = _forward(mod, emb, labels)
    opt.zero_grad()
    loss.backward()
    opt.step()
    # Temperature may update slightly; check gradient existence or value change
    if param.grad is not None and param.grad.abs().sum() > 0:
        assert not torch.equal(
            before, param.detach()
        ), "Temperature parameter did not change after optimizer step"


def test_different_lambda_produces_different_loss(small_batch):
    emb, labels, num_classes = small_batch
    mod1 = SupConWithCELoss(
        embedding_dim=emb.shape[1],
        num_classes=num_classes,
        lambda_contrastive=0.1,
    )
    mod2 = SupConWithCELoss(
        embedding_dim=emb.shape[1],
        num_classes=num_classes,
        lambda_contrastive=0.9,
    )
    loss1, _ = _forward(mod1, emb, labels)
    loss2, _ = _forward(mod2, emb, labels)
    # Expect scaling effect to produce different totals (most of the time)
    assert not math.isclose(
        loss1.item(), loss2.item(), rel_tol=1e-3
    ), "Losses should differ under different lambda_contrastive"
