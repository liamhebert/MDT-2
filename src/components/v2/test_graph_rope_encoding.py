from components.v2.graph_rope_encoding import RoPE
import torch
from pytest import mark


@mark.parametrize("rope_mixed", [True, False])
def test_rope_smoke_test(rope_mixed: bool):
    rope = RoPE(
        head_dim=64, num_heads=4, rope_theta=10.0, rope_mixed=rope_mixed
    )
    q = torch.randn(6, 4, 64)
    k = q.clone()

    spatial = torch.arange(0, 12).reshape(6, 2)

    q, k = rope(q, k, spatial)
    assert q.shape == (6, 4, 64)
    assert torch.isfinite(q).all()
    torch.testing.assert_close(q, k)
