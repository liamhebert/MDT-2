"""Tests for the GraphEncoderLayer module."""

import torch

from components.v2.graph_encoder_layer import (
    BaseGraphTransformer,
    GraphTransformerBlock,
)
from components.v2.graph_attention_mask import generate_graph_attn_mask_mod
from pytest import mark


def create_graph_transformer_factory(
    n_heads,
    n_kv_heads,
    dim,
    ffn_dim,
    norm_eps,
    head_dim,
    diff_attn=False,
    use_rope=False,
):
    """Constructor to create a graph transformers, needed for init."""

    def graph_transformer_factory(depth):
        return GraphTransformerBlock(
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            dim=dim,
            ffn_dim=ffn_dim,
            norm_eps=norm_eps,
            head_dim=head_dim,
            differential_attention=diff_attn,
            use_rope=use_rope,
            depth=depth,
        )

    return graph_transformer_factory


@mark.parametrize("diff_attn", [True, False])
@mark.parametrize("use_rope", [True, False])
def test_graph_transformer(diff_attn: bool, use_rope: bool):
    """Smoke test for the GraphTransformerBlock."""

    n_heads = 16
    n_kv_heads = 16
    dim = 64
    ffn_dim = 256
    norm_eps = 1e-6
    head_dim = 16
    seq_len = 8

    graph_tfmr_factory = create_graph_transformer_factory(
        n_heads,
        n_kv_heads,
        dim,
        ffn_dim,
        norm_eps,
        head_dim,
        diff_attn,
        use_rope,
    )

    num_layers = 3
    graph_tfmr = BaseGraphTransformer(
        num_layers=num_layers,
        graph_tfmr_factory=graph_tfmr_factory,
    )

    x = torch.randn(seq_len, dim)
    graph_ids = torch.randint(0, 2, (seq_len,))
    spatial_distance_matrix = torch.randn(seq_len, seq_len)
    rotary_pos = torch.randint(0, 10, (seq_len, 2))
    max_spatial_distance = 4
    mask = generate_graph_attn_mask_mod(
        graph_ids=graph_ids,
        spatial_distance_matrix=spatial_distance_matrix,
        max_spatial_distance=max_spatial_distance,
        num_heads=n_heads,
        block_size=4,
    )
    out = graph_tfmr(x, mask, rotary_pos)
    assert out.shape == x.shape
