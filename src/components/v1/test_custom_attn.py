"""Tests the legacy multi-head attention layer."""

from omegaconf import DictConfig
import pytest
import torch

from components.v1.custom_attn import MultiheadAttention
from components.v1.graph_encoder_layer import GraphEncoderStack


@pytest.fixture(scope="module")
def multihead_attention_fixture(
    graph_encoder_stack_fixture: tuple[GraphEncoderStack, DictConfig],
) -> tuple[MultiheadAttention, DictConfig]:
    """
    Fixture to build the multihead attention layer.
    """
    model, config = graph_encoder_stack_fixture
    return model.layers[0].self_attn, config


@pytest.fixture(scope="function")
def multihead_attention_input(
    multihead_attention_fixture: tuple[MultiheadAttention, DictConfig]
) -> dict[str, torch.Tensor]:
    """
    Sample input to test the multi-head attention layer.
    """
    _, config = multihead_attention_fixture
    batch_size = 5
    num_nodes = 10
    embed_dim = config.embedding_dim
    num_heads = config.num_attention_heads

    return {
        "query": torch.rand(batch_size, num_nodes, embed_dim),
        "attn_bias": torch.rand(batch_size, num_heads, num_nodes, num_nodes),
        "key_padding_mask": torch.randint(0, 2, (batch_size, num_nodes)),
        "attn_mask": torch.rand(batch_size * num_heads, num_nodes, num_nodes),
    }


def test_forward(
    multihead_attention_fixture: tuple[MultiheadAttention, DictConfig],
    multihead_attention_input: dict[str, torch.Tensor],
):
    """
    Test the forward pass of the multi-head attention layer.
    """
    model, _ = multihead_attention_fixture
    output = model(**multihead_attention_input)
    assert output.size() == multihead_attention_input["query"].size()


def test_neg_inf_bias(
    multihead_attention_fixture: tuple[MultiheadAttention, DictConfig],
    multihead_attention_input: dict[str, torch.Tensor],
):
    """
    Test the forward pass of the multi-head attention layer with negative
    infinity bias.
    """
    model, _ = multihead_attention_fixture
    multihead_attention_input["attn_bias"] = torch.full(
        multihead_attention_input["attn_bias"].size(), float("-inf")
    )
    output = model(**multihead_attention_input)
    assert output.size() == multihead_attention_input["query"].size()
    assert torch.isnan(output).all()
