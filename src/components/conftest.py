"""
PyTest fixtures specific to the components module.
"""

from hydra.utils import instantiate
from omegaconf import DictConfig
import pytest
import torch
from transformers import BertConfig
from transformers import ViTConfig

from components.graph_fusion_layer import GraphFusionStack
from components.v1.discussion_transformer import DiscussionTransformer
from components.v1.graph_encoder_layer import GraphEncoderStack
from components.v2.graph_attention_mask import PADDING_GRAPH_ID


@pytest.fixture(scope="module")
def discussion_transformer_fixture(
    cfg_test_global: DictConfig,
) -> tuple[DiscussionTransformer, DictConfig]:
    """
    Fixture to build the discussion transformer model.
    """
    model_config = cfg_test_global.model.encoder
    return instantiate(model_config), model_config


@pytest.fixture(scope="module")
def graph_fusion_stack_fixture(
    discussion_transformer_fixture: tuple[DiscussionTransformer, DictConfig],
) -> tuple[GraphFusionStack, DictConfig]:
    """
    Fixture to build the graph fusion stack.
    """
    model, config = discussion_transformer_fixture
    return model.fusion_layers[0], config


@pytest.fixture(scope="module")
def graph_encoder_stack_fixture(
    discussion_transformer_fixture: tuple[DiscussionTransformer, DictConfig],
) -> tuple[GraphEncoderStack, DictConfig]:
    """
    Fixture to build the graph encoder stack.
    """
    model, config = discussion_transformer_fixture
    return model.graphormer_layers[0], config.graph_stack_config


def discussion_transformer_input(
    discussion_transformer_fixture: tuple[DiscussionTransformer, DictConfig],
) -> dict[str, torch.Tensor]:
    """
    Fixture to build the discussion transformer input.
    """
    _, config = discussion_transformer_fixture
    batch_size = 3
    num_nodes = 5
    flattened_batch_size = batch_size * num_nodes
    # Mask should be static.
    sequence_length = 5

    text_config: BertConfig = instantiate(config.text_model_config.test_config)
    vision_config: ViTConfig = instantiate(config.vit_model_config.test_config)

    graph_ids = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    num_padding = flattened_batch_size - len(graph_ids)
    num_valid = len(graph_ids)
    graph_ids = graph_ids + [PADDING_GRAPH_ID] * (num_padding)
    text_attention_mask = torch.tile(
        torch.tensor([[1, 1, 1, 0, 0]]), (num_valid, 1)
    )
    text_attention_mask = torch.stack(
        [text_attention_mask, torch.zeros((num_padding, sequence_length))]
    )

    input_ids = torch.randint(
        0, text_config.vocab_size, (flattened_batch_size, sequence_length)
    )

    return {
        # B x T
        # == Text inputs ==
        "input_ids": input_ids,
        "text_attention_mask": torch.tile(
            torch.tensor([[1, 1, 1, 0, 0]]), (flattened_batch_size, 1)
        ),
        "token_type_ids": torch.zeros_like(input_ids),
        # == Image inputs ==
        "image_input": torch.rand(
            flattened_batch_size,
            sequence_length,  # Assumes sequence_length patches
            vision_config.hidden_size,
        ),
        # TODO(liamhebert): Add padding mask for images here.
        # Placeholder
        "image_padding_mask": torch.zeros((flattened_batch_size)),
        # == Graph inputs ==
        # Attention graph ids
        # TODO(liamhebert): Should we directly pass the blockmask in the
        # forward pass (constructing it in the collator function) or construct
        # it in the forward pass?
        # For now, we will construct it in the forward pass.
        "graph_ids": torch.Tensor(graph_ids),
        "spatial_pos": torch.randint(
            0, 10, (flattened_batch_size, flattened_batch_size, 2)
        ),
        "in_degree": torch.randint(0, 5, (flattened_batch_size, num_nodes)),
        "out_degree": torch.randint(0, 5, (flattened_batch_size, num_nodes)),
    }


@pytest.fixture(scope="function")
def graph_fusion_layer_input(
    graph_fusion_stack_fixture: tuple[GraphFusionStack, DictConfig],
) -> dict[str, torch.Tensor]:
    """
    Fixture to build the graph fusion layer input.
    """
    _, config = graph_fusion_stack_fixture
    batch_size = 5
    sequence_length = 5
    num_bottle_neck = config.num_bottle_neck

    text_config: BertConfig = instantiate(config.text_model_config.test_config)
    vision_config: ViTConfig = instantiate(config.vit_model_config.test_config)

    return {
        "bert_hidden_states": torch.rand(
            batch_size, sequence_length, text_config.hidden_size
        ),
        "vit_hidden_states": torch.rand(
            batch_size, sequence_length, vision_config.hidden_size
        ),
        "bottle_neck": torch.ones(
            (batch_size, num_bottle_neck, config.embedding_dim)
        ),
        "bert_attention_mask": torch.tile(
            torch.tensor([[1, 1, 1, 0, 0]]), (batch_size, 1)
        ),
        "image_padding_mask": torch.ones(batch_size).bool(),
    }


@pytest.fixture(scope="function")
def graph_encoder_layer_input(
    graph_encoder_stack_fixture: tuple[GraphEncoderStack, DictConfig],
) -> dict[str, torch.Tensor]:
    """
    Fixture to build the graph encoder layer input.
    """
    _, config = graph_encoder_stack_fixture
    batch_size = 5
    nodes = 10
    embed_dim = config.embedding_dim
    num_heads = config.num_attention_heads

    return {
        "x": torch.rand(batch_size, nodes, embed_dim),
        "self_attn_bias": torch.rand(batch_size, num_heads, nodes, nodes),
        "self_attn_mask": torch.zeros((batch_size * num_heads, nodes, nodes)),
        "self_attn_padding_mask": torch.zeros((batch_size, nodes)),
    }
