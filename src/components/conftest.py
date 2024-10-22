"""
PyTest fixtures specific to the components module.
"""

from hydra.utils import instantiate
from omegaconf import DictConfig
import pytest
import torch
from transformers import BertConfig
from transformers import ViTConfig

from components.discussion_transformer import DiscussionTransformer
from components.graph_fusion_layer import GraphFusionStack


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
def graph_fusion_layer_fixture(
    discussion_transformer_fixture: tuple[DiscussionTransformer, DictConfig]
) -> tuple[GraphFusionStack, DictConfig]:
    """
    Fixture to build the graph fusion layer.
    """
    model, config = discussion_transformer_fixture
    return model.fusion_layers[0], config


@pytest.fixture(scope="function")
def graph_fusion_layer_input(
    graph_fusion_layer_fixture: tuple[DiscussionTransformer, DictConfig]
) -> dict[str, torch.Tensor]:
    """
    Fixture to build the graph fusion layer input.
    """
    _, config = graph_fusion_layer_fixture
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
        "image_indices": torch.arange(0, batch_size),
    }
