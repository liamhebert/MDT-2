"""
Tests for the discussion transformer module.
"""

from hydra.utils import instantiate
from omegaconf import DictConfig
from transformers import BertConfig
from transformers import ViTConfig
import torch
from pytest import mark

from data.types import TextFeatures, ImageFeatures
from components.v2.discussion_transformer import DiscussionTransformer
from components.v2.discussion_transformer_all_tokens import (
    DiscussionTransformerAllTokens,
)
from components.v2.graph_attention_mask import generate_graph_attn_mask_tensor


def discussion_transformer_input(
    config: DictConfig,
) -> dict[str, torch.Tensor | dict[str, torch.Tensor] | int]:
    """
    Fixture to build the discussion transformer input.
    """

    batch_size = 3
    num_nodes = 4
    total_nodes = batch_size * num_nodes
    # Mask should be static.
    sequence_length = 5

    text_config: BertConfig = instantiate(config.text_model_config.test_config)
    vision_config: ViTConfig = instantiate(config.vit_model_config.test_config)
    graph_ids = torch.tensor(
        [0, 1, 2] + [0] * num_nodes + [1] * num_nodes + [2] * num_nodes
    )

    return {
        "text_input": {
            TextFeatures.InputIds: torch.randint(
                0,
                text_config.vocab_size,
                (total_nodes, sequence_length),
            ),
            TextFeatures.AttentionMask: torch.tile(
                torch.tensor([[1, 1, 1, 0, 0]]), (total_nodes, 1)
            ),
            TextFeatures.TokenTypeIds: torch.zeros(
                (total_nodes, sequence_length), dtype=torch.long
            ),
        },
        # == Image inputs ==
        "image_input": {
            ImageFeatures.PixelValues: torch.rand(
                5,
                vision_config.num_channels,
                vision_config.image_size,
                vision_config.image_size,
            ),
        },
        "image_padding_mask": (
            torch.tensor([1] * 5 + [0] * (total_nodes - 5)).bool()
        ),
        # == Graph inputs ==
        "rotary_pos": torch.randint(0, 10, (total_nodes + batch_size, 2)),
        "out_degree": torch.randint(0, 5, (total_nodes,)),
        "num_total_graphs": batch_size,
        "graph_mask": generate_graph_attn_mask_tensor(
            graph_ids,
            torch.full(
                (total_nodes + batch_size, total_nodes + batch_size),
                1,
                dtype=torch.long,
            ),
        ),
        # Attention graph ids
        "graph_ids": graph_ids,
    }


def test_build_discussion_transformer(
    discussion_transformer_fixture: tuple[DiscussionTransformer, DictConfig],
):
    """
    Tests to ensure the discussion transformer is built correctly.
    """
    model, config = discussion_transformer_fixture

    # Test individual fusion stack size
    assert all(
        [
            len(block.fusion_layer) == config.fusion_stack_size
            for block in model.blocks
        ]
    ), (
        f"Expected {config.fusion_stack_size=}."
        f"Got {[len(layer) for layer in model.fusion_layers]}"
    )

    # Test number of fusion stacks
    assert len(model.blocks) == config.num_fusion_stack - 1

    # Test individual graphormer stack size
    assert all(
        len(block.graph_layer) == config.graph_stack_factory.num_layers
        for block in model.blocks
    )

    # Test number of bottleneck embeddings
    assert model.bottle_neck.shape[0] == config.num_bottle_neck


def test_build_bert_encoder(
    discussion_transformer_fixture: tuple[DiscussionTransformer, DictConfig],
):
    """
    Tests to ensure the BERT encoder is built correctly.
    """
    model, config = discussion_transformer_fixture

    # Build sample bert model to trigger method.
    total_fusion_layers = config.fusion_stack_size * config.num_fusion_stack

    # Because we build text config objects for testing, we have to build it here.
    text_model_config = instantiate(config.text_model_config)
    test_config: BertConfig = text_model_config.test_config
    bert_model, text_fusion_layers = model.build_bert_encoder(
        num_fusion_layers=total_fusion_layers, **text_model_config
    )

    # Test number of total fusion layers
    assert len(text_fusion_layers) == total_fusion_layers

    # Test that the bert model has the correct number of layers
    assert (
        len(bert_model.encoder.layer)
        == test_config.num_hidden_layers - total_fusion_layers
    )
    assert len(bert_model.encoder.layer) != 0

    # Test that the hidden_dropout and attention_dropout are set correctly
    assert (
        bert_model.config.hidden_dropout_prob == test_config.hidden_dropout_prob
    )
    assert (
        bert_model.config.attention_probs_dropout_prob
        == test_config.attention_probs_dropout_prob
    )


def test_build_vit_encoder(
    discussion_transformer_fixture: tuple[DiscussionTransformer, DictConfig],
):
    """
    Tests to ensure the ViT encoder is built correctly.
    """
    model, config = discussion_transformer_fixture

    # Build sample bert model to trigger method.
    total_fusion_layers = config.fusion_stack_size * config.num_fusion_stack

    # Because we build ViT config objects for testing, we have to build it here.
    vit_model_config = instantiate(config.vit_model_config)
    vit_model, vit_fusion_layers = model.build_vit_encoder(
        num_fusion_layers=total_fusion_layers, **vit_model_config
    )
    test_config: ViTConfig = vit_model_config.test_config

    # Test number of total fusion layers
    assert len(vit_fusion_layers) == total_fusion_layers

    # Test that the bert model has the correct number of layers
    assert (
        len(vit_model.encoder.layer)
        == test_config.num_hidden_layers - total_fusion_layers
    )
    assert len(vit_model.encoder.layer) != 0

    # Test that the hidden_dropout and attention_dropout are set correctly
    assert (
        vit_model.config.hidden_dropout_prob == test_config.hidden_dropout_prob
    )
    assert (
        vit_model.config.attention_probs_dropout_prob
        == test_config.attention_probs_dropout_prob
    )


@mark.parametrize(
    "model_cls", [DiscussionTransformer, DiscussionTransformerAllTokens]
)
def test_forward(
    discussion_transformer_fixture: tuple[DiscussionTransformer, DictConfig],
    model_cls: type[DiscussionTransformer],
):
    _, config = discussion_transformer_fixture
    config._target_ = model_cls.__module__ + "." + model_cls.__name__

    model: DiscussionTransformer = instantiate(config)

    model.forward(**discussion_transformer_input(config))  # type: ignore
