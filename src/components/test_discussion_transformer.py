"""
Tests for the discussion transformer module.
"""

from hydra.utils import instantiate
from omegaconf import DictConfig
from transformers import BertConfig
from transformers import ViTConfig

from components.discussion_transformer import DiscussionTransformer


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
            len(layer) == config.fusion_stack_size
            for layer in model.fusion_layers
        ]
    ), (
        f"Expected {config.fusion_stack_size=}."
        f"Got {[len(layer) for layer in model.fusion_layers]}"
    )

    # Test number of fusion stacks
    assert len(model.fusion_layers) == config.num_fusion_stack

    # Test number of graphormer layers
    assert len(model.graphormer_layers) == config.num_fusion_stack + 1

    # Test individual graphormer stack size
    assert all(
        len(layer) == config.graph_stack_config.num_layers
        for layer in model.graphormer_layers
    )

    # Test number of bottleneck embeddings
    assert model.bottle_neck.num_embeddings == config.num_bottle_neck


def test_build_bert_encoder(
    discussion_transformer_fixture: tuple[DiscussionTransformer, DictConfig]
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
    bert_model, text_fusion_layers, text_pooler = model.build_bert_encoder(
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
    discussion_transformer_fixture: tuple[DiscussionTransformer, DictConfig]
):
    """
    Tests to ensure the ViT encoder is built correctly.
    """
    model, config = discussion_transformer_fixture

    # Build sample bert model to trigger method.
    total_fusion_layers = config.fusion_stack_size * config.num_fusion_stack

    # Because we build ViT config objects for testing, we have to build it here.
    vit_model_config = instantiate(config.vit_model_config)
    vit_model, vit_fusion_layers, vit_pooler = model.build_vit_encoder(
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


# TODO(liamhebert): Add tests for the forward method of the discussion
# transformer.
