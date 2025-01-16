"""
Tests the GraphEncoderStack and GraphEncoderLayer classes.
"""

from omegaconf import DictConfig
import torch

from components.v1.graph_encoder_layer import GraphEncoderStack


def test_forward(
    graph_encoder_stack_fixture: tuple[GraphEncoderStack, DictConfig],
    graph_encoder_layer_input: dict[str, torch.Tensor],
):
    """
    Tests the forward method of the graph encoder layer.
    """
    model, config = graph_encoder_stack_fixture

    with torch.no_grad():
        output = model(**graph_encoder_layer_input)

    assert output.shape == graph_encoder_layer_input["x"].shape
