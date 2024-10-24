"""
Tests the GraphEncoderStack and GraphEncoderLayer classes.
"""

from omegaconf import DictConfig
import torch

from components.graph_encoder_layer import GraphEncoderStack


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


# def test_attn_mask_all_zeroes(
#     graph_encoder_stack_fixture: tuple[GraphEncoderStack, DictConfig],
#     graph_encoder_layer_input: dict[str, torch.Tensor],
# ):
#     """
#     Tests the forward method with an attention mask of all zeroes.
#     """
#     model, config = graph_encoder_stack_fixture

#     graph_encoder_layer_input["attn_mask"] = torch.ones_like(
#         graph_encoder_layer_input["attn_mask"]
#     )

#     with torch.no_grad():
#         output = model(**graph_encoder_layer_input)

#     assert output.shape == graph_encoder_layer_input["x"].shape
#     assert output == torch.ones_like(output)


# def test_attn_padding_mask_all_zeroes(
#     graph_encoder_stack_fixture: tuple[GraphEncoderStack, DictConfig],
#     graph_encoder_layer_input: dict[str, torch.Tensor],
# ):
#     """
#     Tests the forward method with an attention padding mask of all zeroes.
#     """
#     model, config = graph_encoder_stack_fixture

#     graph_encoder_layer_input["self_attn_padding_mask"][:, 1] = True

#     with torch.no_grad():
#         output = model(**graph_encoder_layer_input)

#     assert output.shape == graph_encoder_layer_input["x"].shape
#     print(output[:, 1])
#     print(graph_encoder_layer_input["x"][:, 1])
#     torch.testing.assert_close(output, torch.ones_like(output))
