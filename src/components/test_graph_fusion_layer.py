"""
Tests the GraphFusionStack and GraphFusionLayer classes.
"""

from omegaconf import DictConfig
import torch
from torch import nn

from components.graph_fusion_layer import GraphFusionLayer
from components.graph_fusion_layer import GraphFusionStack
from pytest import mark


def test_forward(
    graph_fusion_layer_input: dict[str, torch.Tensor],
    graph_fusion_stack_fixture: tuple[GraphFusionStack, DictConfig],
):
    """
    Tests the forward method of the graph fusion layer.
    """
    model, config = graph_fusion_stack_fixture

    with torch.no_grad():
        bert_output, vit_output, bottle_neck_output = model(
            **graph_fusion_layer_input
        )

    assert (
        bert_output.shape
        == graph_fusion_layer_input["bert_hidden_states"].shape
    )
    assert (
        vit_output.shape == graph_fusion_layer_input["vit_hidden_states"].shape
    )
    assert (
        bottle_neck_output.shape
        == graph_fusion_layer_input["bottle_neck"].shape
    )


class MockModalityLayer(nn.Module):
    """Mock class to replace Bert and ViT layers for testing.

    Rather then actually processing the hidden states, this class simply returns
    the hidden states as is. This is useful for testing bottleneck logic, since
    the forward signature is compatible with Bert and ViT.
    """

    scale: float

    def __init__(self, scale: float = 1.0):
        """Initializes the MockModalityLayer.

        Args:
            scale (float, optional): Optional factor to scale the hidden states,
                to make them unique. Defaults to 1.0.
        """
        super().__init__()
        # To not break dtype checks, we need to have at least one parameter.
        self.layer = nn.Parameter(torch.tensor(0.0))
        self.scale = scale

    def forward(
        self, hidden_states: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor]:
        """Returns the hidden_states scaled by the scale factor.

        Args:
            hidden_states (torch.Tensor): Tensor to scale.

        Returns:
            tuple[torch.Tensor]: Single item tuple containing the scaled
                hidden_states. We return a tuple to match the expected tuple
                output of a Bert or ViT layer.
        """
        return (hidden_states * self.scale,)


@mark.parametrize("use_projection", [True, False])
def test_fusion_layer(
    graph_fusion_layer_input: dict[str, torch.Tensor],
    use_projection: bool,
):
    """
    Tests the forward method of the graph fusion layer with projection layers.
    """
    dim = graph_fusion_layer_input["bottle_neck"].shape[-1]

    model = GraphFusionLayer(
        MockModalityLayer(),
        MockModalityLayer(),
        use_projection=use_projection,
        bottleneck_dim=dim,
        bert_dim=dim,
        vit_dim=dim,
    )
    with torch.no_grad():
        _, _, bottle_neck_output = model(**graph_fusion_layer_input)

    # Since the MockModalityLayer returns the hidden states as is, the
    # bottleneck tokens should be unchanged.
    if not use_projection:
        assert torch.allclose(
            bottle_neck_output, graph_fusion_layer_input["bottle_neck"]
        )


def test_selective_bottleneck_averaging(
    graph_fusion_layer_input: dict[str, torch.Tensor],
):
    """
    Tests the forward method of the graph fusion layer.
    """
    bottle_neck_dim = graph_fusion_layer_input["bottle_neck"].shape[-1]
    bert_dim = graph_fusion_layer_input["bert_hidden_states"].shape[-1]
    vit_dim = graph_fusion_layer_input["vit_hidden_states"].shape[-1]
    model = GraphFusionLayer(
        MockModalityLayer(),
        # To make sure we capture the correct selective positions, we double the
        # bottleneck tokens for the ViT model, so that the average is different.
        # (ie: (1 + 1) / 2 = 1, whereas (1 + 2) / 2 = 1.5)
        MockModalityLayer(scale=2.0),
        use_projection=False,
        bottleneck_dim=bottle_neck_dim,
        bert_dim=bert_dim,
        vit_dim=vit_dim,
    )
    # Only items 3 and 1 should have the full bottleneck, the rest should be
    # half.
    padding_mask = graph_fusion_layer_input["image_padding_mask"]
    padding_mask = torch.zeros_like(padding_mask)
    padding_mask[1] = 1
    padding_mask[3] = 1

    graph_fusion_layer_input["image_padding_mask"] = padding_mask.bool()

    # Ensuring we only have 2 images to process.
    graph_fusion_layer_input["vit_hidden_states"] = graph_fusion_layer_input[
        "vit_hidden_states"
    ][:2]

    with torch.no_grad():
        _, _, bottle_neck_output = model(**graph_fusion_layer_input)

    expected = torch.einsum(
        "ijk,i -> ijk",
        graph_fusion_layer_input["bottle_neck"],
        torch.tensor([1, 1.5, 1, 1.5, 1]),
    )

    # Since the MockModalityLayer returns the hidden states as is, the
    # bottleneck tokens should be unchanged.
    torch.testing.assert_close(bottle_neck_output, expected)
