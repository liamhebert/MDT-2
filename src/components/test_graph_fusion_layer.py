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
    assert False


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


class DummyGateScalar(nn.Module):
    """Gate that returns a constant scalar per token in [0,1]."""

    def __init__(self, value: float):
        super().__init__()
        self.value = float(value)

    def forward(
        self, text_bn: torch.Tensor, img_bn: torch.Tensor
    ) -> torch.Tensor:
        # Return shape (B_img, T, 1)
        b, t, _ = text_bn.shape
        return torch.full(
            (b, t, 1), self.value, dtype=text_bn.dtype, device=text_bn.device
        )


class DummyGatePerDim(nn.Module):
    """Gate that returns a constant vector per token in [0,1]^D."""

    def __init__(self, value: float):
        super().__init__()
        self.val = float(value)

    def forward(
        self, text_bn: torch.Tensor, img_bn: torch.Tensor
    ) -> torch.Tensor:
        # Return shape (B_img, T, D)
        b, t, d = text_bn.shape
        return torch.full(
            (b, t, d), self.val, dtype=text_bn.dtype, device=text_bn.device
        )


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


def test_bottleneck_gating_scalar_zero_one(
    graph_fusion_layer_input: dict[str, torch.Tensor],
):
    """Gating using per-token gate should interpolate text and image embeddings.

    When g=0 -> fused == text; g=1 -> fused == image.
    """
    bottle_neck_dim = graph_fusion_layer_input["bottle_neck"].shape[-1]
    bert_dim = graph_fusion_layer_input["bert_hidden_states"].shape[-1]
    vit_dim = graph_fusion_layer_input["vit_hidden_states"].shape[-1]

    # Set which samples have images (positions 1 and 3)
    padding_mask = torch.zeros_like(
        graph_fusion_layer_input["image_padding_mask"]
    )
    padding_mask[1] = 1
    padding_mask[3] = 1
    graph_fusion_layer_input["image_padding_mask"] = padding_mask.bool()
    # Ensure vit has exactly the number of image samples
    graph_fusion_layer_input["vit_hidden_states"] = graph_fusion_layer_input[
        "vit_hidden_states"
    ][: int(padding_mask.sum())]

    # Model with gating enabled, scalar gate
    model = GraphFusionLayer(
        MockModalityLayer(),
        MockModalityLayer(scale=2.0),
        use_projection=False,
        bottleneck_dim=bottle_neck_dim,
        bert_dim=bert_dim,
        vit_dim=vit_dim,
        use_gating=True,
        gate_per_dim=False,
    )

    # Case g=0 -> output should equal text bottleneck
    model.gate_mlp = DummyGateScalar(0.0)
    with torch.no_grad():
        _, _, bottle_neck_output_zero = model(**graph_fusion_layer_input)

    expected_zero = torch.einsum(
        "ijk,i -> ijk",
        graph_fusion_layer_input["bottle_neck"],
        torch.tensor([1, 1, 1, 1, 1], dtype=bottle_neck_output_zero.dtype),
    )
    torch.testing.assert_close(bottle_neck_output_zero, expected_zero)

    # Case g=1 -> output should equal image bottleneck (scale 2.0) for image
    # rows, text otherwise
    model.gate_mlp = DummyGateScalar(1.0)
    with torch.no_grad():
        _, _, bottle_neck_output_one = model(**graph_fusion_layer_input)

    scale = torch.ones_like(padding_mask, dtype=bottle_neck_output_one.dtype)
    scale[padding_mask.bool()] = 2.0
    expected_one = torch.einsum(
        "ijk,i -> ijk",
        graph_fusion_layer_input["bottle_neck"],
        scale.to(bottle_neck_output_one.dtype),
    )
    torch.testing.assert_close(bottle_neck_output_one, expected_one)


def test_bottleneck_gating_per_dim(
    graph_fusion_layer_input: dict[str, torch.Tensor],
):
    """Per-dimension gate should match scalar behavior when set to 0 or 1."""
    bottle_neck_dim = graph_fusion_layer_input["bottle_neck"].shape[-1]
    bert_dim = graph_fusion_layer_input["bert_hidden_states"].shape[-1]
    vit_dim = graph_fusion_layer_input["vit_hidden_states"].shape[-1]

    padding_mask = torch.zeros_like(
        graph_fusion_layer_input["image_padding_mask"]
    )
    padding_mask[0] = 1
    graph_fusion_layer_input["image_padding_mask"] = padding_mask.bool()
    graph_fusion_layer_input["vit_hidden_states"] = graph_fusion_layer_input[
        "vit_hidden_states"
    ][: int(padding_mask.sum())]

    model = GraphFusionLayer(
        MockModalityLayer(),
        MockModalityLayer(scale=2.0),
        use_projection=False,
        bottleneck_dim=bottle_neck_dim,
        bert_dim=bert_dim,
        vit_dim=vit_dim,
        use_gating=True,
        gate_per_dim=True,
    )

    # g=0 per-dim
    model.gate_mlp = DummyGatePerDim(0.0)
    with torch.no_grad():
        _, _, bottle_neck_out = model(**graph_fusion_layer_input)

    torch.testing.assert_close(
        bottle_neck_out, graph_fusion_layer_input["bottle_neck"]
    )
