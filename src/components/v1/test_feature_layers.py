from omegaconf import DictConfig
import pytest
import torch

from components.v1.discussion_transformer import DiscussionTransformer
from components.v1.feature_layers import GraphAttnBias
from components.v1.feature_layers import GraphNodeFeature


@pytest.fixture(scope="class")
def graph_node_feature_fixture(
    discussion_transformer_fixture: tuple[DiscussionTransformer, DictConfig],
) -> tuple[GraphNodeFeature, DictConfig]:
    """
    Fixture to build the graph node feature layer.
    """
    model, config = discussion_transformer_fixture
    return model.graph_node_feature, config.graph_node_feature


@pytest.fixture(scope="function")
def graph_node_feature_input(
    graph_node_feature_fixture: tuple[GraphNodeFeature, DictConfig]
) -> dict[str, torch.Tensor]:
    """
    Sample input to test the graph node feature layer.
    """
    _, config = graph_node_feature_fixture
    batch_size = 5
    num_nodes = 10

    return {
        "x": torch.rand(batch_size, num_nodes, config.hidden_dim),
        "out_degree": torch.randint(
            0, config.num_out_degree, (batch_size, num_nodes)
        ),
    }


@pytest.fixture(scope="class")
def graph_attn_bias_fixture(
    discussion_transformer_fixture: tuple[DiscussionTransformer, DictConfig],
) -> tuple[GraphAttnBias, DictConfig]:
    """
    Fixture to build the graph node feature layer.
    """
    model, config = discussion_transformer_fixture
    return model.graph_attn_bias, config.graph_attn_bias


@pytest.fixture(scope="function")
def graph_attn_bias_input(
    graph_attn_bias_fixture: tuple[GraphNodeFeature, DictConfig]
) -> dict[str, torch.Tensor]:
    """
    Fixture to build the graph node feature input.
    """
    _, config = graph_attn_bias_fixture
    batch_size = 5
    num_nodes = 10

    return {
        "attn_bias": torch.zeros(batch_size, num_nodes + 1, num_nodes + 1),
        "spatial_pos": torch.randint(
            0, config.num_spatial, (batch_size, num_nodes, num_nodes)
        ),
    }


class TestGraphNodeFeature:
    """
    Tests the GraphNodeFeature class.
    """

    def test_forward(
        self,
        graph_node_feature_fixture: tuple[GraphNodeFeature, DictConfig],
        graph_node_feature_input: dict[str, torch.Tensor],
    ):
        """
        Tests the forward method of the graph node feature layer.
        """
        model, _ = graph_node_feature_fixture
        with torch.no_grad():
            output = model(**graph_node_feature_input)

        expected_shape = (
            graph_node_feature_input["x"].shape[0],
            graph_node_feature_input["x"].shape[1] + 1,  # added graph node
            graph_node_feature_input["x"].shape[2],
        )

        assert output.shape == expected_shape

        # Test that the graph global token is added at the start
        graph_global_token = model.graph_token.weight.repeat(output.shape[0], 1)
        torch.testing.assert_close(output[:, 0, :], graph_global_token)


class TestGraphAttnBias:
    """
    Tests the GraphAttnBias class.
    """

    def test_forward(
        self,
        graph_attn_bias_fixture: tuple[GraphAttnBias, DictConfig],
        graph_attn_bias_input: dict[str, torch.Tensor],
    ):
        """
        Tests the forward method of the graph attn bias layer.
        """
        model, config = graph_attn_bias_fixture
        batch, num_nodes, _ = graph_attn_bias_input["attn_bias"].shape
        num_heads = config.num_heads

        with torch.no_grad():
            output = model(**graph_attn_bias_input)

        expected_shape = (
            batch,
            num_heads,
            num_nodes,
            num_nodes,
        )

        assert output.shape == expected_shape

        expected_global_attn_bias = (
            model.graph_token_virtual_distance.weight.view(
                1, config.num_heads, 1
            )
        )
        expected_global_attn_bias = expected_global_attn_bias.repeat(
            batch, 1, num_nodes
        )
        # Graph token to all other nodes
        torch.testing.assert_close(
            output[:, :, 0, :], expected_global_attn_bias
        )

        # All other nodes to graph token
        torch.testing.assert_close(
            output[:, :, 1:, 0], expected_global_attn_bias[:, :, :-1]
        )
