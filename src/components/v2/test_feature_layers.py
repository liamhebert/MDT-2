from omegaconf import DictConfig
import pytest
import torch

from components.v2.discussion_transformer import DiscussionTransformer
from components.v2.feature_layers import GraphNodeFeature


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
    graph_node_feature_fixture: tuple[GraphNodeFeature, DictConfig],
) -> dict[str, torch.Tensor | int]:
    """
    Sample input to test the graph node feature layer.
    """
    _, config = graph_node_feature_fixture
    batch_size = 5
    num_nodes = 10

    return {
        "x": torch.rand(batch_size, config.hidden_dim),
        "out_degree": torch.randint(0, config.num_out_degree, (batch_size,)),
        "num_total_graphs": 2,
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
            output_x = model(**graph_node_feature_input)

        expected_shape = (
            graph_node_feature_input["x"].shape[0] + 2,  # Two graph tokens
            graph_node_feature_input["x"].shape[1],
        )

        assert output_x.shape == expected_shape

        # Test that the two graph global token are added at the start
        graph_global_token = model.graph_token.weight.repeat(2, 1)
        torch.testing.assert_close(output_x[:2, :], graph_global_token)
