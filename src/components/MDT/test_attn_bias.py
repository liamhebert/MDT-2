from components.MDT.attn_bias import GraphAttnBias
from components.v2.graph_attention_mask import generate_graph_attn_mask_tensor
import torch


def test_forward():
    layer = GraphAttnBias(
        num_heads=4,
        num_spatial=10,
    )
    graph_ids = torch.tensor([0, 1, 0, 0, 1, 1, 1])
    spatial_distance_matrix = torch.randint(
        low=0, high=8, size=(graph_ids.shape[0], graph_ids.shape[0])
    ).long()
    mask = generate_graph_attn_mask_tensor(
        graph_ids=graph_ids,
        spatial_distance_matrix=spatial_distance_matrix,
        max_spatial_distance=10,
        block_size=4,
    )

    output = layer.forward(
        attn_bias=mask,
        spatial_pos=spatial_distance_matrix,
    )
    assert output.shape == (1, 4, 7, 7)


# def test_forward(
#     self,
#     graph_attn_bias_fixture: tuple[GraphAttnBias, DictConfig],
#     graph_attn_bias_input: dict[str, torch.Tensor],
# ):
#     """
#     Tests the forward method of the graph attn bias layer.
#     """
#     model, config = graph_attn_bias_fixture
#     batch, num_nodes, _ = graph_attn_bias_input["attn_bias"].shape
#     num_heads = config.num_heads

#     with torch.no_grad():
#         output = model(**graph_attn_bias_input)

#     expected_shape = (
#         batch,
#         num_heads,
#         num_nodes,
#         num_nodes,
#     )

#     assert output.shape == expected_shape

#     expected_global_attn_bias = model.graph_token_virtual_distance.weight.view(
#         1, config.num_heads, 1
#     )
#     expected_global_attn_bias = expected_global_attn_bias.repeat(
#         batch, 1, num_nodes
#     )
#     # Graph token to all other nodes
#     torch.testing.assert_close(output[:, :, 0, :], expected_global_attn_bias)

#     # All other nodes to graph token
#     torch.testing.assert_close(
#         output[:, :, 1:, 0], expected_global_attn_bias[:, :, :-1]
#     )
