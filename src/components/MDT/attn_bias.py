import torch
import torch.nn as nn


def init_params(module: nn.Module) -> None:
    """Initializes embedding modules in the model to normal distribution.

    Args:
        module (nn.Module): If an nn.Embedding module, module to set to normal
        distribution.
    """
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class GraphAttnBias(nn.Module):
    """
    Compute attention bias for each head.
    """

    def __init__(
        self,
        num_heads: int,
        num_spatial: int,
    ):
        """Initialize the GraphAttnBias module.

        This module computes a scalar bias term to add to the attention scores
        for each head in the Graph Transformer model. This bias term is used to
        encode information about the spatial position of nodes in the graph.

        Args:
            num_heads (int): Number of attention heads used by the Graph
                Transformer model. This is important to ensure that we have
                the correct dimensionality for the attention bias (one per head).
            num_spatial (int): Number of spatial positions to create embeddings
                for. For instance, if there are 10 possible spatial positions,
                this value should be 10 to create an embedding for each
                position.
        """
        super(GraphAttnBias, self).__init__()
        self.num_heads = num_heads

        # Since we want to compute a bias scalar for attention, we create
        # embeddings of size num_heads, which corresponds to a single value
        # added per head.
        self.spatial_pos_encoder = nn.Embedding(
            (num_spatial * num_spatial) + 5,
            num_heads,
            padding_idx=0,
        )

        self.apply(lambda module: init_params(module))

    def forward(
        self,
        attn_bias: torch.Tensor,
        spatial_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Update attention bias with learned values for spatial position.

        Args:
            attn_bias (torch.Tensor): Initial attention bias for each node pair,
                with shape (1, Nodes, Nodes).
            spatial_pos (torch.Tensor): Set of distance indices between nodes,
                with shape (Nodes, Nodes).

        Returns:
            torch.Tensor: Returns an updated attn_bias with added values
                corresponding to spatial pos + a virtual distance for the
                graph token. Shape (Nodes, Nodes).
        """
        # We want to change this to instead return score mod and mask mod.

        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.repeat(
            1, self.num_heads, 1, 1
        )  # [1, n_head, n_node, n_node]

        # spatial pos
        # [n_node, n_node, n_head] -> [n_head, n_node, n_node]

        spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(
            2, 0, 1
        )
        graph_attn_bias = graph_attn_bias + spatial_pos_bias

        # # reset spatial pos here
        # t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        # graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
        # graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t

        return graph_attn_bias
