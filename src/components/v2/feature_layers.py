"""
Modules related to computing extra aux features for the Graph Transformer.
"""

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


class GraphNodeFeature(nn.Module):
    """
    Compute node features for each node in the graph.
    """

    def __init__(
        self,
        num_out_degree: int,
        hidden_dim: int,
    ):
        """Initialize the GraphNodeFeature module.

        This module computes initial node features for each node in the graph.
        These node features are used to augment the initial input embeddings
        given by the fusion models. Currently, this includes augmenting the
        initial embeddings with out_degree embeddings and adding an extra graph
        pooling node (gCLS) to the graph.

        Args:
            num_out_degree (int): Number of out_degree embeddings to create. Any
                out_degree passed beyond this limit will be set to the same
                embedding.
            hidden_dim (int): The hidden_dimension of the embeddings to create.
        """
        super(GraphNodeFeature, self).__init__()

        # 1 for graph token
        self.out_degree_encoder = nn.Embedding(
            num_out_degree, hidden_dim, padding_idx=0
        )

        self.graph_token = nn.Embedding(1, hidden_dim)

        self.apply(lambda module: init_params(module))

    def forward(
        self,
        x: torch.Tensor,
        out_degree: torch.Tensor,
        graph_ids: torch.Tensor,
        num_total_graphs: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes additional node features for the graph.

        Args:
            x (torch.Tensor): Current hidden state of the model, with shape
                (Batch * Graph_Nodes, Hidden_Dim)
            out_degree (torch.Tensor): Tensor of out-degrees for each node in
                the graph, with shape (Batch * Graph_Nodes).
            graph_ids (torch.Tensor): Tensor of graph ids for each node in the
                graph, with shape (Batch * Graph_Nodes).
            num_total_graphs (int): Total number of unique graphs in the batch.

        Returns:
            torch.Tensor: New batch of graphs where each node is enhanced with
                an embedding corresponding to the out_degree of the node.
                Further, we add a new node to each graph corresponding to a
                discussion pooling token ([gCLS]) as the first node in the
                graph. These tokens are added to the beginning of the sequence.

                Returns with shape
                (Batch * Graph_Nodes + num_unique_graphs, Hidden_Dim)
            torch.Tensor: Updated Graph IDs to include the new graph tokens added
                to each graph. This tensor has shape
                (Batch * Graph_Nodes + num_unique_graphs)
        """
        node_feature = x + self.out_degree_encoder(out_degree)

        graph_token_feature = self.graph_token.weight.repeat(
            num_total_graphs, 1
        )
        graph_token_graph_ids = torch.arange(
            0, num_total_graphs, device=x.device
        )

        # TODO(limahebert): Should we concat graph token to the end or
        # to the start? Keep in mind how we sample bottleneck tokens.
        graph_node_feature = torch.cat(
            [graph_token_feature, node_feature], dim=0
        )
        graph_ids = torch.cat([graph_token_graph_ids, graph_ids], dim=0)

        return graph_node_feature, graph_ids


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

        # self.edge_encoder = nn.Embedding(
        #     num_edges + 1, num_heads, padding_idx=0
        # )
        # self.edge_type = edge_type
        # if self.edge_type == "multi_hop":
        #     self.edge_dis_encoder = nn.Embedding(
        #         num_edge_dis * num_heads * num_heads, 1
        #     )

        # Since we want to compute a bias scalar for attention, we create
        # embeddings of size num_heads, which corresponds to a single value
        # added per head.
        self.spatial_pos_encoder = nn.Embedding(
            num_spatial, num_heads, padding_idx=0
        )

        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)

        self.apply(lambda module: init_params(module))

    def forward(
        self, attn_bias: torch.Tensor, spatial_pos: torch.Tensor
    ) -> torch.Tensor:
        """Update attention bias with learned values for spatial position.

        Args:
            attn_bias (torch.Tensor): Initial attention bias for each node pair,
                with shape (Batch, Nodes + 1, Nodes + 1), +1 for the graph node.
            spatial_pos (torch.Tensor): Set of distance indices between nodes,
                with shape (Batch, Nodes + 1, Nodes + 1).

        Returns:
            torch.Tensor: Returns an updated attn_bias with added values
                corresponding to spatial pos + a virtual distance for the
                graph token. Shape (Batch, Nodes + 1, Nodes + 1)
        """
        # We want to change this to instead return score mod and mask mod.

        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(
            1, self.num_heads, 1, 1
        )  # [n_graph, n_head, n_node+1, n_node+1]

        # spatial pos
        # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(
            0, 3, 1, 2
        )
        graph_attn_bias[:, :, 1:, 1:] = (
            graph_attn_bias[:, :, 1:, 1:] + spatial_pos_bias
        )

        # reset spatial pos here
        t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
        graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t

        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)  # reset

        return graph_attn_bias
