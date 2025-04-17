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
        num_total_graphs: int,
    ) -> torch.Tensor:
        """Computes additional node features for the graph.

        Args:
            x (torch.Tensor): Current hidden state of the model, with shape
                (Batch * Graph_Nodes, Hidden_Dim)
            out_degree (torch.Tensor): Tensor of out-degrees for each node in
                the graph, with shape (Batch * Graph_Nodes).
            num_total_graphs (int): Total number of unique graphs in the batch.

        Returns:
            torch.Tensor: New batch of graphs where each node is enhanced with
                an embedding corresponding to the out_degree of the node.
                Further, we add a new node to each graph corresponding to a
                discussion pooling token ([gCLS]) as the first node in the
                graph. These tokens are added to the beginning of the sequence.

                Returns with shape
                (Batch * Graph_Nodes + num_unique_graphs, Hidden_Dim)
        """
        node_feature = x + self.out_degree_encoder(out_degree)

        graph_token_feature = self.graph_token.weight.repeat(
            num_total_graphs, 1
        )
        graph_token_feature = graph_token_feature.to(x.dtype)

        # TODO(limahebert): Should we concat graph token to the end or
        # to the start? Keep in mind how we sample bottleneck tokens.
        graph_node_feature = torch.cat(
            [graph_token_feature, node_feature], dim=0
        )

        return graph_node_feature
