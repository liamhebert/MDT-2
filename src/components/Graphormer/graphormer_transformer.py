import torch
from torch import nn
from typing import Tuple, Callable
from omegaconf import DictConfig

from transformers.models.bert.modeling_bert import BertModel
from transformers.models.bert.modeling_bert import BertPooler
from transformers import AutoModel

from components.MDT.attn_bias import GraphAttnBias
from components.v2.feature_layers import GraphNodeFeature
from components.v2.graph_encoder_layer import BaseGraphTransformer
from components.v2.discussion_transformer import freeze_module_params


class GraphormerDiscussionTransformer(nn.Module):
    """
    Primary model class to create discussion and comment embeddings.
    """

    embedding_dim: int
    graph_node_feature: GraphNodeFeature
    graph_attn_bias: GraphAttnBias
    text_model: BertModel
    text_pooler: BertPooler
    graphormer_layers: nn.ModuleList

    def __init__(
        self,
        graph_node_feature: GraphNodeFeature,
        graph_attn_bias: GraphAttnBias,
        graph_stack_factory: Callable[[int], BaseGraphTransformer],
        text_model_config: DictConfig,
        embedding_dim: int = 768,
        freeze_initial_encoders: bool = False,
        block_size: int = 0,
        **kwargs,
    ):
        super().__init__()

        self.text_model = AutoModel.from_pretrained(
            text_model_config["bert_model_name"],
            attn_implementation="sdpa",
        ).train()
        self.graph_attn_bias = graph_attn_bias
        self.graph_node_feature = graph_node_feature

        if freeze_initial_encoders:
            freeze_module_params(self.text_model, freeze=True)

        self.graphormer_model = graph_stack_factory(depth=0)  # type: ignore
        self.block_size = block_size
        self.embedding_dim = embedding_dim

    def forward(
        self,
        text_input: dict[str, torch.Tensor],
        spatial_pos: torch.Tensor,
        out_degree: torch.Tensor,
        num_total_graphs: int,
        graph_mask: torch.Tensor,
        graph_ids: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The forward function of the Discussion Transformer model.

        Args:
            == Text inputs ==
            text_input (dict[str, torch.Tensor]): The tokenized text inputs,
                containing:
                - text_input_ids (torch.Tensor): batched tokenized text ids, with
                    shape (batch_size * nodes, T)
                - text_token_type_ids (torch.Tensor): batched token type ids, with
                    shape (batch_size * nodes, T)
                - text_attention_mask (torch.Tensor): batched text attention mask,
                    with shape (batch_size * nodes, T), where 1 indicates a token
                    that should be attended to and 0 indicates padding.

            == Graph inputs ==
            graph_ids (torch.Tensor): Id of the graph each node belongs to,
                where padding nodes are assigned the value PADDING_GRAPH_ID, with
                shape (batch_size * nodes). This is used to mask out attention.
                It is assumed that the graph_ids are contiguous and start from 0.
            spatial_pos (torch.Tensor): Matrix with shape
                (batch_size * nodes, batch_size * nodes, 2) indicating the
                number of up hops and down hops between each node in the graph.
            in_degree (torch.Tensor): batched in-degrees, corresponding to the
                in-degree of each node in the graph. Padded with 0s and shifted
                by 1. Shape (batch_size * nodes).
            out_degree (torch.Tensor): batched out-degrees, corresponding to the
                out-degree of each node in the graph. Padded with 0s and shifted
                by 1. Shape (batch_size * nodes).
            num_total_graphs (int): Total number of unique graphs in the batch,
                shape (). Note that this is different then graph_ids, which is
                node-wise.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Returns
                - node_embedding: The final node embeddings for each node in the
                    graph with shape (batch_size * nodes, C).
                - global_embedding: The final global embedding for the graph,
                    with shape (batch_size, C).
        """
        # Ideally, we keep a consistent shape and just flatten here, rather then
        # having to dynamically index here. Attention mask should handle this.

        bert_output = self.text_model(**text_input).pooler_output

        flattened_batch, seq_len, hidden_dim = bert_output.size()

        # This does not have the graph ids in them
        graph_x = bert_output
        assert graph_x.size() == (flattened_batch, hidden_dim)

        graph_x = self.graph_node_feature(graph_x, out_degree, num_total_graphs)

        # account for padding while computing the representation

        graph_mask = self.graph_attn_bias(
            attn_bias=graph_mask, spatial_pos=spatial_pos
        )

        graph_x = self.graphormer_model(
            graph_x,
            mask=graph_mask,
        )

        global_embedding = graph_x[:num_total_graphs, :]

        assert global_embedding.size() == (num_total_graphs, hidden_dim)
        node_embedding = graph_x[num_total_graphs:, :]
        return node_embedding, global_embedding
