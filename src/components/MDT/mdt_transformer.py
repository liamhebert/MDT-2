import torch
from torch import nn
from typing import Tuple
from data.types import TextFeatures

from transformers.models.vit.modeling_vit import ViTLayer
from transformers.models.vit.modeling_vit import ViTModel
from transformers.models.vit.modeling_vit import ViTPooler
from transformers.models.bert.modeling_bert import BertLayer
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.bert.modeling_bert import BertPooler

from components.MDT.attn_bias import GraphAttnBias
from components.v2.feature_layers import GraphNodeFeature
from components.v2.graph_encoder_layer import BaseGraphTransformer
from components.v2.discussion_transformer import DiscussionTransformerPrototype
from components.graph_fusion_layer import GraphFusionStack
from components.v2.graph_attention_layers import RMSNorm


class MDTBlock(nn.Module):

    graph_layer: BaseGraphTransformer
    fusion_layer: GraphFusionStack
    num_bottlenecks: int

    def __init__(
        self,
        graph_layer: BaseGraphTransformer,
        bert_layer_stack: list[BertLayer],
        vit_layer_stack: list[ViTLayer],
        embedding_dim: int = 768,
        num_bottlenecks: int = 1,
    ):
        super().__init__()
        self.fusion_layer = GraphFusionStack(
            bert_layer_stack,
            vit_layer_stack,
            use_projection=False,
            bottleneck_dim=embedding_dim,
            bert_dim=embedding_dim,
            vit_dim=embedding_dim,
        )
        # self.pre_norm = RMSNorm(embedding_dim)
        self.graph_layer = graph_layer
        self.post_norm = RMSNorm(embedding_dim)
        self.num_bottlenecks = num_bottlenecks

    def forward(
        self,
        x: torch.Tensor,
        num_total_graphs: int,
        graph_mask: torch.Tensor,
        bert_output: torch.Tensor,
        vit_output: torch.Tensor,
        bottle_neck: torch.Tensor,
        image_padding_mask: torch.Tensor,
        text_attention_mask: torch.Tensor,
    ):
        # Apply pre-normalization for better gradient flow and stability

        # Process through graph layer
        graph_x = self.graph_layer(
            x,
            mask=graph_mask,
        )

        # extract bottle_neck tokens
        graph_tokens = graph_x[:num_total_graphs, :]
        node_tokens = graph_x[num_total_graphs:, :]

        bottle_neck = torch.cat(
            (node_tokens.unsqueeze(1), bottle_neck[:, 1:, :]), dim=1
        )

        # Apply pre-normalization before fusion

        bert_output, vit_output, bottle_neck = self.fusion_layer(
            bert_hidden_states=bert_output,
            vit_hidden_states=vit_output,
            bottle_neck=bottle_neck,
            image_padding_mask=image_padding_mask,
            bert_attention_mask=text_attention_mask,
        )

        node_tokens = bottle_neck[:, 0, :]

        # Combine graph tokens and node tokens
        graph_x = torch.cat((graph_tokens, node_tokens), dim=0)

        return graph_x, bert_output, vit_output, bottle_neck


class MDTDiscussionTransformer(DiscussionTransformerPrototype):
    """
    Primary model class to create discussion and comment embeddings.
    """

    embedding_dim: int
    graph_node_feature: GraphNodeFeature
    graph_attn_bias: GraphAttnBias
    vit_model: ViTModel
    vit_pooler: ViTPooler
    text_model: BertModel
    text_pooler: BertPooler
    fusion_layers: nn.ModuleList
    graphormer_layers: nn.ModuleList
    num_bottle_neck: int
    bottle_neck: nn.Parameter

    def __init__(
        self,
        graph_attn_bias: GraphAttnBias,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.graph_attn_bias = graph_attn_bias

    def build_discussion_block(
        self,
        graph_layer: BaseGraphTransformer,
        bert_layer_stack: list[BertLayer],
        vit_layer_stack: list[ViTLayer],
        embedding_dim: int,
        num_bottlenecks: int,
    ):
        """Returns a DiscussionTransformerBlock with the given parameters."""
        return MDTBlock(
            graph_layer=graph_layer,
            bert_layer_stack=bert_layer_stack,
            vit_layer_stack=vit_layer_stack,
            embedding_dim=embedding_dim,
            num_bottlenecks=num_bottlenecks,
        )

    def forward(
        self,
        text_input: dict[str, torch.Tensor],
        image_input: dict[str, torch.Tensor],
        image_padding_mask: torch.Tensor,
        indexing_spatial_pos: torch.Tensor,
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

            == Image inputs ==
            image_input (torch.Tensor): batched and tokenized image features,
                with shape (batch_size * nodes, D)
            image_padding_mask (torch.Tensor): batched boolean tensor indicating
                whether a node has an image. Shape (batch_size * nodes).

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

        bert_output = self.text_model(**text_input).last_hidden_state
        text_attention_mask = text_input[TextFeatures.AttentionMask]

        flattened_batch, seq_len, hidden_dim = bert_output.size()

        if image_input["pixel_values"].shape[0] > 0:
            vit_output = self.vit_model(**image_input).last_hidden_state
        else:
            vit_output = None

        bottle_neck = self.bottle_neck.repeat(flattened_batch, 1, 1)

        bert_output, vit_output, bottle_neck = self.first_fusion_layer(
            bert_hidden_states=bert_output,
            vit_hidden_states=vit_output,
            bottle_neck=bottle_neck,
            image_padding_mask=image_padding_mask,
            bert_attention_mask=text_attention_mask,
        )
        # This does not have the graph ids in them
        graph_x = bottle_neck[:, 0, :]
        assert graph_x.size() == (flattened_batch, hidden_dim)

        # This does
        # graph_x, graph_ids = self.graph_node_feature(
        #     graph_x, out_degree, graph_ids, num_total_graphs
        # )
        graph_x = self.graph_node_feature(graph_x, out_degree, num_total_graphs)

        # assert graph_ids.shape == graph_x.shape[:1], (
        #     f"Expected same shape, got {graph_ids.shape=} and"
        #     f" {graph_x.shape[:1]=}"
        # )

        # account for padding while computing the representation

        graph_mask = self.graph_attn_bias(
            attn_bias=graph_mask, spatial_pos=indexing_spatial_pos
        )

        for block in self.blocks:
            graph_x, bert_output, vit_output, bottle_neck = block(
                graph_x,
                num_total_graphs,
                graph_mask,
                bert_output,
                vit_output,
                bottle_neck,
                image_padding_mask,
                text_attention_mask,
            )

        # Normalized
        graph_x = self.final_graphormer_layer(
            graph_x,
            mask=graph_mask,
        )
        if self.graph_token_average:
            global_embedding = self.average_embeddings_by_index(
                graph_x, graph_ids
            )
        else:
            global_embedding = graph_x[:num_total_graphs, :]

        assert global_embedding.size() == (num_total_graphs, hidden_dim)
        # Output is averaged bottleneck and bert cls
        node_embedding = (
            graph_x[num_total_graphs:, :] + bert_output[:, 0, :]
        ) / 2
        return node_embedding, global_embedding
