"""Modules related to the primary DiscussionTransformer model.

This model relates to the MDT model as presented in

Hebert, L., Sahu, G., Guo, Y., Sreenivas, N. K., Golab, L., & Cohen, R. (2024).
Multi-Modal Discussion Transformer: Integrating Text, Images and Graph
Transformers to Detect Hate Speech on Social Media.
Proceedings of the AAAI Conference on Artificial Intelligence,
38(20), 22096-22104. https://doi.org/10.1609/aaai.v38i20.30213
"""

from typing import Tuple

import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertLayer
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.vit.modeling_vit import ViTLayer
from transformers.models.vit.modeling_vit import ViTModel

from components.v2.feature_layers import GraphNodeFeature
from components.v2.discussion_blocks import DiscussionTransformerBlock
from components.v2.discussion_transformer import DiscussionTransformerPrototype
from components.graph_fusion_layer import GraphFusionStack
from components.v2.graph_encoder_layer import BaseGraphTransformer
from components.v2.graph_attention_layers import RMSNorm

from utils.pylogger import RankedLogger

from data.types import TextFeatures

logger = RankedLogger(rank_zero_only=True)


class DiscussionTransformerBlockAllTokens(DiscussionTransformerBlock):

    graph_layer: BaseGraphTransformer
    fusion_layer: GraphFusionStack
    norm: RMSNorm
    num_bottlenecks: int

    def forward(
        self,
        x: torch.Tensor,
        num_total_graphs: int,
        graph_mask: torch.Tensor,
        rotary_pos: torch.Tensor,
        bert_output: torch.Tensor,
        vit_output: torch.Tensor,
        bottle_neck: torch.Tensor,
        image_padding_mask: torch.Tensor,
        text_attention_mask: torch.Tensor,
    ):
        """
        Processes the fusion and graph layer for the Discussion Transformer block.
        This block processes the input through the fusion layer and then the graph
        layer, and returns the output of the graph layer.

        Args:
            x (Tensor): The input to the graph layer, with shape
                (batch_size * nodes * num_bot + num_graphs, embed_dim).
            num_total_graphs (int): The number of graphs in the batch.
            graph_mask (Tensor): The attention mask for the graph layer, with
                shape (batch_size * nodes * num_bot, ).
            rotary_pos (Tensor): The rotary position embeddings for the graph
                layer, with shape (batch_size * node_tokens * num_bot, 2).
            bert_output (Tensor): The output of the BERT model, with shape
                (batch_size * nodes, seq_len, embed_dim).
            vit_output (Tensor): The output of the ViT model, with shape
                (batch_size * nodes, img_seq_len, embed_dim).
            bottle_neck (Tensor): The updated bottleneck tokens, with shape
                (batch_size * node_tokens, num_bottlenecks, embed_dim). NOTE this
                is unused with the all tokens model, since we update everything
                anyways. Here for compatibility.
            image_padding_mask (Tensor): The padding mask for the image input,
                with shape (batch_size * nodes, img_seq_len).
            text_attention_mask (Tensor): The attention mask for the text input,
                with shape (batch_size * nodes, seq_len).
        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]: The output of the graph layer,
                the output of the BERT model, the output of the ViT model, and
                the output of the bottleneck layer.


        """
        x = self.post_norm(x)

        graph_x = self.graph_layer(
            x,
            mask=graph_mask,
            spatial_pos=rotary_pos,
        )

        # extract bottle_neck tokens
        graph_tokens = graph_x[:num_total_graphs, :]
        node_tokens = graph_x[num_total_graphs:, :]
        # TODO(liamhebert): Experiment adding graph tokens here too
        bottle_neck = node_tokens.reshape(
            node_tokens.shape[0] // self.num_bottlenecks,
            self.num_bottlenecks,
            node_tokens.shape[1],
        )
        bottle_neck = self.pre_norm(bottle_neck)

        bert_output, vit_output, bottle_neck = self.fusion_layer(
            bert_hidden_states=bert_output,
            vit_hidden_states=vit_output,
            bottle_neck=bottle_neck,
            image_padding_mask=image_padding_mask,
            bert_attention_mask=text_attention_mask,
        )
        # bottle_neck = self.post_norm(bottle_neck)

        # Bottleneck shape are [num_nodes, num_bot, embed_dim]
        # We want to flatten this so that we get [num_nodes * num_bot, embed_dim]
        node_tokens = bottle_neck.flatten(start_dim=0, end_dim=1)

        graph_x = torch.cat((graph_tokens, node_tokens), dim=0)

        return graph_x, bert_output, vit_output, bottle_neck


class DiscussionTransformerAllTokens(DiscussionTransformerPrototype):
    """
    Primary model class to create discussion and comment embeddings.
    """

    embedding_dim: int
    graph_node_feature: GraphNodeFeature
    vit_model: ViTModel
    text_model: BertModel
    first_fusion_layer: GraphFusionStack
    blocks: nn.ModuleList
    final_graphormer_layer: BaseGraphTransformer
    num_bottle_neck: int
    bottle_neck: nn.Parameter
    block_size: int

    def build_discussion_block(
        self,
        graph_layer: BaseGraphTransformer,
        bert_layer_stack: list[BertLayer],
        vit_layer_stack: list[ViTLayer],
        embedding_dim: int,
        num_bottlenecks: int,
    ) -> DiscussionTransformerBlock:
        """Returns a DiscussionTransformerBlock with the given parameters."""
        return DiscussionTransformerBlockAllTokens(
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
        rotary_pos: torch.Tensor,
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
                - input_ids (torch.Tensor): batched tokenized text ids, with
                    shape (batch_size * nodes, T)
                - token_type_ids (torch.Tensor): batched token type ids, with
                    shape (batch_size * nodes, T)
                - attention_mask (torch.Tensor): batched text attention mask,
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

        num_bottle_neck = self.num_bottle_neck
        bottle_neck = self.bottle_neck.repeat(flattened_batch, 1, 1)

        bert_output, vit_output, bottle_neck = self.first_fusion_layer(
            bert_hidden_states=bert_output,
            vit_hidden_states=vit_output,
            bottle_neck=bottle_neck,
            image_padding_mask=image_padding_mask,
            bert_attention_mask=text_attention_mask,
        )

        graph_x = bottle_neck.flatten(start_dim=0, end_dim=1)
        assert graph_x.size() == (flattened_batch * num_bottle_neck, hidden_dim)

        # = Format all graph tensors to fit the bottleneck flattened format =
        repeats = torch.full(
            (graph_mask.shape[-1],),
            num_bottle_neck,
            dtype=torch.long,
            device=graph_mask.device,
        )
        repeats[:num_total_graphs] = 1
        # Out degree
        out_degree = torch.repeat_interleave(
            out_degree, num_bottle_neck, dim=-1
        )
        assert (
            out_degree.shape
            == (graph_x.shape[0],)
            == (flattened_batch * num_bottle_neck,)
        ), (
            out_degree.shape,
            graph_x.shape[0],
            flattened_batch * num_bottle_neck,
        )

        graph_x = self.graph_node_feature(graph_x, out_degree, num_total_graphs)

        # Graph ids
        graph_ids = torch.repeat_interleave(graph_ids, repeats, dim=-1)
        assert graph_ids.shape == (graph_x.shape[0],), (
            graph_ids.shape,
            graph_x.shape[0],
        )

        # Spatial position
        rotary_pos = torch.repeat_interleave(rotary_pos, repeats, dim=0)
        assert rotary_pos.shape == (
            graph_x.shape[0],
            2,
        ), (rotary_pos.shape, (graph_x.shape[0], 2))

        # Graph mask
        graph_mask = torch.repeat_interleave(graph_mask, repeats, dim=-1)
        graph_mask = torch.repeat_interleave(graph_mask, repeats, dim=-2)

        assert graph_mask.shape == (
            1,
            1,
            graph_x.shape[0],
            graph_x.shape[0],
        ), (
            graph_mask.shape,
            (
                1,
                1,
                graph_x.shape[0],
                graph_x.shape[0],
            ),
        )

        # account for padding while computing the representation

        for block in self.blocks:
            graph_x, bert_output, vit_output, bottle_neck = block(
                graph_x,
                num_total_graphs,
                graph_mask,
                rotary_pos,
                bert_output,
                vit_output,
                bottle_neck,
                image_padding_mask,
                text_attention_mask,
            )

        graph_x = self.final_graphormer_layer(
            graph_x,
            mask=graph_mask,
            spatial_pos=rotary_pos,
        )

        if self.graph_token_average:
            global_embedding = self.average_embeddings_by_index(
                graph_x, graph_ids
            )
        else:
            global_embedding = graph_x[:num_total_graphs, :]

        assert global_embedding.size() == (num_total_graphs, hidden_dim)
        node_embedding = graph_x[num_total_graphs:, :]
        return node_embedding, global_embedding
