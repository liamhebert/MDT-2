from components.v2.graph_attention_layers import RMSNorm
from components.graph_fusion_layer import GraphFusionStack
from components.v2.graph_encoder_layer import BaseGraphTransformer
from transformers.models.vit.modeling_vit import ViTLayer
from transformers.models.bert.modeling_bert import BertLayer

import torch
import torch.nn as nn


class DiscussionTransformerBlock(nn.Module):

    graph_layer: BaseGraphTransformer
    fusion_layer: GraphFusionStack
    norm: RMSNorm
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
            use_projection=True,
            bottleneck_dim=embedding_dim,
            bert_dim=embedding_dim,
            vit_dim=embedding_dim,
        )
        self.pre_norm = RMSNorm(embedding_dim)
        self.graph_layer = graph_layer
        self.post_norm = RMSNorm(embedding_dim)
        self.num_bottlenecks = num_bottlenecks

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
        bottle_neck = torch.cat(
            (node_tokens.unsqueeze(1), bottle_neck[:, 1:, :]), dim=1
        )
        bottle_neck = self.pre_norm(bottle_neck)

        bert_output, vit_output, bottle_neck = self.fusion_layer(
            bert_hidden_states=bert_output,
            vit_hidden_states=vit_output,
            bottle_neck=bottle_neck,
            image_padding_mask=image_padding_mask,
            bert_attention_mask=text_attention_mask,
        )
        bottle_neck = self.post_norm(bottle_neck)

        node_tokens = bottle_neck[:, 0, :]

        graph_x = torch.cat((graph_tokens, node_tokens), dim=0)

        return graph_x, bert_output, vit_output, bottle_neck
