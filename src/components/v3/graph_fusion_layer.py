"""
Layers associated with computing multi-modal embeddings.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers.modeling_utils import ModuleUtilsMixin
from transformers.models.bert.modeling_bert import BertLayer
from components.v2.graph_attention_layers import RMSNorm


class ModuleUtilsMixinWrapper(ModuleUtilsMixin):
    """Wrapper to make ModuleUtilsMixin compatible with non HF modules.

    ModuleUtilsMixin expects a config object with an is_decoder attribute. This
    is a dummy config object that satisfies that requirement.
    """

    @dataclass
    class MockConfig:
        """
        MockConfig object to satisfy ModuleUtilsMixin requirements.
        """

        is_decoder: bool = False

    config: MockConfig = MockConfig()


class GraphFusionLayer(nn.Module, ModuleUtilsMixinWrapper):
    """Fuse text and image inputs with shared bottleneck and optional gating.

    The layer concatenates a set of bottleneck tokens to each modality's hidden
    states, applies the respective encoder layer, then extracts the updated
    bottleneck tokens and fuses the text/image bottleneck representations. When
    gating is enabled, fusion is computed as per-token
    fused = (1 - g) * text_bn + g * image_bn, with g in [0,1]; otherwise a
    simple average is used. Fusion only applies to rows with images present as
    indicated by ``image_padding_mask``.
    """

    bert_encoder: nn.Module
    gradient_checkpointing: bool
    bottle_neck_norm: RMSNorm

    def __init__(
        self,
        bert_layer: nn.Module,
        use_projection: bool = False,
        bottleneck_dim: int = 768,
        bert_dim: int = 768,
    ) -> None:
        """Initializes the GraphFusionLayer module.

        This module fuses the text and image inputs using a shared set of
        bottleneck tokens. In practice, this works by concatenating the set of
        bottleneck tokens to the input embeddings for each modality. Then, the
        concatenated embeddings are passed through the respective encoders.
        After encoding, the bottleneck tokens are averaged between both
        modalities.

        NOTE: If there are no image inputs, the vision path is skipped and the
        bottleneck tokens come solely from the text path.

        Args:
            bert_layer (BertLayer): The BERT layer to use for the text inputs
            use_projection (bool, optional): Whether to project the bottleneck
                embeddings into modality specific versions before concatenating
                them. Defaults to False.
        """
        super().__init__()

        self.bert_encoder = bert_layer
        self.gradient_checkpointing = False
        self.use_projection = use_projection
        if use_projection:
            # TODO(liamhebert): This should be tuned to the specific use case.
            self.bottle_to_bert_projection = nn.Linear(bottleneck_dim, bert_dim)
            self.bert_to_bottle_projection = nn.Linear(bert_dim, bottleneck_dim)

        self.bottle_neck_norm = RMSNorm(bottleneck_dim)

    @torch.compiler.disable
    def forward(
        self,
        bert_hidden_states: torch.Tensor,
        bottle_neck: torch.Tensor,
        bert_attention_mask: Optional[torch.Tensor] = None,
        bert_position_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute both modality layers with bottleneck information passing.

        Args:
            bert_hidden_states (torch.Tensor): Tensor of shape (batch_size,
                sequence_length, hidden_size) representing the input hidden
                states for the text model.
            bottle_neck (torch.Tensor): Tensor of shape (batch_size,
                num_bottleneck, hidden_size) representing the hidden state of
                the bottleneck tokens. These tokens are added to both inputs and
                then removed before returning.
            bert_attention_mask (torch.Tensor, optional): Tensor of shape
                (batch_size, sequence_length) representing the attention mask to
                avoid attending to padding. Tokens with 1 are *not masked* and
                tokens with 0 are *masked*. If None, all tokens are attended to.
                Defaults to None.

        Returns:
            Three tensors, in order of
            - torch.Tensor: The BERT hidden output, without the bottleneck
                tokens with shape (batch_size, sequence_length, hidden_size).
            - torch.Tensor: The updated bottleneck embeddings after fusion, with
                shape (batch_size, num_bottleneck, hidden_size).
        """
        text_batch, _, text_dim = bert_hidden_states.shape

        (
            bottleneck_batch,
            num_bottleneck_tokens,
            bottleneck_dim,
        ) = bottle_neck.shape

        assert text_batch == bottleneck_batch

        # TODO(liamhebert): Eventually, we will want to uncomment this line
        # once we have static sizes for vision and text inputs.
        # assert vision_batch == text_batch

        bottle_neck = self.bottle_neck_norm(bottle_neck)

        if self.use_projection:
            bert_bottle_neck = self.bottle_to_bert_projection(bottle_neck)
        else:
            assert text_dim == bottleneck_dim, f"{text_dim=}, {bottleneck_dim=}"
            bert_bottle_neck = bottle_neck

        bert_hidden_states_in = torch.cat(
            [bert_bottle_neck, bert_hidden_states], dim=1
        )

        # If we have a custom attention mask, we have to append the bottleneck
        # tokens to the mask as well.
        if bert_attention_mask is not None:
            bert_attention_mask_in = torch.cat(
                (torch.ones_like(bottle_neck[:, :, 0]), bert_attention_mask),
                dim=1,
            )
        else:
            bert_attention_mask_in = None

        bert_hidden_output_out = self.bert_forward(
            bert_hidden_states_in, bert_attention_mask_in, bert_position_ids
        )

        bert_hidden_output = bert_hidden_output_out[:, num_bottleneck_tokens:]
        bottle_neck_output = bert_hidden_output_out[:, :num_bottleneck_tokens]
        if self.use_projection:
            bottle_neck_output = self.bert_to_bottle_projection(
                bottle_neck_output
            )

        return bert_hidden_output, bottle_neck_output

    def bert_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        bert_position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes the BERT layer for the current hidden_state.

        Args:
            hidden_states (torch.Tensor): Tensor of shape (batch_size,
                sequence_length, hidden_size) representing the input hidden
                states.
            attention_mask (torch.Tensor, optional): Tensor of shape
                (batch_size, sequence_length) representing the attention mask to
                avoid attending to padding. Defaults to None.

        Returns:
            torch.Tensor: Tensor of shape (batch_size, sequence_length,
                hidden_size) representing the output hidden states from the BERT
                layer.
        """

        attention_mask_in = None
        if attention_mask is not None:
            assert (
                attention_mask.dim() == 2
            ), f"Expected 2D attention mask, got {attention_mask.shape}"
            bsz, slen = attention_mask.shape
            attention_mask_in = self.get_extended_attention_mask(
                attention_mask, (bsz, slen)  # type: ignore[arg-type]
            )

        if bert_position_ids is not None:
            # if len(hidden_states.shape) == 3:
            #     # If the input is 3D, we need to add a batch dimension
            #     hidden_states = hidden_states.unsqueeze(0)
            # print(hidden_states.shape, attention_mask_in.shape)
            layer_outputs = self.bert_encoder(
                hidden_states,
                attention_mask_in,
            )
        else:
            # if len(hidden_states.shape) == 3:
            #     # If the input is 3D, we need to add a batch dimension
            #     hidden_states = hidden_states.unsqueeze(0)
            # print(hidden_states.shape, attention_mask_in.shape)
            layer_outputs = self.bert_encoder(
                hidden_states,
                attention_mask_in,
            )

        hidden_states = layer_outputs[0]
        return hidden_states


class GraphFusionStack(nn.Module):
    """
    Stack of multiple GraphFusionLayers, executed sequentially.
    """

    def __init__(
        self,
        bert_layers: list[BertLayer],
        use_projection: bool = False,
        bottleneck_dim: int = 768,
        bert_dim: int = 768,
    ) -> None:
        """Constructs a stack of GraphFusionLayers.

        This module is a utility class to run multiple GraphFusionLayers in
        sequence.

        See GraphFusionLayer for more information on how the fusion works.

        Args:
            bert_layers (list[BertLayer]): The list of BERT layers to fuse.
            use_projection (bool, optional): Whether to project the bottleneck
                tokens before passing them into their respective modality
                encoders.  Defaults to False.
        """
        super().__init__()

        self.fusion_layers = nn.ModuleList(
            [
                GraphFusionLayer(
                    bert_layer,
                    use_projection,
                    bottleneck_dim=bottleneck_dim,
                    bert_dim=bert_dim,
                )
                for bert_layer in bert_layers
            ]
        )

    def forward(
        self,
        bert_hidden_states: torch.Tensor,
        bottle_neck: torch.Tensor,
        bert_attention_mask: Optional[torch.FloatTensor] = None,
        bert_position_ids: Optional[torch.FloatTensor] = None,
    ):
        """Computes the stack of text and image layers with bottleneck
        information passing.

        Args:
            bert_hidden_states (torch.Tensor): Tensor of shape (batch_size,
                sequence_length, hidden_size) representing the input hidden
                states for the text model.
            bottle_neck (torch.Tensor): Tensor of shape (batch_size,
                num_bottleneck, hidden_size) representing the hidden state of
                the bottleneck tokens. These tokens are added to both inputs and
                then removed before returning.
            bert_attention_mask (torch.Tensor, optional): Tensor of shape
                (batch_size, sequence_length) representing the attention mask to
                avoid attending to padding. Defaults to None.

        Returns:
            Three tensors, in order of
            - torch.Tensor: The BERT hidden output, without the bottleneck
                tokens with shape (batch_size, sequence_length, hidden_size).
            - torch.Tensor: The new embeddings for the bottleneck tokens, taken
                from the average of the image and text bottleneck embeddings,
                with shape (batch_size, num_bottleneck, hidden_size).
        """
        for f_layer in self.fusion_layers:
            bert_hidden_states, bottle_neck = f_layer(
                bert_hidden_states=bert_hidden_states,
                bottle_neck=bottle_neck,
                bert_attention_mask=bert_attention_mask,
                bert_position_ids=bert_position_ids,
            )

        return bert_hidden_states, bottle_neck

    def __len__(self) -> int:
        """
        Returns the number of layers in the stack.
        """
        return len(self.fusion_layers)
