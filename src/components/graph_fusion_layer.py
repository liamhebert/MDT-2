"""
Layers associated with computing multi-modal embeddings.
"""

from typing import Optional

import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertLayer
from transformers.models.vit.modeling_vit import ViTLayer


class GraphFusionLayer(nn.Module):
    """
    Fuses text and image inputs using a shared set of bottleneck tokens.
    """

    bert_encoder: BertLayer
    vit_encoder: ViTLayer
    gradient_checkpointing: bool

    def __init__(
        self,
        bert_layer: BertLayer,
        vit_layer: ViTLayer,
        use_projection: bool = False,
    ) -> None:
        """Initializes the GraphFusionLayer module.

        This module fuses the text and image inputs using a shared set of
        bottleneck tokens. In practice, this works by concatenating the set of
        bottleneck tokens to the input embeddings for each modality. Then, the
        concatenated embeddings are passed through the respective encoders.
        After encoding, the bottleneck tokens are averaged between both
        modalities.

        NOTE: If there are no image inputs, the vit_layer is ignored and the
        bottleneck tokens will consist of only the bert_layer outputs.

        Args:
            bert_layer (BertLayer): The BERT layer to use for the text inputs
            vit_layer (ViTLayer): The ViT layer to use for the image inputs
            use_projection (bool, optional): Whether to project the bottleneck
                embeddings into modality specific versions before concatenating
                them. Currently not implemented. Defaults to False.
        """
        super().__init__()

        self.bert_encoder = bert_layer
        self.vit_encoder = vit_layer
        self.gradient_checkpointing = False
        if use_projection:
            self.bert_projection: nn.Module = nn.Linear(768, 768)
            self.vit_projection: nn.Module = nn.Linear(768, 768)
            raise NotImplementedError("Projection not implemented")
        else:
            self.bert_projection: nn.Module = nn.Identity()
            self.vit_projection: nn.Module = nn.Identity()

    def forward(
        self,
        bert_hidden_states: torch.Tensor,
        vit_hidden_states: torch.Tensor,
        bottle_neck: torch.Tensor,
        bert_attention_mask: Optional[torch.FloatTensor] = None,
        x_image_indexes: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes both modality layers with bottle_neck information passing.

        Args:
            bert_hidden_states (torch.Tensor): Tensor of shape (batch_size,
                sequence_length, hidden_size) representing the input hidden
                states for the text model.
            vit_hidden_states (torch.Tensor): Tensor of shape (batch_size,
                sequence_length, hidden_size) representing the input hidden
                states for the vision model.
            bottle_neck (torch.Tensor): Tensor of shape (batch_size,
                num_bottleneck, hidden_size) representing the hidden state of
                the bottleneck tokens. These tokens are added to both inputs and
                then removed before returning.
            bert_attention_mask (torch.Tensor, optional): Tensor of shape
                (batch_size, sequence_length) representing the attention mask to
                avoid attending to padding. Defaults to None.
            x_image_indexes (torch.Tensor, optional): Boolean tensor of shape
                (batch_size) indicating whether that position has an image or
                not. Defaults to None.

        Returns:
            Three tensors, in order of
            - torch.Tensor: The BERT hidden output, without the bottleneck
                tokens with shape (batch_size, sequence_length, hidden_size).
            - torch.Tensor: The ViT hidden output, without the bottleneck tokens
                with shape (batch_size, sequence_length, hidden_size).
            - torch.Tensor: The new embeddings for the bottleneck tokens, taken
                from the average of the image and text bottleneck embeddings,
                with shape (batch_size, num_bottleneck, hidden_size).
        """
        text_batch, _, text_dim = bert_hidden_states.shape
        vision_batch, _, vision_dim = vit_hidden_states.shape
        (
            bottleneck_batch,
            num_bottleneck_tokens,
            bottleneck_dim,
        ) = bottle_neck.shape
        assert text_dim == vision_dim == bottleneck_dim
        assert text_batch == vision_batch == bottleneck_batch

        bert_hidden_states_in = torch.cat(
            [bottle_neck, bert_hidden_states], dim=1
        )

        bert_hidden_output_out = self.bert_forward(
            bert_hidden_states_in, bert_attention_mask, None, None, None
        )

        bert_hidden_output = bert_hidden_output_out[:, num_bottleneck_tokens:]
        bottle_neck_output = bert_hidden_output_out[:, :num_bottleneck_tokens]

        # TODO(liamhebert): Check how this behaves when some images are present
        # and others are not.
        if vit_hidden_states is not None:
            vit_hidden_states_in = torch.cat(
                [bottle_neck[x_image_indexes], vit_hidden_states], dim=1
            )
            vit_hidden_output_out = self.vit_forward(vit_hidden_states_in)
            vit_hidden_output = vit_hidden_output_out[:, num_bottleneck_tokens:]
            bottle_neck_output[x_image_indexes] = (
                vit_hidden_output_out[:, :num_bottleneck_tokens]
                + bottle_neck_output[x_image_indexes]
            ) / 2

        else:
            vit_hidden_output = None

        return bert_hidden_output, vit_hidden_output, bottle_neck_output

    def vit_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Computes the vision layer for the current hidden_state.

        A large reason for this layer is to enable gradient checkpointing during
        training, allowing for more memory efficient training.

        Args:
            hidden_states (torch.Tensor): Tensor of shape (batch_size,
                sequence_length, hidden_size) representing the input hidden
                states for the vision model.

        Returns:
            torch.Tensor: The result of the image layer, with shape
                (Batch, Tokens, Embed)
        """
        # TODO(liamhebert): Check whether we need additional tokens here, feels
        # odd that we have no mask here.

        # if output_hidden_states:
        #     all_hidden_states = all_hidden_states + (hidden_states,)

        layer_head_mask = None

        if self.gradient_checkpointing and self.training:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs, False)

                return custom_forward

            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.vit_encoder),
                hidden_states,
                layer_head_mask,
            )
        else:
            layer_outputs = self.vit_encoder(
                hidden_states, layer_head_mask, False
            )

        hidden_states = layer_outputs[0]

        return hidden_states

    def bert_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        """Computes the BERT layer for the current hidden_state.

        Args:
            hidden_states (torch.Tensor): Tensor of shape (batch_size,
                sequence_length, hidden_size) representing the input hidden
                states.
            attention_mask (torch.Tensor, optional): Tensor of shape
                (batch_size, sequence_length) representing the attention mask to
                avoid attending to padding. Defaults to None.
            head_mask (torch.Tensor, optional): Tensor of shape (num_heads,)
                representing the head mask to nullify selected heads of the
                self-attention modules. Defaults to None.
            encoder_hidden_states (torch.Tensor, optional): Tensor of shape
                (batch_size, sequence_length, hidden_size) representing the
                encoder hidden states for cross-attention. Defaults to None.
            encoder_attention_mask (torch.Tensor, optional): Tensor of shape
                (batch_size, sequence_length) representing the attention mask
                for the cross-attention encoder hidden states. Defaults to None.

        Returns:
            torch.Tensor: Tensor of shape (batch_size, sequence_length,
                hidden_size) representing the output hidden states from the BERT
                layer.
        """
        layer_head_mask = head_mask if head_mask is not None else None
        past_key_value = None

        if self.gradient_checkpointing and self.training:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs, past_key_value, False)

                return custom_forward

            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.bert_encoder),
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
            )
        else:
            layer_outputs = self.bert_encoder(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                False,
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
        vit_layers: list[ViTLayer],
        use_projection: bool = False,
    ) -> None:
        """Constructs a stack of GraphFusionLayers.

        This module is a utility class to run multiple GraphFusionLayers in
        sequence.

        See GraphFusionLayer for more information on how the fusion works.

        Args:
            bert_layers (list[BertLayer]): The list of BERT layers to fuse.
            vit_layers (list[BertLayer]): The list of ViT layers to fuse.
            use_projection (bool, optional): Whether to project the bottleneck
                tokens before passing them into their respective modality
                encoders.  Defaults to False.
        """
        super().__init__()

        self.fusion_layers = nn.ModuleList(
            [
                GraphFusionLayer(
                    bert_layer,
                    vit_layer,
                    use_projection,
                )
                for bert_layer, vit_layer in zip(bert_layers, vit_layers)
            ]
        )

    def forward(
        self,
        bert_hidden_states: torch.Tensor,
        vit_hidden_states: torch.Tensor,
        bottle_neck: torch.Tensor,
        bert_attention_mask: Optional[torch.FloatTensor] = None,
        x_image_indexes: Optional[torch.Tensor] = None,
    ):
        """Computes the stack of text and image layers with bottleneck
        information passing.

        Args:
            bert_hidden_states (torch.Tensor): Tensor of shape (batch_size,
                sequence_length, hidden_size) representing the input hidden
                states for the text model.
            vit_hidden_states (torch.Tensor): Tensor of shape (batch_size,
                sequence_length, hidden_size) representing the input hidden
                states for the vision model.
            bottle_neck (torch.Tensor): Tensor of shape (batch_size,
                num_bottleneck, hidden_size) representing the hidden state of
                the bottleneck tokens. These tokens are added to both inputs and
                then removed before returning.
            bert_attention_mask (torch.Tensor, optional): Tensor of shape
                (batch_size, sequence_length) representing the attention mask to
                avoid attending to padding. Defaults to None.
            x_image_indexes (torch.Tensor, optional): Boolean tensor of shape
                (batch_size) indicating whether that position has an image or
                not. Defaults to None.

        Returns:
            Three tensors, in order of
            - torch.Tensor: The BERT hidden output, without the bottleneck
                tokens with shape (batch_size, sequence_length, hidden_size).
            - torch.Tensor: The ViT hidden output, without the bottleneck tokens
                with shape (batch_size, sequence_length, hidden_size).
            - torch.Tensor: The new embeddings for the bottleneck tokens, taken
                from the average of the image and text bottleneck embeddings,
                with shape (batch_size, num_bottleneck, hidden_size).
        """
        for f_layer in self.fusion_layers:
            bert_hidden_states, vit_hidden_states, bottle_neck = f_layer(
                bert_hidden_states,
                vit_hidden_states,
                bottle_neck,
                bert_attention_mask,
                x_image_indexes,
            )

        return bert_hidden_states, vit_hidden_states, bottle_neck
