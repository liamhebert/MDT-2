"""
Layers associated with computing multi-modal embeddings.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers.modeling_utils import ModuleUtilsMixin
from transformers.models.bert.modeling_bert import BertLayer
from transformers.models.vit.modeling_vit import ViTLayer
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
    """
    Fuses text and image inputs using a shared set of bottleneck tokens.
    """

    bert_encoder: BertLayer
    vit_encoder: ViTLayer
    gradient_checkpointing: bool
    bert_projection: nn.Module
    vit_projection: nn.Module
    bottle_neck_norm: RMSNorm

    def __init__(
        self,
        bert_layer: BertLayer,
        vit_layer: ViTLayer,
        use_projection: bool = False,
        bottleneck_dim: int = 768,
        bert_dim: int = 768,
        vit_dim: int = 768,
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
        self.use_projection = use_projection
        if use_projection:
            # TODO(liamhebert): This should be tuned to the specific use case.
            self.bottle_to_bert_projection = nn.Linear(bottleneck_dim, bert_dim)
            self.bottle_to_vit_projection = nn.Linear(bottleneck_dim, vit_dim)
            self.bert_to_bottle_projection = nn.Linear(bert_dim, bottleneck_dim)
            self.vit_to_bottle_projection = nn.Linear(vit_dim, bottleneck_dim)

        self.bottle_neck_norm = RMSNorm(bottleneck_dim)

    @torch.compiler.disable
    def forward(
        self,
        bert_hidden_states: torch.Tensor,
        vit_hidden_states: torch.Tensor,
        bottle_neck: torch.Tensor,
        image_padding_mask: torch.Tensor,
        bert_attention_mask: Optional[torch.Tensor] = None,
        bert_position_ids: Optional[torch.Tensor] = None,
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
            image_padding_mask (torch.Tensor): Boolean tensor of shape
                (batch_size) indicating whether that position has an image or
                not.
            bert_attention_mask (torch.Tensor, optional): Tensor of shape
                (batch_size, sequence_length) representing the attention mask to
                avoid attending to padding. Tokens with 1 are *not masked* and
                tokens with 0 are *masked*. If None, all tokens are attended to.
                Defaults to None.

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

        # TODO(liamhebert): Check how this behaves when some images are present
        # and others are not.
        if vit_hidden_states is not None:
            assert image_padding_mask.dtype == torch.bool, (
                f"Mask must be bool, got {image_padding_mask=}, "
                f"{image_padding_mask.dtype=}"
            )
            img_bottle_neck = bottle_neck[image_padding_mask]

            if self.use_projection:
                img_bottle_neck = self.bottle_to_vit_projection(img_bottle_neck)
            else:
                _, _, vision_dim = vit_hidden_states.shape
                assert (
                    vision_dim == bottleneck_dim
                ), f"{vision_dim=}, {bottleneck_dim=}"

            vit_hidden_states_in = torch.cat(
                [img_bottle_neck, vit_hidden_states], dim=1
            )

            vit_hidden_output_out = self.vit_forward(vit_hidden_states_in)
            vit_hidden_output = vit_hidden_output_out[:, num_bottleneck_tokens:]
            vit_bot_output = vit_hidden_output_out[:, :num_bottleneck_tokens]
            if self.use_projection:
                vit_bot_output = self.vit_to_bottle_projection(vit_bot_output)

            # Initialize image_bottleneck_tokens with the full batch size
            updated_tokens = bottle_neck_output.clone()
            # Calculate the averaged bottleneck tokens *only* for samples with
            # images
            # Note: This part is still indexing, but not in-place assignment
            image_bottleneck_tokens = (
                vit_bot_output + bottle_neck_output[image_padding_mask]
            ) / 2
            # Assign the calculated averaged tokens to the correct rows of
            # image_bottleneck_tokens

            updated_tokens[image_padding_mask] = image_bottleneck_tokens

            expanded_image_mask = image_padding_mask.view(-1, 1, 1).expand_as(
                bottle_neck_output
            )

            bottle_neck_with_images = torch.where(
                expanded_image_mask,
                updated_tokens,
                bottle_neck_output,
            )
            bottle_neck_output = bottle_neck_with_images

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

        layer_outputs = self.vit_encoder(hidden_states, layer_head_mask, False)

        hidden_states = layer_outputs[0]

        return hidden_states

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
        past_key_value = None

        attention_mask_in = (
            self.get_extended_attention_mask(
                attention_mask, attention_mask.shape
            )
            if attention_mask is not None
            else None
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
        vit_layers: list[ViTLayer],
        use_projection: bool = False,
        bottleneck_dim: int = 768,
        bert_dim: int = 768,
        vit_dim: int = 768,
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
                    bottleneck_dim=bottleneck_dim,
                    bert_dim=bert_dim,
                    vit_dim=vit_dim,
                )
                for bert_layer, vit_layer in zip(bert_layers, vit_layers)
            ]
        )

    def forward(
        self,
        bert_hidden_states: torch.Tensor,
        vit_hidden_states: torch.Tensor,
        bottle_neck: torch.Tensor,
        image_padding_mask: torch.Tensor,
        bert_attention_mask: Optional[torch.FloatTensor] = None,
        bert_position_ids: Optional[torch.FloatTensor] = None,
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
            image_padding_mask (torch.Tensor, optional): Boolean tensor of shape
                (batch_size) indicating whether that position has an image or
                not. Defaults to None.
            bert_attention_mask (torch.Tensor, optional): Tensor of shape
                (batch_size, sequence_length) representing the attention mask to
                avoid attending to padding. Defaults to None.

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
                bert_hidden_states=bert_hidden_states,
                vit_hidden_states=vit_hidden_states,
                bottle_neck=bottle_neck,
                image_padding_mask=image_padding_mask,
                bert_attention_mask=bert_attention_mask,
                bert_position_ids=bert_position_ids,
            )

        return bert_hidden_states, vit_hidden_states, bottle_neck

    def __len__(self) -> int:
        """
        Returns the number of layers in the stack.
        """
        return len(self.fusion_layers)
