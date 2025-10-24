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


class BottleneckGateMLP(nn.Module):
    """Contextual gate for fusing text/image bottleneck tokens.

    Given per-token text and image bottleneck embeddings, predicts a gate g in
    [0,1] to combine them: fused = (1 - g) * text + g * image. The gate can be
    scalar per token or per-dimension depending on configuration.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: Optional[int] = None,
        per_dim: bool = False,
    ) -> None:
        super().__init__()
        out_dim = embed_dim if per_dim else 1
        hidden = hidden_dim or (4 * embed_dim)
        self.net = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden, bias=True),
            nn.SiLU(),
            nn.Linear(hidden, out_dim, bias=True),
            nn.Sigmoid(),
        )

    def forward(
        self, text_bn: torch.Tensor, img_bn: torch.Tensor
    ) -> torch.Tensor:
        # text_bn/img_bn: (B_img, T, D)
        x = torch.cat([text_bn, img_bn], dim=-1)
        g = self.net(x)  # (B_img, T, 1 or D)
        return g


class GraphFusionLayer(nn.Module, ModuleUtilsMixinWrapper):
    """Fuse text and image inputs with bottleneck tokens and optional gating.

    The layer concatenates a set of bottleneck tokens to each modality's hidden
    states, applies the respective encoder layer, then extracts the updated
    bottleneck tokens and fuses the text/image bottleneck representations. When
    gating is enabled, fusion is computed as per-token
    fused = (1 - g) * text_bn + g * image_bn, with g in [0,1]; otherwise a
    simple average is used. Fusion only applies to rows with images present as
    indicated by ``image_padding_mask``.
    """

    bert_encoder: nn.Module
    vit_encoder: nn.Module
    gradient_checkpointing: bool
    bert_projection: nn.Module
    vit_projection: nn.Module
    bottle_neck_norm: RMSNorm
    use_gating: bool
    gate_per_dim: bool
    gate_mlp: Optional[nn.Module]

    def __init__(
        self,
        bert_layer: nn.Module,
        vit_layer: nn.Module,
        use_projection: bool = False,
        bottleneck_dim: int = 768,
        bert_dim: int = 768,
        vit_dim: int = 768,
        use_gating: bool = False,
        gate_per_dim: bool = False,
        gate_hidden_dim: Optional[int] = None,
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
            vit_layer (ViTLayer): The ViT layer to use for the image inputs
            use_projection (bool, optional): Whether to project the bottleneck
                embeddings into modality specific versions before concatenating
                them. Defaults to False.
            bert_dim (int, optional): The hidden size of the BERT layer. Used
                when `use_projection` is True.
            vit_dim (int, optional): The hidden size of the ViT layer. Used
                when `use_projection` is True.
            bottleneck_dim (int, optional): The hidden size of the bottleneck
                tokens.
            use_gating (bool, optional): If True, enable a learned gate to fuse
                text/image bottleneck tokens instead of simple averaging.
            gate_per_dim (bool, optional): If True, predict a per-dimension gate
                vector; else predict a scalar gate per token.
            gate_hidden_dim (int, optional): Hidden size of the gating MLP.
        """
        super().__init__()

        self.bert_encoder = bert_layer
        self.vit_encoder = vit_layer
        self.gradient_checkpointing = False
        self.use_projection = use_projection
        self.use_gating = use_gating
        self.gate_per_dim = gate_per_dim
        if use_projection:
            # TODO(liamhebert): This should be tuned to the specific use case.
            self.bottle_to_bert_projection = nn.Linear(bottleneck_dim, bert_dim)
            self.bottle_to_vit_projection = nn.Linear(bottleneck_dim, vit_dim)
            self.bert_to_bottle_projection = nn.Linear(bert_dim, bottleneck_dim)
            self.vit_to_bottle_projection = nn.Linear(vit_dim, bottleneck_dim)

        self.bottle_neck_norm = RMSNorm(bottleneck_dim)
        if self.use_gating:
            self.gate_mlp = BottleneckGateMLP(
                embed_dim=bottleneck_dim,
                hidden_dim=gate_hidden_dim,
                per_dim=gate_per_dim,
            )
        else:
            self.gate_mlp = None

    @torch.compiler.disable
    def forward(
        self,
        bert_hidden_states: torch.Tensor,
        vit_hidden_states: torch.Tensor,
        bottle_neck: torch.Tensor,
        image_padding_mask: torch.Tensor,
        bert_attention_mask: Optional[torch.Tensor] = None,
        bert_position_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Compute both modality layers with bottleneck information passing.

        Args:
            bert_hidden_states (torch.Tensor): Tensor of shape (batch_size,
                sequence_length, hidden_size) representing the input hidden
                states for the text model.
            vit_hidden_states (torch.Tensor): Tensor of shape (batch_size,
                sequence_length, hidden_size) representing the input hidden
                states for the vision model. If None, the vision path is skipped
                and the returned ViT output will be None.
            bottle_neck (torch.Tensor): Tensor of shape (batch_size,
                num_bottleneck, hidden_size) representing the hidden state of
                the bottleneck tokens. These tokens are added to both inputs and
                then removed before returning.
            image_padding_mask (torch.Tensor): Boolean tensor of shape
                (batch_size,) indicating which rows have images (True). Used to
                select rows for the vision path and for fusing bottleneck tokens.
            bert_attention_mask (torch.Tensor, optional): Tensor of shape
                (batch_size, sequence_length) representing the attention mask to
                avoid attending to padding. Tokens with 1 are *not masked* and
                tokens with 0 are *masked*. If None, all tokens are attended to.
                Defaults to None.
            bert_position_ids (torch.Tensor, optional): Tensor of shape
                (batch_size, sequence_length) representing the position ids for
                the BERT layer. Defaults to None.

        Returns:
            Three tensors, in order of
            - torch.Tensor: The BERT hidden output, without the bottleneck
                tokens with shape (batch_size, sequence_length, hidden_size).
            - Optional[torch.Tensor]: The ViT hidden output, without the
                bottleneck tokens with shape (batch_size, sequence_length,
                hidden_size) if images are present; otherwise None.
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
            vit_bottleneck_output = vit_hidden_output_out[
                :, :num_bottleneck_tokens
            ]
            if self.use_projection:
                vit_bottleneck_output = self.vit_to_bottle_projection(
                    vit_bottleneck_output
                )

            # Initialize image_bottleneck_tokens with the full batch size
            updated_tokens = bottle_neck_output.clone()
            # Calculate the averaged bottleneck tokens *only* for samples with
            # images
            # Note: This part is still indexing, but not in-place assignment
            image_subset = bottle_neck_output[image_padding_mask]
            # Average the bottleneck tokens from both modalities
            assert vit_bottleneck_output.shape == image_subset.shape, (
                f"{vit_bottleneck_output.shape=}, {image_subset.shape=},"
                f" {image_padding_mask=}"
            )
            if self.use_gating and self.gate_mlp is not None:
                # Gate is in [0,1]; if per-dim, it will be broadcast along last
                # dim.
                gate = self.gate_mlp(image_subset, vit_bottleneck_output)
                if gate.dim() == 3 and gate.shape[-1] == 1:
                    # broadcast scalar gate per token
                    gate = gate.expand_as(image_subset)
                image_bottleneck_tokens = (
                    1.0 - gate
                ) * image_subset + gate * vit_bottleneck_output
            else:
                image_bottleneck_tokens = (
                    vit_bottleneck_output + image_subset
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

        if isinstance(self.vit_encoder, ViTLayer):
            layer_outputs = self.vit_encoder(hidden_states)
        else:
            layer_outputs = self.vit_encoder(hidden_states, attention_mask=None)
        if isinstance(layer_outputs, tuple):
            # TODO(liamhebert): I have no idea why sometimes it's a tuple and
            # other times it is not. Putting this here to fix that issue.
            layer_outputs = layer_outputs[0]
        return layer_outputs

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
            bert_position_ids (torch.Tensor, optional): Tensor of shape
                (batch_size, sequence_length) representing the position ids for
                the BERT layer. Defaults to None.

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
        vit_layers: list[ViTLayer],
        use_projection: bool = False,
        bottleneck_dim: int = 768,
        bert_dim: int = 768,
        vit_dim: int = 768,
        use_gating: bool = False,
        gate_per_dim: bool = False,
        gate_hidden_dim: int | None = None,
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
                    use_gating=use_gating,
                    gate_per_dim=gate_per_dim,
                    gate_hidden_dim=gate_hidden_dim,
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
