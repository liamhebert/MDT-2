"""
Layers associated with computing graph embeddings.
"""

from typing import Optional

import torch
import torch.nn as nn

from components.v1.custom_attn import MultiheadAttention


class GraphormerGraphEncoderLayer(nn.Module):
    """
    Layer which uses a transformer to aggregate graph node embeddings.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: nn.Module = nn.ReLU(),
        pre_layernorm: bool = False,
    ) -> None:
        """Constructs a Graph Transformer encoder layer.

        Args:
            embedding_dim (int, optional): The input embedding dimension.
                Defaults to 768.
            ffn_embedding_dim (int, optional): The dimension of the feedforward
                network used in attention. Defaults to 3072.
            num_attention_heads (int, optional): The number of attention heads.
                Must divide embedding_dim. Defaults to 8.
            dropout (float, optional): The input dropout probability. Defaults
                to 0.1.
            attention_dropout (float, optional): The dropout probability for
                attention weights. Defaults to 0.1.
            activation_dropout (float, optional): Dropout probability to apply
                between output ffn layers. Defaults to 0.1.
            activation_fn (nn.Module, optional): Activation function used before
                output ffn. Defaults to nn.ReLU().
            pre_layernorm (bool, optional): Whether to apply layer-norm before
                self attention and output ffns. Defaults to False.
        """
        super().__init__()

        # Initialize parameters
        self.pre_layernorm = pre_layernorm

        self.dropout_module = nn.Dropout(dropout)
        self.activation_dropout_module = nn.Dropout(activation_dropout)

        # Initialize blocks
        self.activation_fn = activation_fn
        self.self_attn = MultiheadAttention(
            embedding_dim, num_attention_heads, dropout=attention_dropout
        )

        # Layer norm associated with the self-attention layer
        self.self_attn_layer_norm = nn.LayerNorm(embedding_dim)

        self.fc1 = nn.Linear(embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, embedding_dim)

        # Layer norm associated with the position-wise feed-forward NN
        self.final_layer_norm = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
        self_attn_bias: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for the GraphormerGraphEncoderLayer.

        Args:
            x (torch.Tensor): Tensor with shape (batch, nodes, embed_dim),
                representing the input node embeddings.
            self_attn_bias (torch.Tensor, optional): Tensor with shape (batch,
                heads, nodes, nodes) containing the bias to apply to the
                attention scores between nodes.
            self_attn_mask (torch.Tensor, optional): If specified, a 2D or 3D
                mask preventing attention to certain positions. Must be of shape
                (nodes, nodes) or (batch * num_heads, nodes, nodes). A 2D mask
                will be broadcasted across the batch while a 3D mask allows for
                a different mask for each entry in the batch. Binary and float
                masks are supported. For a binary mask, a True value indicates
                that the corresponding position is not allowed to attend. For a
                float mask, the mask values will be added to the attention weight.
                If both attn_mask and key_padding_mask are supplied, their types
                should match.
            self_attn_padding_mask (Optional[torch.Tensor]): If specified, a
                binary mask of shape (batch, nodes) indicating which elements
                within key to ignore for the purpose of attention (i.e. treat as
                “padding”). A True value indicates that the corresponding key
                value will be ignored for the purpose of attention. For a float
                mask, it will be directly added to the corresponding key value.

        Returns:
            torch.Tensor: Output tensor after applying the encoder layer, with
                shape (batch, nodes, embed_dim).
        """
        residual = x
        if self.pre_layernorm:
            x = self.self_attn_layer_norm(x)
        x = self.self_attn(
            query=x,
            attn_bias=self_attn_bias,
            key_padding_mask=self_attn_padding_mask,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = residual + x
        if not self.pre_layernorm:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.pre_layernorm:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x
        if not self.pre_layernorm:
            x = self.final_layer_norm(x)
        return x


class GraphEncoderStack(nn.Module):
    """
    Stack of multiple GraphEncoderLayers, executed sequentially.
    """

    layers: nn.ModuleList

    def __init__(
        self,
        num_layers: int,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: nn.Module = nn.ReLU(),
    ) -> None:
        """Constructs a stack of GraphEncoderLayers.

        This module is a utility class to run multiple GraphEncoderLayers in
        sequence.

        See GraphEncoderLayer for more information on how the fusion works.

        Args:
            num_layers (int): The number of consecutive GraphEncoderLayers to
                stack.
            embedding_dim (int, optional): The input embedding dimension.
                Defaults to 768.
            ffn_embedding_dim (int, optional): The dimension of the feedforward
                network used in attention. Defaults to 3072.
            num_attention_heads (int, optional): The number of attention heads.
                Must divide embedding_dim. Defaults to 8.
            dropout (float, optional): The input dropout probability. Defaults
                to 0.1.
            attention_dropout (float, optional): The dropout probability for
                attention weights. Defaults to 0.1.
            activation_dropout (float, optional): Dropout probability to apply
                between output ffn layers. Defaults to 0.1.
            activation_fn (nn.Module, optional): Activation function used before
                output ffn. Defaults to nn.ReLU().
            pre_layernorm (bool, optional): Whether to apply layer-norm before
                self attention and output ffns. Defaults to False.
        """
        super().__init__()
        self.layers = nn.ModuleList(
            [
                GraphormerGraphEncoderLayer(
                    embedding_dim,
                    ffn_embedding_dim,
                    num_attention_heads,
                    dropout,
                    attention_dropout,
                    activation_dropout,
                    activation_fn,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        self_attn_bias: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for the GraphEncoderStack.

        Args:
            x (torch.Tensor): Tensor with shape (batch, nodes, embed_dim),
                representing the input node embeddings.
            self_attn_bias (torch.Tensor, optional): Tensor with shape (batch,
                heads, nodes, nodes) containing the bias to apply to the
                attention scores between nodes.
            self_attn_mask (torch.Tensor, optional): If specified, a 2D or 3D
                mask preventing attention to certain positions. Must be of shape
                (nodes, nodes) or (batch * num_heads, nodes, nodes). A 2D mask
                will be broadcasted across the batch while a 3D mask allows for
                a different mask for each entry in the batch. Binary and float
                masks are supported. For a binary mask, a True value indicates
                that the corresponding position is not allowed to attend. For a
                float mask, the mask values will be added to the attention weight.
                If both attn_mask and key_padding_mask are supplied, their types
                should match.
            self_attn_padding_mask (Optional[torch.Tensor]): If specified, a
                binary mask of shape (batch, nodes) indicating which elements
                within key to ignore for the purpose of attention (i.e. treat as
                “padding”). A True value indicates that the corresponding key
                value will be ignored for the purpose of attention. For a float
                mask, it will be directly added to the corresponding key value.

        Returns:
            torch.Tensor: Output tensor after applying the stack of encoder
                layers.
        """
        for layer in self.layers:
            x = layer(x, self_attn_bias, self_attn_mask, self_attn_padding_mask)
        return x

    def __len__(self) -> int:
        """
        Returns the number of layers in the stack.
        """
        return len(self.layers)
