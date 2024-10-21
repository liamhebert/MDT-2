"""
Custom attention modules to assist with adding aux features during attention.
"""

import math
from typing import Optional

import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F


class MultiheadAttention(nn.Module):
    """Custom Multi-Head Attention (MHA) which adds bias features during
    attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        """Initializes the Custom MHA module.

        This adds an extra attention_bias to the attention computation, provided
        by `attn_bias` which has a shape of (num_heads).

        Args:
            embed_dim (int): Embedding dimension of the input for each head.
            num_heads (int): Number of heads to split the input into.
            dropout (float, optional): Dropout probability for the attention
            values. Defaults to 0.0.
            bias (bool, optional): Whether to add a bias term to the key, value,
                query projections. Defaults to True.
        """
        super().__init__()
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.dropout_module = nn.Dropout(dropout)

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initializes all parameters to fit Xavier uniform.

        Empirically observed the convergence to be much better with the scaled
        initialization.
        """
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query: Tensor,
        attn_bias: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Input shape: Batch x Comments x Embed

        Args:
            query (Tensor): Tensor with shape (batch, nodes, embed_dim) to apply
                the MHA layer to.
            attn_bias (Tensor): Tensor with shape (batch, heads, nodes, nodes)
                containing the bias to apply to the attention scores between
                nodes.
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where padding
                elements are indicated by 1s.
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from attending to those positions.

        Returns:
            Tensor: The new hidden state after processed by the MHA layer.
        """

        bsz, tgt_len, embed_dim = query.size()
        # Since MHA is computed [T, B, E], we need to transpose our input to
        # match
        query = query.transpose(1, 0)

        assert (
            embed_dim == self.embed_dim
        ), f"query dim {embed_dim} != {self.embed_dim}"

        q = self.q_proj(query)
        k = self.k_proj(query)
        v = self.v_proj(query)
        q *= self.scaling

        q = (
            q.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )

        k = (
            k.contiguous()
            .view(-1, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )

        v = (
            v.contiguous()
            .view(-1, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )

        assert k.size(1) == tgt_len

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == tgt_len

        attn_weights = torch.bmm(q, k.transpose(1, 2))

        assert list(attn_weights.size()) == [
            bsz * self.num_heads,
            tgt_len,
            tgt_len,
        ]

        if attn_bias is not None:
            attn_weights += attn_bias.view(
                bsz * self.num_heads, tgt_len, tgt_len
            )

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(
                bsz, self.num_heads, tgt_len, tgt_len
            )
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )
            attn_weights = attn_weights.view(
                bsz * self.num_heads, tgt_len, tgt_len
            )

        attn_weights_float = F.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        assert v is not None
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [
            bsz * self.num_heads,
            tgt_len,
            self.head_dim,
        ]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        # We then restore the original shape of the tensor
        attn = attn.transpose(1, 0)

        return attn
