"""Modules that operate over graphs, such as our transformer."""

from enum import Enum
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import BlockMask
from torch.nn.attention.flex_attention import flex_attention
from utils import RankedLogger
from components.v2.graph_attention_layers import (
    Attention,
    RMSNorm,
    DifferentialAttention,
)

log = RankedLogger(__name__, rank_zero_only=True)

if torch.cuda.is_available():
    flex_attention_comp = torch.compile(flex_attention)
    # flex_attention_comp = flex_attention
else:
    flex_attention_comp = flex_attention


class InitStdFactor(Enum):
    """Initialization Enums"""

    DISABLED = "disabled"  # Init std is divided by 1.0
    GLOBAL_DEPTH = "global_depth"  # Init std is divided by sqrt(2*n_layers)
    CURRENT_DEPTH = "current_depth"  # Init std is divided by sqrt(2*depth)
    DIM_RATIO = "dim_ratio"  # Init std is divided by model_dim/4096


class FeedForward(nn.Module):
    """FeedForward module using SwiGLU activation."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        """Init FeedForward module.

        Computes: `FFN(x) = W2(W3 * Swish(W1(x)))`

        Args:
            dim (int): Input and output dimension.
            hidden_dim (int): Intermediate hidden dimension.
        """
        super().__init__()

        self.dim = dim
        self.hidden_dim = hidden_dim

        self.w1 = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
        )
        self.w3 = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
        )
        self.w2 = nn.Linear(
            hidden_dim,
            dim,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes SwiGLU feedforward.

        Args:
            x (torch.Tensor): Tensor of shape `(..., D)`

        Returns:
            torch.Tensor: Processed tensor of shape `(..., D)`
        """
        # B S D
        # View as returns a copy of x, preserving the gradient
        x1 = self.w1(x.view_as(x))
        x3 = self.w3(x.view_as(x))
        output = self.w2(F.silu(x1) * x3)
        return output

    def reset_parameters(self, init_std=None, factor=1.0):
        """Reset parameters of the FeedForward module."""

        in_init_std = init_std or (self.dim ** (-0.5))
        out_init_std = init_std or (self.hidden_dim ** (-0.5))
        in_init_std = in_init_std / factor
        out_init_std = out_init_std / factor
        for w in [self.w1, self.w3]:
            nn.init.trunc_normal_(
                w.weight,
                mean=0.0,
                std=in_init_std,
                a=-3 * in_init_std,
                b=3 * in_init_std,
            )
        nn.init.trunc_normal_(
            self.w2.weight,
            mean=0.0,
            std=out_init_std,
            a=-3 * out_init_std,
            b=3 * out_init_std,
        )


class GraphTransformerBlock(nn.Module):
    """Transformer block computing multi-head self-attention and feedforward."""

    def __init__(
        self,
        n_heads: int,
        n_kv_heads: int,
        dim: int,
        ffn_dim: int,
        norm_eps: float = 1e-5,
        head_dim: int | None = None,
        depth: int = 0,
        differential_attention: bool = False,
    ):
        super().__init__()

        if head_dim is None:
            self.head_dim = dim // n_heads
        else:
            self.head_dim = head_dim

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads

        assert n_heads % self.n_kv_heads == 0
        assert dim % n_heads == 0
        assert (
            n_heads % 2 == 0
        ), f"Number of heads must be divisible by 2. {self.n_heads=}"
        assert (
            n_kv_heads % 2 == 0
        ), f"Number of heads must be divisible by 2. {self.n_kv_heads=}"
        assert (self.head_dim & (self.head_dim - 1)) == 0, (
            f"Head dim must be a power of 2. "
            f"{self.head_dim=}, {self.n_heads=}, {dim=}"
        )

        self.dim = dim

        if differential_attention:
            self.attention = DifferentialAttention(
                dim=dim,
                head_dim=self.head_dim,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                depth=depth,
            )
        else:
            self.attention = Attention(
                dim=dim,
                head_dim=self.head_dim,
                n_heads=self.n_heads,
                n_kv_heads=self.n_kv_heads,
            )
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=ffn_dim,
        )
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        # freq_cis: torch.Tensor,
        mask: BlockMask | None = None,
    ) -> torch.Tensor:
        h = x + self.attention(
            self.attention_norm(x),
            # freq_cis,
            mask=mask,
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def init_weights(self, init_std=None, factor=1.0):
        """Reset parameters of the Transformer module and submodules."""
        self.attention.reset_parameters(init_std, factor)
        self.attention_norm.reset_parameters()

        self.feed_forward.reset_parameters(init_std, factor)
        self.ffn_norm.reset_parameters()


class BaseGraphTransformer(nn.Module):
    """Module encapsulating a stack of GraphTransformerBlocks."""

    def __init__(
        self,
        num_layers: int,
        graph_tfmr_factory: Callable[[int], GraphTransformerBlock],
        depth: int = 0,
        init_base_std: float | None = None,
        init_std_factor: str | InitStdFactor = "disabled",
    ):
        super().__init__()
        self.init_base_std = init_base_std
        self.init_std_factor = InitStdFactor(init_std_factor)
        self.starting_depth: int = depth * num_layers

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                graph_tfmr_factory(depth=self.starting_depth + i)  # type: ignore
            )

        self.dim = self.layers[0].dim
        self.num_heads = self.layers[0].n_heads

    def __len__(self) -> int:
        """Number of layers in the transformer."""
        return len(self.layers)

    def forward(self, x: torch.Tensor, mask: BlockMask | None = None):
        # TODO(liamhebert): Add rotary positional encoding frequency cis
        # TODO(liamhebert): Consider projecting up before layers and down after
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x

    def init_weights(self):
        """Reset parameters of all Transformer modules and submodules."""
        # self.reset_parameters()
        for depth, layer in enumerate(self.layers):
            assert isinstance(layer, nn.Module)

            depth = self.starting_depth + depth
            factor = {
                InitStdFactor.CURRENT_DEPTH: (2 * (depth + 1)) ** 0.5,
                InitStdFactor.GLOBAL_DEPTH: (
                    2 * (len(self.layers) + 1 + self.starting_depth)
                )
                ** 0.5,
                InitStdFactor.DIM_RATIO: self.dim / 4096,
                InitStdFactor.DISABLED: 1.0,
            }[self.init_std_factor]

            layer.init_weights(self.init_base_std, factor)
