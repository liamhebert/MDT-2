from enum import Enum
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import BlockMask
from torch.nn.attention.flex_attention import flex_attention

if torch.cuda.is_available():
    flex_attention_comp = torch.compile(flex_attention)
else:
    flex_attention_comp = flex_attention


class InitStdFactor(Enum):
    DISABLED = "disabled"  # Init std is divided by 1.0
    GLOBAL_DEPTH = "global_depth"  # Init std is divided by sqrt(2*n_layers)
    CURRENT_DEPTH = "current_depth"  # Init std is divided by sqrt(2*depth)
    DIM_RATIO = "dim_ratio"  # Init std is divided by model_dim/4096


def repeat_kv(x: torch.Tensor, n_rep: int, dim: int) -> torch.Tensor:
    """
    torch.repeat_interleave(x, dim=2, repeats=n_rep)
    """
    assert (
        dim == 1
    ), "Only dim=1 is supported. Check the implementation for other dims."
    slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :]
        .expand(slen, n_kv_heads, n_rep, head_dim)
        .reshape(slen, n_kv_heads * n_rep, head_dim)
    )


class RMSNorm(nn.Module):
    """Initialize the RMSNorm normalization layer.

    Args:
        dim (int): The dimension of the input tensor.
        eps (float, optional): A small value added to the denominator for
            numerical stability. Default is 1e-6.

    Attributes:
        eps (float): A small value added to the denominator for numerical
            stability.
        weight (nn.Parameter): Learnable scaling parameter.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        output = self._norm(x.float())
        return (output * self.weight.float()).type_as(x)

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)  # type: ignore


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        head_dim: int,
        n_heads: int,
        n_kv_heads: int,
        use_biased_attention: bool = False,
    ):
        super().__init__()

        self.dim = dim
        self.head_dim = head_dim

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.heads_per_group = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(
            dim,
            n_heads * head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            dim,
            n_kv_heads * head_dim,
            bias=False,
        )
        self.wv = nn.Linear(
            dim,
            n_kv_heads * head_dim,
            bias=False,
        )

        self.wo = nn.Linear(
            n_heads * head_dim,
            dim,
            bias=False,
        )
        self.use_biased_attention = use_biased_attention

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # B S D
        seq_len, dim = x.shape
        xq = self.wq(x.view_as(x))
        xk = self.wk(x.view_as(x))
        xv = self.wv(x.view_as(x))

        output_shape = xq.shape
        # S D -> S H D
        xq = xq.view(seq_len, self.n_heads, self.head_dim)
        xk = xk.view(seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(seq_len, self.n_kv_heads, self.head_dim)

        xk = repeat_kv(xk, self.heads_per_group, dim=1)
        xv = repeat_kv(xv, self.heads_per_group, dim=1)

        # sdpa requires H S D format
        # S H D -> H S D
        xq, xk, xv = map(lambda e: e.transpose(0, 1), (xq, xk, xv))
        # Add a singleton dimension for the batch
        # H S D -> 1 H S D
        xq, xk, xv = xq.unsqueeze(0), xk.unsqueeze(0), xv.unsqueeze(0)

        if self.use_biased_attention:
            assert mask is None or isinstance(mask, torch.Tensor)

            output = F.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                is_causal=False,
                attn_mask=mask,
            )
        else:
            assert isinstance(mask, BlockMask)

            output = flex_attention_comp(xq, xk, xv, block_mask=mask)

        # 1 H S D -> H S D
        xq, xk, xv = xq.squeeze(0), xk.squeeze(0), xv.squeeze(0)

        # H S D -> S H D
        output = output.transpose(0, 1).contiguous()

        output = self.wo(output.reshape(output_shape))

        return output

    def reset_parameters(self, init_std=None, factor=1.0):
        init_std = init_std or (self.dim ** (-0.5))

        for w in [self.wq, self.wk, self.wv]:
            nn.init.trunc_normal_(
                w.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )

        nn.init.trunc_normal_(
            self.wo.weight,
            mean=0.0,
            std=init_std / factor,
            a=-3 * init_std,
            b=3 * init_std,
        )


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
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
        # B S D
        # View as returns a copy of x, preserving the gradient
        x1 = self.w1(x.view_as(x))
        x3 = self.w3(x.view_as(x))
        output = self.w2(F.silu(x1) * x3)
        return output

    def reset_parameters(self, init_std=None, factor=1.0):
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
    def __init__(
        self,
        n_heads: int,
        n_kv_heads: int,
        dim: int,
        ffn_dim: int,
        norm_eps: float = 1e-5,
    ):
        super().__init__()

        self.head_dim = dim // n_heads
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
        assert self.head_dim % 2 == 0, (
            f"Head dim must be divisible by 2. "
            f"{self.head_dim=}, {self.n_heads=}, {dim=}"
        )

        self.dim = dim

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
        self.attention.reset_parameters(init_std, factor)
        self.attention_norm.reset_parameters()

        self.feed_forward.reset_parameters(init_std, factor)
        self.ffn_norm.reset_parameters()


class BaseGraphTransformer(nn.Module):

    def __init__(
        self,
        num_layers: int,
        graph_tfmr_factory: Callable[[], GraphTransformerBlock],
        depth: int = 0,
        init_base_std: float | None = None,
        init_std_factor: str | InitStdFactor = "disabled",
    ):
        super().__init__()
        self.init_base_std = init_base_std
        self.init_std_factor = InitStdFactor(init_std_factor)
        self.starting_depth: int = depth * num_layers

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(graph_tfmr_factory())

        self.dim = self.layers[0].dim
        self.num_heads = self.layers[0].n_heads

    def __len__(self) -> int:
        return len(self.layers)

    def forward(self, x: torch.Tensor, mask: BlockMask | None = None):
        # TODO(liamhebert): Add rotary positional encoding frequency cis
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x

    def init_weights(self):
        # self.reset_parameters()
        for depth, layer in enumerate(self.layers):
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
