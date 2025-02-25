"""Custom attention layers for graph encoders, such as RoPe and DiffAttn."""

from torch import nn
import torch.nn.functional as F
import torch
from torch.nn.attention.flex_attention import BlockMask
from torch.nn.attention.flex_attention import flex_attention
import math
from components.v2.graph_rope_encoding import RoPE

if torch.cuda.is_available():
    flex_attention_comp = torch.compile(flex_attention)
    # flex_attention_comp = flex_attention
else:
    flex_attention_comp = flex_attention


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
    """RMSNorm normalization layer with a learnable scaling parameter.

    Notably, this layer includes a learable scaling weight that is applied
    elementwise to the normalized tensor.

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
        """Initialize the RMSNorm normalization layer."""
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        """Computes the RMSNorm without the scaling parameter."""
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        """Applies the RMSNorm normalization to the input tensor."""
        output = self._norm(x)
        return (output * self.weight).type_as(x)

    def reset_parameters(self):
        """Resets the scaling parameter to 1."""
        nn.init.ones_(self.weight)  # type: ignore


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        head_dim: int,
        n_heads: int,
        n_kv_heads: int,
        use_biased_attention: bool = False,
        use_rope: bool = False,
        rope_theta: float = 10.0,
        rope_mixed: bool = True,
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

        if use_rope:
            self.rope = RoPE(
                head_dim=head_dim,
                num_heads=n_heads,
                rope_theta=rope_theta,
                rope_mixed=rope_mixed,
            )
        else:
            self.rope = None

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | BlockMask | None = None,
        spatial_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Computes the standard multi-head scaled dot-product attention.

        NOTE: Our implementation assumes a packed batch with shape (S, D). We
        distinguish between graphs using the mask.

        Args:
            x (torch.Tensor): Input tensor of shape `(S, D)`.
            mask (torch.Tensor | BlockMask | None, optional): Mask tensor to
                apply to the attention.
                If mask is a FlexAttention BlockMask, then we will use
                FlexAttention, which has the benefit of being sparse.

                If mask is a torch.Tensor or None, then we will use the
                standard scaled dot-product attention.
            spatial_pos (torch.Tensor | None): Spatial position tensor of shape
                `(S, 2)` for use with RoPE. If RoPE is not used, then this can
                be None.

        Returns:
            torch.Tensor: Output tensor of shape `(S, D)`.
        """
        # S D
        seq_len, dim = x.shape
        xq = self.wq(x.view_as(x))
        xk = self.wk(x.view_as(x))
        xv = self.wv(x.view_as(x))

        output_shape = xq.shape
        # S D -> S H D
        xq = xq.view(seq_len, self.n_heads, self.head_dim)
        xk = xk.view(seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(seq_len, self.n_kv_heads, self.head_dim)

        # ROPE HERE
        if self.rope is not None:
            assert spatial_pos is not None
            xq, xk = self.rope(xq, xk, spatial_pos)

        xk = repeat_kv(xk, self.heads_per_group, dim=1)
        xv = repeat_kv(xv, self.heads_per_group, dim=1)

        # sdpa requires H S D format
        # S H D -> H S D
        xq, xk, xv = map(lambda e: e.transpose(0, 1), (xq, xk, xv))
        # Add a singleton dimension for the batch
        # H S D -> 1 H S D
        xq, xk, xv = xq.unsqueeze(0), xk.unsqueeze(0), xv.unsqueeze(0)

        if isinstance(mask, torch.Tensor):
            output = F.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                is_causal=False,
                attn_mask=mask,
            )
        elif isinstance(mask, BlockMask):

            if xq.is_cuda:
                divide = 2
                kernel_options = {
                    "BLOCK_M": int(64 / divide),
                    "BLOCK_N": int(64 / divide),
                    "BLOCK_M1": int(32 / divide),
                    "BLOCK_N1": int(64 / divide),
                    "BLOCK_M2": int(64 / divide),
                    "BLOCK_N2": int(32 / divide),
                }
                # kernel_options = None
            else:
                kernel_options = None

            output = flex_attention_comp(
                xq, xk, xv, block_mask=mask, kernel_options=kernel_options
            )
            assert isinstance(output, torch.Tensor)
        else:
            raise ValueError("Mask must be a BlockMask or a torch.Tensor")

        # 1 H S D -> H S D
        xq, xk, xv = xq.squeeze(0), xk.squeeze(0), xv.squeeze(0)

        # H S D -> S H D
        output = output.transpose(0, 1).contiguous()

        output = self.wo(output.reshape(output_shape))

        return output

    def reset_parameters(self, init_std=None, factor=1.0):
        """Reset parameters of the Attention projections to Normal."""
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


class DifferentialAttention(nn.Module):

    def __init__(
        self,
        dim: int,
        head_dim: int,
        n_heads: int,
        n_kv_heads: int,
        depth: int = 0,  # added
        use_rope: bool = False,
        rope_theta: float = 10.0,
        rope_mixed: bool = True,
    ):
        super().__init__()

        # Note that we lose half of the head_dim here, so the effective head_dim
        # is actually head_dim / 2.

        self.num_heads = n_heads
        self.num_kv_heads = n_kv_heads

        self.head_dim = head_dim // 2
        self.dim = dim

        self.wq = nn.Linear(dim, head_dim * n_heads, bias=False)
        self.wk = nn.Linear(dim, head_dim * n_kv_heads, bias=False)
        self.wv = nn.Linear(dim, head_dim * n_kv_heads, bias=False)
        self.wo = nn.Linear(head_dim * n_heads, dim, bias=False)

        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * depth)
        self.lambda_q1 = nn.Parameter(
            torch.zeros(self.head_dim).normal_(mean=0, std=0.1)
        )
        self.lambda_k1 = nn.Parameter(
            torch.zeros(self.head_dim).normal_(mean=0, std=0.1)
        )
        self.lambda_q2 = nn.Parameter(
            torch.zeros(self.head_dim).normal_(mean=0, std=0.1)
        )
        self.lambda_k2 = nn.Parameter(
            torch.zeros(self.head_dim).normal_(mean=0, std=0.1)
        )

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5)
        if use_rope:
            self.rope = RoPE(
                head_dim=head_dim,
                num_heads=n_heads,
                rope_theta=rope_theta,
                rope_mixed=rope_mixed,
            )
        else:
            self.rope = None

    def _attn_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: BlockMask | torch.Tensor,
    ) -> torch.Tensor:
        """Utility to compute attention."""

        if isinstance(mask, torch.Tensor):
            output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=False,
                attn_mask=mask,
            )
        elif isinstance(mask, BlockMask):

            if q.is_cuda:
                divide = 2
                kernel_options = {
                    "BLOCK_M": int(64 / divide),
                    "BLOCK_N": int(64 / divide),
                    "BLOCK_M1": int(32 / divide),
                    "BLOCK_N1": int(64 / divide),
                    "BLOCK_M2": int(64 / divide),
                    "BLOCK_N2": int(32 / divide),
                }
                # kernel_options = None
            else:
                kernel_options = None

            output = flex_attention_comp(
                q, k, v, block_mask=mask, kernel_options=kernel_options
            )
            assert isinstance(output, torch.Tensor)
        else:
            raise ValueError("Mask must be a BlockMask or a torch.Tensor")

        return output

    def forward(
        self,
        x: torch.Tensor,
        mask: BlockMask | torch.Tensor | None = None,
        spatial_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Computes the differential multi-head scaled dot-product attention.

        NOTE: Our implementation assumes a packed batch with shape (S, D). We
        distinguish between graphs using the mask.

        Args:
            x (torch.Tensor): Input tensor of shape `(S, D)`.
            mask (BlockMask): BlockMask to apply to the attention, which allows
                for sparse attention.
            spatial_pos (torch.Tensor | None): Spatial position tensor of shape
                `(S, 2)` for use with RoPE. If RoPE is not used, then this can
                be None.

        Returns:
            torch.Tensor: Output tensor of shape `(S, D)`.
        """
        # TODO(liamhebert): Maybe add support for sdpa.
        assert mask is not None

        seq_len, dim = x.shape
        xq = self.wq(x.view_as(x))
        xk = self.wk(x.view_as(x))
        xv = self.wv(x.view_as(x))

        xq = xq.view(seq_len, 2 * self.num_heads, self.head_dim)
        xk = xk.view(seq_len, 2 * self.num_kv_heads, self.head_dim)
        xv = xv.view(seq_len, self.num_kv_heads, 2 * self.head_dim)

        if self.rope is not None:
            assert spatial_pos is not None
            xq, xk = self.rope(xq, xk, spatial_pos)

        xq = xq.reshape(1, seq_len, self.num_heads, 2, self.head_dim)
        xk = xk.reshape(1, seq_len, self.num_kv_heads, 2, self.head_dim)
        xv = xv.reshape(1, seq_len, self.num_kv_heads, 2 * self.head_dim)
        # q/k: 1 S H 2 D -> 1 H S 2 D
        # v: 1 S H D -> 1 H S D
        xq, xk, xv = map(lambda e: e.transpose(1, 2), (xq, xk, xv))

        # q/k: 1 H S 2 D -> 1 H S D
        q1, q2 = xq[:, :, :, 0], xq[:, :, :, 1]
        k1, k2 = xk[:, :, :, 0], xk[:, :, :, 1]

        attn1 = self._attn_forward(q1, k1, xv, mask)
        attn2 = self._attn_forward(q2, k2, xv, mask)

        lambda_1 = torch.exp(
            torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1)
        ).type_as(xq)
        lambda_2 = torch.exp(
            torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1)
        ).type_as(xq)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn = attn1 - lambda_full * attn2

        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = attn.reshape(seq_len, self.num_heads * 2 * self.head_dim)

        attn = self.wo(attn)
        return attn

    def reset_parameters(self, init_std=None, factor=1.0):
        """Reset parameters of the Attention projections and lambdas to Normal."""
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

        nn.init.normal_(self.lambda_q1, 0, std=0.1)
        nn.init.normal_(self.lambda_q2, 0, std=0.1)
        nn.init.normal_(self.lambda_k1, 0, std=0.1)
        nn.init.normal_(self.lambda_k2, 0, std=0.1)

        self.subln.reset_parameters()
