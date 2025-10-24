"""Custom attention layers for graph encoders, such as RoPe and DiffAttn."""

from torch import nn
import flash_attn
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
        output = self._norm(x.float()).type_as(x)
        return (output * self.weight.float()).type_as(x)

    def reset_parameters(self):
        """Resets the scaling parameter to 1."""
        nn.init.ones_(self.weight)  # type: ignore


class Attention(nn.Module):
    """A standard multi-head attention mechanism with optional RoPE.

    This class implements a multi-head attention layer that can be configured
    to use Grouped-Query Attention (GQA) by setting `n_kv_heads` to a value
    less than `n_heads`. It also supports Rotary Position Embeddings (RoPE) for
    injecting positional information into the attention mechanism.

    The forward pass can handle both dense and sparse attention masks. If a
    `BlockMask` from `flex_attention` is provided, it performs sparse attention,
    which is more efficient for graph-structured data.

    Attributes:
        dim (int): The input and output dimension of the layer.
        head_dim (int): The dimension of each attention head.
        n_heads (int): The number of query heads.
        n_kv_heads (int): The number of key/value heads.
        heads_per_group (int): The ratio of query heads to key/value heads.
        wq (nn.Linear): The linear layer for the query projection.
        wk (nn.Linear): The linear layer for the key projection.
        wv (nn.Linear): The linear layer for the value projection.
        wo (nn.Linear): The linear layer for the output projection.
        rope (RoPE | None): The RoPE module, if `use_rope` is True.
    """

    def __init__(
        self,
        dim: int,
        head_dim: int,
        n_heads: int,
        n_kv_heads: int,
        use_rope: bool = False,
        rope_theta: float = 10.0,
        rope_mixed: bool = True,
    ):
        """Initializes the Attention module.

        Args:
            dim (int): The input and output dimension.
            head_dim (int): The dimension of each attention head.
            n_heads (int): The number of query heads.
            n_kv_heads (int): The number of key/value heads. For standard MHA,
                this should be equal to `n_heads`. For GQA, this should be
                smaller than `n_heads`.
            use_rope (bool, optional): If True, enables Rotary Position
                Embeddings. Defaults to False.
            rope_theta (float, optional): The theta parameter for RoPE.
                Defaults to 10.0.
            rope_mixed (bool, optional): If True, uses the "mixed" variant of
                RoPE. Defaults to True.
        """
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
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        mask: torch.Tensor | BlockMask | None = None,
        rope_spatial_pos: torch.Tensor | None = None,
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
            rope_spatial_pos (torch.Tensor | None): Spatial position tensor of
                shape `(S, 2)` for use with RoPE. If RoPE is not used, then this
                can be None.

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
            assert rope_spatial_pos is not None
            xq, xk = self.rope(xq, xk, rope_spatial_pos)

        xk = repeat_kv(xk, self.heads_per_group, dim=1)
        xv = repeat_kv(xv, self.heads_per_group, dim=1)

        # Inputs are S H D
        actual_seq_len = cu_seqlens[-1] if cu_seqlens is not None else seq_len
        output = torch.zeros_like(xq)
        output[:actual_seq_len] = flash_attn.flash_attn_varlen_func(
            xq[:actual_seq_len],
            xk[:actual_seq_len],
            xv[:actual_seq_len],
            cu_seqlens_k=cu_seqlens,
            cu_seqlens_q=cu_seqlens,
            max_seqlen_k=max_seqlen,
            max_seqlen_q=max_seqlen,
            dropout_p=0.0,
            causal=False,
        )

        assert isinstance(output, torch.Tensor)

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
    """Attention mechanism which computes weighted difference of two attentions.

    This layer implements a "differential" attention mechanism, where the final
    attention output is a combination of two separate attention computations,
    `attn1` and `attn2`. The combination is controlled by a learnable parameter
    `lambda_full`, which allows the model to dynamically adjust the contribution
    of each attention component.

    This architecture can be useful for capturing different types of
    relationships in the data. For example, `attn1` might focus on local
    interactions, while `attn2` captures more global dependencies.

    Like the standard `Attention` class, this module supports Grouped-Query
    Attention (GQA), Rotary Position Embeddings (RoPE), and sparse attention
    with `BlockMask`.

    Attributes:
        num_heads (int): The number of query heads.
        num_kv_heads (int): The number of key/value heads.
        head_dim (int): The dimension of each attention head (effectively
            halved).
        dim (int): The input and output dimension.
        wq, wk, wv, wo (nn.Linear): Linear layers for projections.
        lambda_q1, lambda_k1, lambda_q2, lambda_k2 (nn.Parameter): Learnable
            parameters for computing the `lambda_full` weight.
        subln (RMSNorm): A normalization layer applied to the attention output.
        rope (RoPE | None): The RoPE module, if `use_rope` is True.
    """

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
        """Initializes the DifferentialAttention module.

        Args:
            dim (int): The input and output dimension.
            head_dim (int): The dimension of each attention head. Note that this
                is effectively halved in this implementation.
            n_heads (int): The number of query heads.
            n_kv_heads (int): The number of key/value heads.
            depth (int, optional): The depth of the layer in the transformer
                stack, used for initializing `lambda_init`. Defaults to 0.
            use_rope (bool, optional): If True, enables RoPE. Defaults to False.
            rope_theta (float, optional): The theta parameter for RoPE.
                Defaults to 10.0.
            rope_mixed (bool, optional): If True, uses the "mixed" variant of
                RoPE. Defaults to True.
        """
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
            torch.zeros(self.head_dim, dtype=torch.bfloat16).normal_(
                mean=0, std=0.1
            )
        )
        self.lambda_k1 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.bfloat16).normal_(
                mean=0, std=0.1
            )
        )
        self.lambda_q2 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.bfloat16).normal_(
                mean=0, std=0.1
            )
        )
        self.lambda_k2 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.bfloat16).normal_(
                mean=0, std=0.1
            )
        )

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5)
        if use_rope:
            self.rope = RoPE(
                head_dim=self.head_dim,
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
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        """Utility to compute attention."""

        output = flash_attn.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_k=cu_seqlens,
            cu_seqlens_q=cu_seqlens,
            max_seqlen_k=max_seqlen,
            max_seqlen_q=max_seqlen,
            dropout_p=0.0,
            causal=False,
        )
        assert isinstance(output, torch.Tensor)
        return output

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        mask: BlockMask | torch.Tensor | None = None,
        rope_spatial_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Computes the differential multi-head scaled dot-product attention.

        NOTE: Our implementation assumes a packed batch with shape (S, D). We
        distinguish between graphs using the mask.

        Args:
            x (torch.Tensor): Input tensor of shape `(S, D)`.
            mask (BlockMask): BlockMask to apply to the attention, which allows
                for sparse attention.
            rope_spatial_pos (torch.Tensor | None): Spatial position tensor of
                shape `(S, 2)` for use with RoPE. If RoPE is not used, then this
                can be None.

        Returns:
            torch.Tensor: Output tensor of shape `(S, D)`.
        """
        # TODO(liamhebert): Maybe add support for sdpa.
        assert mask is not None

        seq_len, dim = x.shape
        xq = self.wq(x.view_as(x))
        xk = self.wk(x.view_as(x))
        xv = self.wv(x.view_as(x))

        xq = xq.view(seq_len, self.num_heads, 2, self.head_dim)
        xk = xk.view(seq_len, self.num_kv_heads, 2, self.head_dim)
        xv = xv.view(seq_len, self.num_kv_heads, 2 * self.head_dim)

        if self.rope is not None:
            assert rope_spatial_pos is not None
            xq, xk = self.rope(xq, xk, rope_spatial_pos)

        # q/k: 1 H S 2 D -> 1 H S D
        q1, q2 = xq[:, :, 0], xq[:, :, 1]
        k1, k2 = xk[:, :, 0], xk[:, :, 1]

        attn1 = self._attn_forward(q1, k1, xv, cu_seqlens, max_seqlen)
        attn2 = self._attn_forward(q2, k2, xv, cu_seqlens, max_seqlen)

        lambda_1 = torch.exp(
            torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()
        ).type_as(xq)
        lambda_2 = torch.exp(
            torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()
        ).type_as(xq)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn = attn1 - lambda_full * attn2

        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = attn.reshape(seq_len, self.num_heads * 2 * self.head_dim)

        attn = self.wo(attn)
        return attn

    def reset_parameters(self, init_std=None, factor=1.0):
        """Reset parameters of projections and lambdas to Normal."""
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
