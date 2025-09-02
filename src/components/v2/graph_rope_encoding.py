import torch
import torch.nn as nn


def init_2d_freqs(
    dim: int, num_heads: int, theta: float = 10.0, rotate: bool = True
):
    """Initializes 2D frequency tensors for RoPE.

    This function generates frequency tensors for 2D Rotary Position Embeddings
    (RoPE), which are used to encode positional information in the attention
    mechanism. It creates separate frequencies for the x and y dimensions,
    allowing for the representation of 2D spatial relationships.

    The `rotate` parameter enables the generation of rotated RoPE, a technique
    that can enhance model performance by breaking the symmetry of standard
    RoPE and creating more diverse positional encodings across different
    attention heads.

    Args:
        dim (int): The dimension of the head.
        num_heads (int): The number of attention heads.
        theta (float, optional): A parameter controlling the frequency range.
            Defaults to 10.0.
        rotate (bool, optional): If True, applies a random rotation to the
            frequencies for each head. Defaults to True.

    Returns:
        torch.Tensor: A tensor of shape `(2, num_heads, dim // 2)` containing
            the frequencies for the x and y dimensions for each head.
    """
    freqs_x: list[torch.Tensor] = []
    freqs_y: list[torch.Tensor] = []
    mag = 1 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    for i in range(num_heads):
        angles = torch.rand(1) * 2 * torch.pi if rotate else torch.zeros(1)
        fx = torch.cat(
            [mag * torch.cos(angles), mag * torch.cos(torch.pi / 2 + angles)],
            dim=-1,
        )
        fy = torch.cat(
            [mag * torch.sin(angles), mag * torch.sin(torch.pi / 2 + angles)],
            dim=-1,
        )
        freqs_x.append(fx)
        freqs_y.append(fy)
    freq_x = torch.stack(freqs_x, dim=0)
    freq_y = torch.stack(freqs_y, dim=0)
    freqs = torch.stack([freq_x, freq_y], dim=0)
    return freqs


def compute_mixed_cis(
    freqs: torch.Tensor, t_x: torch.Tensor, t_y: torch.Tensor, num_heads: int
):
    """Computes complex numbers (cisoids) for the "mixed" RoPE variant.

    This function takes pre-computed frequencies and the x and y coordinates of
    tokens to generate the complex numbers (cisoids) required for applying RoPE.
    In the "mixed" variant, the frequencies for the x and y dimensions are
    combined before being converted to their polar form. This mixing allows the
    model to learn more complex spatial relationships.

    The resulting cisoids are then used in `apply_rotary_emb` to rotate the
    query and key vectors in the attention mechanism, thereby injecting
    positional information.

    Args:
        freqs (torch.Tensor): A tensor of pre-computed frequencies of shape
            `(2, num_heads, dim // 2)`.
        t_x (torch.Tensor): A tensor of x-coordinates for each token in the
            sequence, of shape `(N,)`.
        t_y (torch.Tensor): A tensor of y-coordinates for each token in the
            sequence, of shape `(N,)`.
        num_heads (int): The number of attention heads.

    Returns:
        torch.Tensor: A tensor of complex numbers (cisoids) of shape
            `(N, num_heads, dim // 2)`, ready to be applied to query and key
            tensors.
    """
    N = t_x.shape[0]
    # No float 16 for this range
    with torch.autocast(freqs.device.type, enabled=False):
        freqs_x = (t_x.unsqueeze(-1) @ freqs[0].unsqueeze(-2)).view(
            N, num_heads, -1
        )
        freqs_y = (t_y.unsqueeze(-1) @ freqs[1].unsqueeze(-2)).view(
            N, num_heads, -1
        )
        freqs_cis = torch.polar(torch.ones_like(freqs_x), freqs_x + freqs_y)

    return freqs_cis


def compute_axial_cis(
    dim: int, t_x: torch.Tensor, t_y: torch.Tensor, theta: float = 100.0
):
    """Computes complex numbers (cisoids) for the "axial" RoPE variant.

    This function generates the complex numbers (cisoids) for the "axial" RoPE
    method, where positional information for the x and y dimensions is kept
    separate (axial). This is in contrast to the "mixed" variant, where
    frequencies are combined.

    The axial approach treats the x and y positions independently, which can be
    beneficial for tasks where the two dimensions represent distinct hierarchical
    structures (e.g., discussion threads and comment splits). The resulting
    cisoids are concatenated, with half corresponding to the x-dimension and
    half to the y-dimension.

    Args:
        dim (int): The dimension of the head.
        t_x (torch.Tensor): A tensor of x-coordinates, shape `(N,)`.
        t_y (torch.Tensor): A tensor of y-coordinates, shape `(N,)`.
        theta (float, optional): A parameter controlling the frequency range.
            Defaults to 100.0.

    Returns:
        torch.Tensor: A tensor of complex numbers (cisoids) of shape
            `(N, dim // 2)`, where the first half of the last dimension encodes
            the x-position and the second half encodes the y-position.
    """
    freqs_x = 1.0 / (
        theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim)
    )
    freqs_y = 1.0 / (
        theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim)
    )

    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
    """Applies rotary embeddings to the input tensor.

    This function applies the pre-computed rotary embeddings (cisoids) to an
    input tensor `x` (typically a query or key tensor from a self-attention
    layer). The core operation involves a complex multiplication between the
    input tensor and the cisoids, which effectively "rotates" the input vectors
    in a high-dimensional space to encode their positional information.

    The input tensor `x` is reshaped to be viewed as a complex tensor,
    multiplied with `freqs_cis`, and then transformed back to its original
    real-valued representation.

    Args:
        x (torch.Tensor): The input tensor to which RoPE will be applied, with
            shape `(..., D)`.
        freqs_cis (torch.Tensor): The complex number (cisoid) tensor, with a
            shape compatible with `x` for broadcasting.

    Returns:
        torch.Tensor: The tensor with rotary embeddings applied, having the
            same shape as the input `x`.
    """
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))

    # print((x_ * freqs_cis).shape)
    if len(x_.shape) == 4:
        x_new = torch.einsum("shjk,shk->shjk", x_, freqs_cis)
    else:
        x_new = torch.einsum("shk,shk->shk", x_, freqs_cis)
        torch.testing.assert_close(x_new, x_ * freqs_cis)
    x_out = torch.view_as_real(x_new)

    x_out = x_out.view_as(x)

    x_out = x_out.type_as(x)
    return x_out


class RoPE(nn.Module):
    """A module for applying 2D Rotary Position Embeddings (RoPE).

    This class encapsulates the logic for applying 2D RoPE to query and key
    tensors within a transformer's attention mechanism. It is designed to handle
    2D spatial information, which is particularly useful for graph-structured
    data where nodes have positions in a 2D plane (e.g., hierarchy and split
    in a discussion tree).

    The module supports two modes of operation:
    1.  `rope_mixed=True`: The "mixed" variant, where the frequencies for x and y
        dimensions are combined. This allows the model to learn complex spatial
        relationships. The frequencies are learnable parameters.
    2.  `rope_mixed=False`: The "axial" variant, where x and y dimensions are
        treated independently. This is useful when the dimensions represent
        distinct, orthogonal concepts.

    Attributes:
        rope_mixed (bool): If True, uses the "mixed" RoPE variant.
        rope_theta (float): A parameter controlling the frequency range.
        num_heads (int): The number of attention heads.
        freqs (nn.Parameter | None): Learnable frequency parameters for the
            "mixed" variant.
    """

    def __init__(
        self,
        head_dim: int,
        num_heads: int,
        rope_theta: float = 10.0,
        rope_mixed: bool = True,
    ):
        """Initializes the RoPE module.

        Args:
            head_dim (int): The dimension of each attention head.
            num_heads (int): The number of attention heads.
            rope_theta (float, optional): A parameter controlling the frequency
                range. Defaults to 10.0.
            rope_mixed (bool, optional): If True, uses the "mixed" RoPE variant
                with learnable, rotated frequencies. Defaults to True.
        """
        super().__init__()

        self.rope_mixed = rope_mixed
        self.rope_theta = rope_theta
        self.num_heads = num_heads

        # x = left-right, y = up-down
        # y = Discussion hierarchy
        # x = Split hierarchy

        if self.rope_mixed:
            freqs = init_2d_freqs(
                dim=head_dim,
                num_heads=num_heads,
                theta=rope_theta,
                rotate=True,
            ).view(2, -1)
            self.freqs = nn.Parameter(freqs, requires_grad=True)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, spatial: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Applies RoPE to the query and key tensors.

        This method takes the query (`q`) and key (`k`) tensors from an
        attention layer, along with a `spatial` tensor containing the 2D
        coordinates of each token. It computes the appropriate cisoids based on
        the selected RoPE variant ("mixed" or "axial") and applies them to `q`
        and `k`.

        Args:
            q (torch.Tensor): The query tensor, of shape `(S, H, E)`, where S is
                the sequence length, H is the number of heads, and E is the head
                dimension.
            k (torch.Tensor): The key tensor, with the same shape as `q`.
            spatial (torch.Tensor): A tensor containing the 2D spatial
                coordinates for each token, of shape `(S, 2)`. The first column
                represents the x-coordinate (split hierarchy) and the second
                column represents the y-coordinate (discussion hierarchy).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the modified
                query and key tensors with rotary embeddings applied. Both
                tensors have the same shape as their inputs.
        """
        # Sequence, num_heads, head_dim
        # S, H, E = q.shape[:-2]
        # # Sequence, (Hierarchy, Split)
        # assert spatial.shape == (S, 2)

        E = q.shape[-1]

        t_x = spatial[:, 0].float()
        t_y = spatial[:, 1].float()

        if self.rope_mixed:
            # Precomputed freqs should already have the dim for the heads
            # TODO(liamhebert): Add an assert to check the shape of self.freqs to
            # match the input dim we have
            # assert self.freqs.shape == (
            #     2,
            #     (H * E) / 2,
            # ), f"{self.freqs.shape} != {2, int((H * E) / 2)}"

            freqs_cis = compute_mixed_cis(
                freqs=self.freqs, t_x=t_x, t_y=t_y, num_heads=self.num_heads
            )

        else:
            freqs_cis = compute_axial_cis(
                dim=E,
                t_x=t_x,
                t_y=t_y,
                theta=self.rope_theta,
            )
            # We broadcast the freqs_cis to each head
            freqs_cis = freqs_cis.unsqueeze(1)

        # Now index freq_cis to get the correct values for the current spatial
        # position

        q_rope = apply_rotary_emb(q, freqs_cis)
        k_rope = apply_rotary_emb(k, freqs_cis)

        assert q_rope.shape == q.shape, f"{q_rope.shape} != {q.shape}"
        assert k_rope.shape == k.shape

        return q_rope, k_rope
