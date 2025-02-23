import torch
import torch.nn as nn


def init_2d_freqs(
    dim: int, num_heads: int, theta: float = 10.0, rotate: bool = True
):
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
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))

    x_out = torch.view_as_real(x_ * freqs_cis).flatten(-2)

    x_out = x_out.type_as(x).to(x.device)
    return x_out


class RoPE(nn.Module):
    def __init__(
        self,
        head_dim: int,
        num_heads: int,
        rope_theta: float = 10.0,
        rope_mixed: bool = True,
    ):
        super().__init__()

        self.rope_mixed = rope_mixed
        self.rope_theta = rope_theta

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
        # Sequence, num_heads, head_dim
        S, H, E = q.shape
        # Sequence, (Hierarchy, Split)
        assert spatial.shape == (S, 2)

        t_x = spatial[:, 0].float()
        t_y = spatial[:, 1].float()

        if self.rope_mixed:
            # Precomputed freqs should already have the dim for the heads
            # TODO(liamhebert): Add an assert to check the shape of self.freqs to
            # match the input dim we have
            assert self.freqs.shape == (
                2,
                (H * E) / 2,
            ), f"{self.freqs.shape} != {2, int((H * E) / 2)}"

            freqs_cis = compute_mixed_cis(
                freqs=self.freqs, t_x=t_x, t_y=t_y, num_heads=H
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
