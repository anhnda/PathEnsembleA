"""
Endpoint-preserving monotone transition schedules (bounded cosine parameterization).

s_g(t)      = t + sum_l  a_gl / (2*pi*l) * sin(2*pi*l*t)
s_dot_g(t)  = 1 + sum_l  a_gl        * cos(2*pi*l*t)

with   sum_l |a_gl| <= rho < 1   =>   s_dot_g(t) >= 1 - rho > 0   (strictly monotone),
and    s_g(0) = 0,  s_g(1) = 1                                    (no overshoot).

`rho` is the schedule-strength (replaces sigma). a_gl are sampled symmetric around 0,
so E[s_g(t)] = t : on average the ensemble reduces to the straight IG path.

Schedules live on *groups* (image patches / text tokens), not per-coordinate.
"""

from __future__ import annotations
import torch


def sample_schedule_coeffs(
    n_groups: int,
    L: int,
    rho: float,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Sample coefficients a[g, l], g=0..n_groups-1, l=1..L, symmetric around 0,
    then rescale each group so that sum_l |a_gl| == rho (the hard cap, < 1).

    Returns: a of shape (n_groups, L).
    """
    assert 0.0 <= rho < 1.0, "rho must be in [0, 1)"
    if rho == 0.0 or L == 0:
        return torch.zeros(n_groups, L, device=device, dtype=dtype)

    # symmetric raw coeffs
    a = torch.randn(n_groups, L, generator=generator, device=device, dtype=dtype)
    l1 = a.abs().sum(dim=1, keepdim=True)  # (n_groups, 1)
    # groups with all-zero draw (measure zero) -> leave as zeros
    safe = l1 > 0
    a = torch.where(safe, a / l1.clamp_min(1e-12) * rho, torch.zeros_like(a))
    return a


def _l_vec(L: int, device, dtype) -> torch.Tensor:
    return torch.arange(1, L + 1, device=device, dtype=dtype)  # (L,)


def s_of_t(a: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    s_g(t) for each group.
    a: (n_groups, L), t: scalar tensor or (K,) tensor of times.
    Returns: (n_groups,) if t scalar, else (K, n_groups).
    """
    L = a.shape[1]
    device, dtype = a.device, a.dtype
    l = _l_vec(L, device, dtype)  # (L,)
    t = torch.as_tensor(t, device=device, dtype=dtype)
    scalar = t.dim() == 0
    tt = t.reshape(-1)  # (K,)
    # sin(2 pi l t): (K, L)
    ang = 2.0 * torch.pi * tt[:, None] * l[None, :]
    coef = a / (2.0 * torch.pi * l)[None, :]  # (n_groups, L)
    # (K, n_groups) = tt + sum_l coef_gl * sin(ang_kl)
    extra = torch.einsum("kl,gl->kg", torch.sin(ang), coef)
    s = tt[:, None] + extra  # (K, n_groups)
    return s[0] if scalar else s


def sdot_of_t(a: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    s_dot_g(t). Same shape convention as s_of_t.
    """
    L = a.shape[1]
    device, dtype = a.device, a.dtype
    l = _l_vec(L, device, dtype)
    t = torch.as_tensor(t, device=device, dtype=dtype)
    scalar = t.dim() == 0
    tt = t.reshape(-1)
    ang = 2.0 * torch.pi * tt[:, None] * l[None, :]  # (K, L)
    sdot = 1.0 + torch.einsum("kl,gl->kg", torch.cos(ang), a)  # (K, n_groups)
    return sdot[0] if scalar else sdot


def expand_groups(group_values: torch.Tensor, group_index: torch.Tensor) -> torch.Tensor:
    """
    Map per-group schedule values back to full-coordinate tensor.

    group_values: (..., n_groups)
    group_index:  (D,) long, group id of each of the D coordinates (flattened input)
    Returns:      (..., D)
    """
    return group_values.index_select(-1, group_index)


def make_patch_groups(C: int, H: int, W: int, grid: int) -> torch.Tensor:
    """
    Image group map: a `grid x grid` block layout over (H, W), shared across channels.
    Returns group_index of shape (C*H*W,) matching a C,H,W flatten (row-major).
    """
    gy = (torch.arange(H) * grid // H).clamp_max(grid - 1)  # (H,)
    gx = (torch.arange(W) * grid // W).clamp_max(grid - 1)  # (W,)
    block = gy[:, None] * grid + gx[None, :]  # (H, W)
    full = block[None, :, :].expand(C, H, W).reshape(-1)  # (C*H*W,)
    return full.long()


def make_token_groups(n_tokens: int, emb_dim: int) -> torch.Tensor:
    """
    Text group map: one group per token, shared across the token's embedding dims.
    Returns group_index of shape (n_tokens*emb_dim,) matching an (n_tokens, emb_dim) flatten.
    """
    tok = torch.arange(n_tokens)[:, None].expand(n_tokens, emb_dim).reshape(-1)
    return tok.long()
