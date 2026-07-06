"""
Path-Ensemble Attribution (monotone headline) and the matched Tube-EG control.

Both estimators are accumulated from *the same* gradient states g_r = grad f(gamma_r):

    gamma_r     = x0 + (x - x0) * s(t_r)                       (coordinate-wise, via groups)
    gammadot_r  = (x - x0) * sdot(t_r)

    phi_PEA    += g_r * gammadot_r / T
    phi_TubeEG += g_r * (x - x0)   / T

=> identical gradient budget, baselines, and queried states. Exact decomposition
   (continuous level):  PEA = Tube-EG + increment_term.
   (We do NOT label the increment term a "Hessian correction" for the monotone family.)

Design notes:
 - `grad_fn(states)` must return dF/d(input) for a batch of interpolated inputs, where
   F is the scalar target (e.g. logit of the target class). You own the model; you run it.
 - GPU: everything follows the device of `x`. No .cpu()/.item() in the hot loop except
   optional geometry logging at the end.
 - Nothing here trains or runs a benchmark. It just computes attributions for inputs you pass.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import torch

from .schedules import (
    sample_schedule_coeffs,
    s_of_t,
    sdot_of_t,
    expand_groups,
)


@dataclass
class Geometry:
    """Accumulated path-geometry diagnostics (means over sampled paths & timesteps)."""
    rms_deviation: float = 0.0      # E[ int ||gamma_t - m_t||^2 dt ]
    path_energy: float = 0.0        # E[ int ||gammadot_t - (x-x0)||^2 dt ]
    excess_length: float = 0.0      # E[ int ||gammadot_t|| dt ] - ||x-x0||
    n: int = field(default=0)


GradFn = Callable[[torch.Tensor], torch.Tensor]
"""grad_fn(batch_states: (B, *input_shape)) -> (B, *input_shape) gradients of the scalar target."""


@torch.no_grad()  # gradients come from grad_fn, which manages its own autograd internally
def path_ensemble_attribution(
    x: torch.Tensor,                 # (*input_shape) single input
    baselines: torch.Tensor,        # (P, *input_shape) one baseline per path (baseline pool sampled by you)
    group_index: torch.Tensor,      # (D,) long, D = prod(input_shape)
    grad_fn: GradFn,
    n_groups: int,
    L: int = 6,
    rho: float = 0.5,
    T: int = 25,
    generator: torch.Generator | None = None,
    antithetic: bool = False,
    log_geometry: bool = True,
):
    """
    Returns:
        phi_pea     : (*input_shape)
        phi_tubeeg  : (*input_shape)
        geom        : Geometry
        completeness: dict with 'pea_sum', 'target_delta' filled if you also pass f-values
                      (left to caller; see completeness_check in tests)

    P (number of paths) = baselines.shape[0]. If antithetic=True, each sampled coeff set a
    is paired with -a; supply P even and coeffs are generated for P//2 pairs.
    """
    device, dtype = x.device, x.dtype
    input_shape = x.shape
    D = int(torch.tensor(input_shape).prod().item())
    P = baselines.shape[0]
    assert group_index.numel() == D, "group_index must cover every coordinate"
    assert group_index.max().item() < n_groups

    x_flat = x.reshape(-1)  # (D,)
    phi_pea = torch.zeros(D, device=device, dtype=dtype)
    phi_tube = torch.zeros(D, device=device, dtype=dtype)
    geom = Geometry()

    # midpoint grid t_r = (r + 0.5)/T
    r = torch.arange(T, device=device, dtype=dtype)
    t_grid = (r + 0.5) / T  # (T,)
    dt = 1.0 / T

    # build the per-path coefficient list (respecting antithetic pairing)
    if antithetic:
        assert P % 2 == 0, "antithetic requires an even number of paths P"
        base = sample_schedule_coeffs(
            (P // 2) * n_groups, L, rho, generator=generator, device=device, dtype=dtype
        ).reshape(P // 2, n_groups, L)
        coeffs = torch.cat([base, -base], dim=0)  # (P, n_groups, L)
    else:
        coeffs = sample_schedule_coeffs(
            P * n_groups, L, rho, generator=generator, device=device, dtype=dtype
        ).reshape(P, n_groups, L)

    delta_full = None  # ||x - x0|| accumulator handled per path for excess length

    for p in range(P):
        x0 = baselines[p].reshape(-1)  # (D,)
        delta = x_flat - x0            # (D,)
        a = coeffs[p]                  # (n_groups, L)

        # schedule values on the whole grid: (T, n_groups)
        s_grid = s_of_t(a, t_grid)
        sdot_grid = sdot_of_t(a, t_grid)
        # expand to coordinates: (T, D)
        s_full = expand_groups(s_grid, group_index)
        sdot_full = expand_groups(sdot_grid, group_index)

        # interpolated states gamma_r : (T, D)  -> reshape to (T, *input_shape)
        states = x0[None, :] + delta[None, :] * s_full
        states_shaped = states.reshape(T, *input_shape)

        # single batched backward for all T midpoints
        g = grad_fn(states_shaped).reshape(T, D)  # (T, D)

        gammadot = delta[None, :] * sdot_full  # (T, D)

        # accumulate both estimators from the SAME g
        phi_pea += (g * gammadot).sum(dim=0) * dt
        phi_tube += (g * delta[None, :]).sum(dim=0) * dt

        if log_geometry:
            m = x0[None, :] + delta[None, :] * t_grid[:, None]  # straight path (T, D)
            dev = ((states - m) ** 2).sum(dim=1).mean() * 1.0  # approx int over t via mean
            energy = ((gammadot - delta[None, :]) ** 2).sum(dim=1).mean()
            length = gammadot.norm(dim=1).mean()  # approx int ||gammadot|| dt
            straight_len = delta.norm()
            geom.rms_deviation += float(dev)
            geom.path_energy += float(energy)
            geom.excess_length += float(length - straight_len)
            geom.n += 1

    phi_pea /= P
    phi_tube /= P
    if geom.n:
        geom.rms_deviation /= geom.n
        geom.path_energy /= geom.n
        geom.excess_length /= geom.n

    return phi_pea.reshape(input_shape), phi_tube.reshape(input_shape), geom
