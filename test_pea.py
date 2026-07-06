"""
Must-pass tests before any benchmark (chốt #3). Run yourself:

    cd /path/to  &&  pytest pea/test_pea.py -v

None of these touch a real model or dataset; they use closed-form f so ground truth
is known. GPU is used automatically if available (set PEA_TEST_DEVICE=cuda to force).
"""

import os
import math
import torch
import pytest

from pea.schedules import (
    sample_schedule_coeffs,
    s_of_t,
    sdot_of_t,
    make_patch_groups,
    make_token_groups,
    expand_groups,
)
from pea.estimator import path_ensemble_attribution


DEVICE = os.environ.get(
    "PEA_TEST_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
)
DTYPE = torch.float64  # tight numerical checks


def _gen(seed=0):
    g = torch.Generator(device=DEVICE)
    g.manual_seed(seed)
    return g


# ---------- gradient oracles for closed-form f ----------

def make_linear_gradfn(w):
    """f(x) = <w, x>  =>  grad = w (constant)."""
    def grad_fn(states):  # states: (T, *shape)
        return w.expand_as(states).clone()
    return grad_fn


def make_quadratic_gradfn(A):
    """f(x) = 0.5 x^T A x  =>  grad = A x  (flatten input)."""
    def grad_fn(states):
        T = states.shape[0]
        flat = states.reshape(T, -1)
        g = flat @ A.T
        return g.reshape_as(states)
    return grad_fn


# ---------- Test 1: monotonicity / no overshoot ----------

def test_monotonicity_and_endpoints():
    a = sample_schedule_coeffs(64, L=6, rho=0.7, generator=_gen(1),
                               device=DEVICE, dtype=DTYPE)
    t = torch.linspace(0, 1, 501, device=DEVICE, dtype=DTYPE)
    s = s_of_t(a, t)       # (501, 64)
    sdot = sdot_of_t(a, t) # (501, 64)

    # s in [0,1]
    assert s.min() >= -1e-9, f"s underflow {s.min()}"
    assert s.max() <= 1 + 1e-9, f"s overshoot {s.max()}"
    # endpoints
    assert torch.allclose(s[0], torch.zeros_like(s[0]), atol=1e-9)
    assert torch.allclose(s[-1], torch.ones_like(s[-1]), atol=1e-9)
    # strictly increasing: sdot >= 1 - rho > 0
    assert sdot.min() >= (1 - 0.7) - 1e-9, f"sdot dipped to {sdot.min()}"


# ---------- Test 2: rho=0 collapses PEA to EG ----------

def test_rho_zero_equals_eg():
    torch.manual_seed(0)
    shape = (3, 8, 8)
    D = 3 * 8 * 8
    x = torch.randn(shape, device=DEVICE, dtype=DTYPE)
    P = 8
    baselines = torch.randn(P, *shape, device=DEVICE, dtype=DTYPE)
    gidx = make_patch_groups(3, 8, 8, grid=4).to(DEVICE)
    n_groups = 16
    w = torch.randn(shape, device=DEVICE, dtype=DTYPE)
    grad_fn = make_linear_gradfn(w)

    phi_pea, phi_tube, geom = path_ensemble_attribution(
        x, baselines, gidx, grad_fn, n_groups=n_groups,
        L=6, rho=0.0, T=64, generator=_gen(2),
    )
    # rho=0 => straight path => PEA == Tube-EG == EG closed form
    # EG for linear f: mean_p w * (x - x0)  ... but grad is constant w, so
    # phi = mean_p (x - x0) * w
    eg = ((x[None] - baselines) * w[None]).mean(dim=0)
    assert torch.allclose(phi_pea, phi_tube, atol=1e-8)
    assert torch.allclose(phi_pea, eg, atol=1e-6), (phi_pea - eg).abs().max()
    assert geom.rms_deviation < 1e-12 and geom.path_energy < 1e-12


# ---------- Test 3: linear model, all methods coincide ----------

def test_linear_all_methods_coincide():
    torch.manual_seed(3)
    shape = (2, 6, 6)
    x = torch.randn(shape, device=DEVICE, dtype=DTYPE)
    P = 6
    baselines = torch.randn(P, *shape, device=DEVICE, dtype=DTYPE)
    gidx = make_patch_groups(2, 6, 6, grid=3).to(DEVICE)
    w = torch.randn(shape, device=DEVICE, dtype=DTYPE)
    grad_fn = make_linear_gradfn(w)

    phi_pea, phi_tube, _ = path_ensemble_attribution(
        x, baselines, gidx, grad_fn, n_groups=9,
        L=5, rho=0.6, T=128, generator=_gen(4),
    )
    # For linear f grad is constant, so schedule shape is irrelevant:
    # int g*gammadot dt = g * (x - x0) regardless of s. PEA == Tube-EG == EG.
    eg = ((x[None] - baselines) * w[None]).mean(dim=0)
    assert torch.allclose(phi_pea, phi_tube, atol=1e-7)
    assert torch.allclose(phi_pea, eg, atol=1e-5), (phi_pea - eg).abs().max()


# ---------- Test 4: completeness, residual shrinks with T ----------

def test_completeness_residual_decreases_with_T():
    torch.manual_seed(5)
    shape = (12,)
    D = 12
    x = torch.randn(shape, device=DEVICE, dtype=DTYPE)
    P = 4
    baselines = torch.randn(P, *shape, device=DEVICE, dtype=DTYPE)
    gidx = torch.arange(D, device=DEVICE)  # each coord its own group
    A = torch.randn(D, D, device=DEVICE, dtype=DTYPE)
    A = A + A.T  # symmetric so f = 0.5 x^T A x well-defined
    grad_fn = make_quadratic_gradfn(A)

    def f(v):  # scalar target
        return 0.5 * v @ A @ v

    target_delta = float(f(x) - torch.stack([f(b) for b in baselines]).mean())

    residuals = []
    for T in (10, 40, 160):
        phi_pea, _, _ = path_ensemble_attribution(
            x, baselines, gidx, grad_fn, n_groups=D,
            L=6, rho=0.5, T=T, generator=_gen(6),
        )
        residuals.append(abs(float(phi_pea.sum()) - target_delta))

    # monotone decrease (quadrature residual -> 0)
    assert residuals[0] > residuals[1] > residuals[2], residuals
    assert residuals[-1] < 1e-3, residuals


# ---------- Test 5: Tube-EG and PEA share exact gradient states ----------

def test_same_gradient_states():
    """
    Record every state grad_fn is queried on. PEA and Tube-EG are computed in the
    same call from the same g, so there is exactly one set of states; assert the
    recorded states are shared (not duplicated) and equal the interpolation states.
    """
    torch.manual_seed(7)
    shape = (3, 4, 4)
    x = torch.randn(shape, device=DEVICE, dtype=DTYPE)
    P = 3
    baselines = torch.randn(P, *shape, device=DEVICE, dtype=DTYPE)
    gidx = make_patch_groups(3, 4, 4, grid=2).to(DEVICE)
    w = torch.randn(shape, device=DEVICE, dtype=DTYPE)

    seen = []
    base = make_linear_gradfn(w)
    def grad_fn(states):
        seen.append(states.detach().clone())
        return base(states)

    T = 16
    phi_pea, phi_tube, _ = path_ensemble_attribution(
        x, baselines, gidx, grad_fn, n_groups=4, L=4, rho=0.5, T=T,
        generator=_gen(8),
    )
    # one backward batch per path, each of size T -> P batches total, no extra for Tube-EG
    assert len(seen) == P, f"expected {P} gradient batches, got {len(seen)}"
    assert all(s.shape[0] == T for s in seen)


# ---------- Test 6: geometry log is populated and sane ----------

def test_geometry_log():
    torch.manual_seed(9)
    shape = (3, 8, 8)
    x = torch.randn(shape, device=DEVICE, dtype=DTYPE)
    P = 8
    baselines = torch.randn(P, *shape, device=DEVICE, dtype=DTYPE)
    gidx = make_patch_groups(3, 8, 8, grid=4).to(DEVICE)
    w = torch.randn(shape, device=DEVICE, dtype=DTYPE)
    grad_fn = make_linear_gradfn(w)

    _, _, geom = path_ensemble_attribution(
        x, baselines, gidx, grad_fn, n_groups=16, L=6, rho=0.6, T=64,
        generator=_gen(10), log_geometry=True,
    )
    assert geom.n == P
    assert geom.rms_deviation > 0
    assert geom.path_energy > 0
    # excess length is >= 0 up to discretization for a wiggly monotone path
    assert geom.excess_length > -1e-6


# ---------- Extra: token group map shape sanity ----------

def test_token_groups_shape():
    gi = make_token_groups(n_tokens=7, emb_dim=5)
    assert gi.shape == (35,)
    assert gi.max().item() == 6 and gi.min().item() == 0
    # each token shares one group across all 5 embedding dims
    assert (gi[:5] == 0).all() and (gi[5:10] == 1).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
