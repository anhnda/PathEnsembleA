"""
BlurLIG — de-blur (BlurIG) path + LIG-measure mu_k ∝ |d_k|.

Day la method thang sau khi da loai het cac huong path-based:
  - drift Follmer-lite (keo path ve grad):      LAM PHANG |d_k| -> tut diem (church I-D 3.90).
  - self-diffusion (heat-correlated bridge):    == measure-only, path stochastic VO ICH (4.59, ngang).
  - reparam thoi gian (arc-length theo |d|):    FAIL, roi ve uniform (4.22).
    Ly do: attribution = sum_k w_k (g_k ⊙ dgamma_k). Measure doi TRONG SO w_k tren
    vector dong gop (giu nguyen node). Reparam ep w_k=1 nhung doi NODE -> don node vao
    transition lam dgamma_k NGAN lai, bop chet chinh vung muon nhan. Hai phep chi trung
    khi g_k khong xoay doc path — khong dung tren path that. => measure KHONG absorb
    duoc vao path bang reparam; no phai la 1 buoc rieng.

Ket luan: tren path blur co dinh, chi can 1 pha measure (mu_k ∝ |d_k|). Khong drift,
khong stochastic, khong reparam. Gon hon LIG goc (khong can alternating minimisation).

  d_k   = g_k . dgamma_k              (grad-predicted change, vo huong moi buoc)
  mu_k  = |d_k| / sum_j |d_j|         (gioi han convex tau->0 cua LIG-measure)
  A     = sum_k mu_k * g_k ⊙ dgamma_k
  rescale A sao cho sum(A) = f(x) - f(x0)   (completeness)

API dong nhat voi cac method khac:
    grad_fn(states: (M,3,H,W)) -> grads (M,3,H,W)
    tra ve attribution (3,H,W).

Ngan sach N = so grad eval/anh (o day == so buoc T). GPU theo device cua x.
"""

from __future__ import annotations
import torch
import torch.nn.functional as F


@torch.no_grad()
def _deblur_path(x, x0, T):
    """(T+1,3,H,W): de-blur reference x0 -> x, noi suy deu endpoint-preserving."""
    s = torch.linspace(0.0, 1.0, T + 1, device=x.device).view(-1, 1, 1, 1)
    return x0[None] + s * (x - x0)[None]


def _make_fvals(model, target, score, device):
    """Ham f-value cho completeness rescale. None neu thieu model/target."""
    if model is None or target is None:
        return None

    @torch.no_grad()
    def _f(states):
        logits = model(states.to(device))
        if score == "softmax":
            return F.log_softmax(logits, dim=1)[:, target]
        return logits[:, target]
    return _f


def blur_lig(x, blur_baseline, grad_fn, N,
             model=None, target=None, score="logit"):
    """
    de-blur path + LIG-measure mu_k ∝ |d_k|, Ito left-point.

    x, blur_baseline : (3,H,W) da chuan hoa (blur_baseline = ban mo cua x).
    grad_fn          : (M,3,H,W) -> (M,3,H,W) grad cua logit target theo input.
    N                : so buoc = so grad eval.
    model,target     : de rescale completeness (tuy chon). Thieu -> bo rescale
                       (huong/xep hang van dung, chi lech ti le vo huong).

    Tra ve attribution (3,H,W).
    """
    device = x.device
    x0 = blur_baseline
    T = max(2, N)

    path = _deblur_path(x, x0, T)                      # (T+1,3,H,W)
    g = grad_fn(path[:-1])                             # left-point Ito (T,3,H,W)

    with torch.no_grad():
        dgamma = path[1:] - path[:-1]                 # (T,3,H,W)
        gd = g * dgamma                               # (T,3,H,W) vector dong gop moi buoc
        dk = gd.flatten(1).sum(dim=1)                 # (T,) d_k = g_k . dgamma_k
        mu = dk.abs()
        mu = mu / (mu.sum() + 1e-12)                  # LIG-measure
        attr = (mu.view(-1, 1, 1, 1) * gd).sum(dim=0)  # (3,H,W)

        # completeness rescale: sum(A) -> f(x) - f(x0)
        fvals = _make_fvals(model, target, score, device)
        if fvals is not None:
            total = (fvals(x[None]).reshape(()) - fvals(x0[None]).reshape(()))
            attr = attr * (total / (attr.sum() + 1e-12))

    return attr