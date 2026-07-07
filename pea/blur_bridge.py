"""
Blur-reference bridge attribution — "physical path" tren scale-space.

Dong co (xem thao luan): duong thang IG danh gia grad o vung f phang / off-manifold.
BlurIG di theo heat-semigroup (blur baseline -> x) nen artefact-free, NHUNG path co
dinh, khong biet f: ~90% buoc nam o |df|~0, don het tin hieu vao 2-3 buoc cuoi
(xem Fig.1 cua LIG). O day ta lam 2 thu:

  (1) blur_bridge:      giu 2 dau la (blur_baseline -> x) NHUNG them 1 truong drift
                        huong ve chieu tang f (Follmer-lite). Reference process =
                        de-blur; drift keo path ve vung transition cua f MA VAN neo
                        2 dau nen khong roi manifold scale-space.
                        drift_scale = 0  => thu ve dung BlurIG (zero-noise / zero-drift limit).

  (2) blur_bridge_lig:  cung path nhu tren, nhung thay quadrature deu bang measure
                        mu_k ∝ |d_k|  (d_k = grad-predicted change), tuc gioi han
                        convex tau->0 cua LIG-measure. Tap trung ngan sach vao dung
                        cac buoc noi f that su chuyen. Re, thuong an diem insertion.

API giong het cac method khac:
    grad_fn(states: (M,3,H,W)) -> grads (M,3,H,W)
    tra ve attribution (3,H,W).

Ngan sach N = tong so grad eval/anh. blur_bridge dung 2*T grad (T buoc + 1 probe
drift moi buoc neu drift_iters>0); mac dinh dat T sao cho tong ~ N. Xem `_alloc`.

Khong train, khong smoketest. GPU theo device cua x.
"""

from __future__ import annotations
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Heat / blur reference path: chuoi anh tu blur_baseline -> x qua de-blur dan.
# Ta khong can lai kernel scale-space that; noi suy tuyen tinh trong khong gian
# da-chuan-hoa tu blur_baseline toi x DA la 1 de-blur hop le (blur_baseline la
# ban mo cua x), va giu tinh artefact-free o muc thuc dung. Neu may co lich
# scale-space that thi thay `_heat_ref` bang path do.
# ---------------------------------------------------------------------------
@torch.no_grad()
def _heat_ref(x, x0, T):
    """(T+1,3,H,W): reference de-blur path x0 -> x, noi suy deu (endpoint-preserving)."""
    device = x.device
    s = torch.linspace(0.0, 1.0, T + 1, device=device).view(-1, 1, 1, 1)
    return x0[None] + s * (x - x0)[None]


@torch.no_grad()
def _bridge_anchor(t):
    """He so neo bridge: 0 o 2 dau, cuc dai o giua. Dung de drift khong pha endpoint."""
    return 4.0 * t * (1.0 - t)          # t: (K,) -> (K,)


# ---------------------------------------------------------------------------
# blur_bridge: heat reference + Follmer-lite drift huong tang f.
#   Vong lap: tren reference de-blur path, moi buoc uoc luong grad g_k, roi day
#   path ve chieu g_k (chuan hoa) mot luong ti le drift_scale * anchor(t) * ||dx||.
#   Neo 2 dau bang anchor(t) (=0 tai t=0,1) => luon giu blur_baseline -> x.
#   drift_scale=0 => path == reference == BlurIG.
# ---------------------------------------------------------------------------
def blur_bridge(x, blur_baseline, grad_fn, N,
                drift_scale=0.15, drift_iters=1, ito=True):
    """
    x:            (3,H,W) da chuan hoa.
    blur_baseline:(3,H,W) ban mo cua x (reference start).
    N:            ngan sach grad eval. Chia cho (1 + drift_iters) de gom ca probe drift.
    drift_scale:  cuong do keo path ve chieu tang f (0 => BlurIG). Don vi: ti le ||x-x0||.
    drift_iters:  so vong tinh-chinh path bang grad probe (1 la du; 0 => reference thuan).
    ito:          True -> tich phan Ito (left-point) g_k * dgamma_k.

    Tra ve attribution (3,H,W).
    """
    device = x.device
    x0 = blur_baseline
    T = _alloc(N, drift_iters)                       # so buoc path
    dx = (x - x0)
    dxn = dx.norm() + 1e-12

    # reference de-blur path
    path = _heat_ref(x, x0, T)                        # (T+1,3,H,W)
    t = torch.linspace(0.0, 1.0, T + 1, device=device)
    anch = _bridge_anchor(t).view(-1, 1, 1, 1)        # (T+1,1,1,1)

    # tinh-chinh path: keo ve chieu grad, van neo 2 dau
    for _ in range(max(0, drift_iters)):
        g_nodes = grad_fn(path)                        # (T+1,3,H,W), grad tai moi node
        with torch.no_grad():
            # huong drift = grad da chuan hoa theo tung node (chieu tang f)
            gnorm = g_nodes.flatten(1).norm(dim=1).view(-1, 1, 1, 1) + 1e-12
            dir_ = g_nodes / gnorm                     # (T+1,3,H,W)
            # buoc drift ~ drift_scale * anchor(t) * ||dx|| * dir  (0 o 2 dau)
            path = path + drift_scale * anch * dxn * dir_
            # ep lai dung 2 dau cho chac (chong troi so)
            path[0] = x0
            path[-1] = x

    # tich phan doc path da drift
    g = grad_fn(path[:-1])                             # left-point (T,3,H,W)
    with torch.no_grad():
        dgamma = path[1:] - path[:-1]                  # (T,3,H,W)
        if ito:
            attr = (g * dgamma).sum(dim=0)             # Ito, left-point
        else:
            # trapezoid (Stratonovich-ish): trung binh grad 2 dau
            g2 = grad_fn(path[1:])
            attr = (0.5 * (g + g2) * dgamma).sum(dim=0)
    return attr


# ---------------------------------------------------------------------------
# blur_bridge_lig: cung path blur-bridge, nhung do bang LIG-measure mu_k ∝ |d_k|.
#   d_k = g_k . dgamma_k  (grad-predicted change, vo huong moi buoc).
#   mu_k = |d_k| / sum_j |d_j|   (gioi han convex tau->0 cua LIG-measure update).
#   attr = sum_k mu_k * g_k * dgamma_k, roi rescale completeness ve f(x)-f(x0).
#   => don ngan sach vao dung cac buoc f that su chuyen; bo qua ~90% buoc phang.
# ---------------------------------------------------------------------------
def blur_bridge_lig(x, blur_baseline, grad_fn, N,
                    drift_scale=0.15, drift_iters=1,
                    fvals_fn=None, target=None, model=None, score="logit"):
    """
    Giong blur_bridge nhung dung LIG-measure thay cho quadrature deu.

    fvals_fn (tuy chon): ham (states (M,3,H,W))->f (M,) de tinh completeness rescale
        chinh xac. Neu None va co (model,target) thi tu tao. Neu ca hai None thi
        BO rescale (attr van dung ve huong/xep hang, chi lech ti le vo huong).

    Tra ve attribution (3,H,W).
    """
    device = x.device
    x0 = blur_baseline
    T = _alloc(N, drift_iters)
    dx = (x - x0)
    dxn = dx.norm() + 1e-12

    path = _heat_ref(x, x0, T)
    t = torch.linspace(0.0, 1.0, T + 1, device=device)
    anch = _bridge_anchor(t).view(-1, 1, 1, 1)

    for _ in range(max(0, drift_iters)):
        g_nodes = grad_fn(path)
        with torch.no_grad():
            gnorm = g_nodes.flatten(1).norm(dim=1).view(-1, 1, 1, 1) + 1e-12
            path = path + drift_scale * anch * dxn * (g_nodes / gnorm)
            path[0] = x0
            path[-1] = x

    g = grad_fn(path[:-1])                              # (T,3,H,W)
    with torch.no_grad():
        dgamma = path[1:] - path[:-1]                   # (T,3,H,W)
        dk = (g * dgamma).flatten(1).sum(dim=1)         # (T,) grad-predicted change
        mu = dk.abs()
        mu = mu / (mu.sum() + 1e-12)                    # LIG-measure, convex tau->0 limit
        attr = (mu.view(-1, 1, 1, 1) * g * dgamma).sum(dim=0)   # (3,H,W)

    # completeness rescale: sum attr -> f(x)-f(x0)
    fvals_fn = _resolve_fvals(fvals_fn, model, target, score, device)
    if fvals_fn is not None:
        with torch.no_grad():
            fx = fvals_fn(x[None]).reshape(())
            fx0 = fvals_fn(x0[None]).reshape(())
            total = (fx - fx0)
            s = attr.sum()
            attr = attr * (total / (s + 1e-12))
    return attr


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _alloc(N, drift_iters):
    """Chia ngan sach N cho (buoc tich phan + probe drift). Moi drift_iter dung ~ (T+1) grad,
    tich phan dung T grad => tong ~ (drift_iters+1)*T. Dat T = N // (drift_iters+1)."""
    denom = max(1, drift_iters + 1)
    return max(2, N // denom)


def _resolve_fvals(fvals_fn, model, target, score, device):
    """Tao ham f-value tu (model,target) neu can, de rescale completeness."""
    if fvals_fn is not None:
        return fvals_fn
    if model is None or target is None:
        return None

    @torch.no_grad()
    def _f(states):
        logits = model(states.to(device))
        if score == "softmax":
            return F.log_softmax(logits, dim=1)[:, target]
        return logits[:, target]
    return _f