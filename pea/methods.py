"""
Cac phuong phap path-attribution de so sanh, cung ngan sach gradient N (so lan eval grad/anh).

Tat ca nhan `grad_fn(states)-> grads` (batch (M,3,H,W) -> (M,3,H,W)) va tra ve attribution (3,H,W).
grad_fn tu quan autograd (xem resnet50_gradfn). Cac phep con lai deu duoi torch.no_grad().

Ngan sach:
  - IG (1 baseline):      T = N buoc tren 1 duong thang.
  - EG (pool B baseline): moi baseline T = N//B buoc  -> tong ~ N.
  - SBA (Brownian bridge):B baseline x P path, moi trajectory T buoc -> B*P*T = N.
  - SBA-D (Ito, Def 4/5): B baseline, K buoc doc barycentric path -> B*K = N.

Instance setting (muc1 = delta_x): OT-coupling toi 1 diem la trivial, nen SBA-D o day
la barycentric path tu moi baseline toi x (co the cong do trung binh nhieu baseline),
tich phan Ito nhu Eq. (5). Neu implementation SBA-D that cua may khac, sua ham sba_d.
"""

from __future__ import annotations
import torch


# ---------------------------------------------------------------------------
# IG cho 1 baseline: phi = (x - x0) * mean_t grad(x0 + t (x-x0))
# ---------------------------------------------------------------------------
def ig_single(x, x0, grad_fn, T):
    device = x.device
    alphas = ((torch.arange(T, device=device) + 0.5) / T).view(-1, 1, 1, 1)  # midpoint
    states = x0[None] + alphas * (x - x0)[None]        # (T,3,H,W)
    g = grad_fn(states)                                # (T,3,H,W)
    return (g.mean(dim=0) * (x - x0))                  # (3,H,W)


# ---------------------------------------------------------------------------
# EG: trung binh IG tren pool baseline, chia ngan sach deu
# ---------------------------------------------------------------------------
def eg(x, baselines, grad_fn, N):
    B = baselines.shape[0]
    T = max(1, N // B)
    acc = torch.zeros_like(x)
    for b in range(B):
        acc += ig_single(x, baselines[b], grad_fn, T)
    return acc / B


# ---------------------------------------------------------------------------
# SBA vanilla: Brownian bridge quanh duong thang, tich phan Ito
#   gamma_t = x0 + t(x-x0) + sigma * BB_t ,  BB = Brownian bridge (0 o 2 dau)
#   phi_i  = E[ sum_t grad_i(gamma_t) * (gamma_{t,i} - gamma_{t-1,i}) ]   (Ito, left point)
# ---------------------------------------------------------------------------
@torch.no_grad()
def _brownian_bridge(T, shape, sigma, device, gen):
    # tao Brownian bridge tren luoi [0,1], T+1 diem (gom 2 dau = 0)
    dt = 1.0 / T
    dW = torch.randn(T, *shape, generator=gen, device=device) * (dt ** 0.5)
    W = torch.cat([torch.zeros(1, *shape, device=device), dW.cumsum(dim=0)], dim=0)  # (T+1,...)
    t = torch.linspace(0, 1, T + 1, device=device).view(-1, *([1] * len(shape)))
    BB = W - t * W[-1:]            # bridge: 0 o t=0 va t=1
    return sigma * BB              # (T+1, *shape)


def sba(x, baselines, grad_fn, N, sigma=0.3, P=1, gen=None):
    B = baselines.shape[0]
    T = max(2, N // (B * P))
    shape = x.shape
    acc = torch.zeros_like(x)
    count = 0
    for b in range(B):
        x0 = baselines[b]
        line = lambda tt: x0[None] + tt * (x - x0)[None]
        for _ in range(P):
            bb = _brownian_bridge(T, shape, sigma, x.device, gen)   # (T+1,3,H,W)
            t = torch.linspace(0, 1, T + 1, device=x.device).view(-1, 1, 1, 1)
            gamma = x0[None] + t * (x - x0)[None] + bb               # (T+1,3,H,W)
            g = grad_fn(gamma[:-1])                                  # left point, (T,3,H,W)
            dgamma = gamma[1:] - gamma[:-1]                          # (T,3,H,W)
            acc += (g * dgamma).sum(dim=0)                           # Ito
            count += 1
    return acc / count


# ---------------------------------------------------------------------------
# SBA-D (Def 4/5): barycentric path tu moi baseline toi x, tich phan Ito
#   phi_i = (1/B) sum_b sum_k grad_i(Xbar_tk) * (Xbar_{i,tk} - Xbar_{i,tk-1})
# Instance setting: coupling toi delta_x trivial -> barycentric path = noi suy
#   co trong so tu tap baseline toi x. O day dung barycenter don gian:
#   tai buoc k, Xbar_tk = (1-s_k)*x0_b + s_k*x, voi s_k tang dan (co the cong
#   do lay trung binh cac baseline khac de mo phong coupling). Giu Ito nhu paper.
# ---------------------------------------------------------------------------
def sba_d(x, baselines, grad_fn, N, gen=None):
    B = baselines.shape[0]
    K = max(2, N // B)
    device = x.device
    # barycenter cua pool baseline (mo phong coupling trung binh o instance setting)
    xbar0 = baselines.mean(dim=0)          # (3,H,W)
    s = torch.linspace(0, 1, K + 1, device=device).view(-1, 1, 1, 1)  # (K+1,...)
    acc = torch.zeros_like(x)
    for b in range(B):
        x0 = baselines[b]
        # path b: di tu x0 -> qua vung barycenter -> toi x (cong nhe ve barycenter giua duong)
        # Xbar_tk = (1-s)*x0 + s*x  +  bend*(xbar0 - straight)  de tao coupling-curvature
        straight = x0[None] + s * (x - x0)[None]                 # (K+1,3,H,W)
        bend = 4.0 * s * (1 - s) * (xbar0[None] - straight)      # cong ve barycenter, 0 o 2 dau
        path = straight + bend                                   # (K+1,3,H,W)
        g = grad_fn(path[:-1])                                   # left point (Ito), (K,3,H,W)
        dpath = path[1:] - path[:-1]                             # (K,3,H,W)
        acc += (g * dpath).sum(dim=0)                            # Ito, Eq. (5)
    return acc / B
