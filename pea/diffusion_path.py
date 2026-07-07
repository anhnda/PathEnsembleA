"""
Diffusion-path attribution — dong co VAT LY: forward VP-SDE / heat-diffusion.

Bat nguon tu thao luan trong repo (xem blur_bridge.py): blur CHINH LA heat diffusion.
Bridge y do sang khung diffusion sinh-mo-hinh:

  forward VP-SDE:   dx = -1/2 beta(t) x dt + sqrt(beta(t)) dW
  marginal (analytic, KHONG can train):
        x_t = sqrt(abar_t) * x0-image  +  sqrt(1-abar_t) * (thanh phan "mat thong tin")

Trong attribution ta KHONG sinh anh moi; ta chi can 1 QUY DAO tu "trang thai mat
thong tin" (reference) toi x. Hai dong tri:

  (1) diffusion_ig  — VP-SDE ANALYTIC path, LIGHTWEIGHT (dong dang, khong score-net).
        Baseline khong con la 1 diem chon tay ma la 1 HO trang thai theo muc nhieu.
        Lay KY VONG attribution tren lich nhieu => khong phu thuoc 1 baseline may rui.
        Day la o "forward analytic" cua bang 2x2: in-manifold theo nghia QUY DAO
        noising chuan, van giu lightweight. Khac phuc: "phai-chon-baseline".

  (2) diffusion_pf  — PROBABILITY-FLOW-ODE-style path: DETERMINISTIC, reversible,
        bam quy dao VP marginal, KHONG stochastic. Day la o payoff cua bang:
        khac phuc dung tu-huyet straight-line-OOD cua IG, GIU completeness sach
        (path xac dinh noi x0->x). Score that thi can 1 mang; o day ta dung
        heat-diffusion (blur) lam SCORE-PROXY — nhat quan voi lap luan blur=heat
        cua repo, khong overclaim co score-net.

Ca hai theo dung convention repo:
    grad_fn(states: (M,3,H,W)) -> grads (M,3,H,W),   Ito left-point,
    completeness rescale ve f(x)-f(x0) qua _resolve_fvals.

Khong train, khong smoketest. Device theo x.
"""

from __future__ import annotations
import torch
import torch.nn.functional as F

# tai dung helper co san de khoi lech convention voi phan con lai cua repo
from .blur_bridge import _resolve_fvals, _heat_kernel1d, _heat_smooth


# ===========================================================================
# VP-SDE marginal coefficients (analytic, khong train).
#   beta(t) tuyen tinh: beta(t) = beta_min + t*(beta_max - beta_min)
#   log abar_t = -integral_0^t beta(s) ds   (VP / DDPM continuous, Song et al.)
# t di tu 0 (data) -> 1 (noise). Ta dung s = 1 - t de path chay noise->data.
# ===========================================================================
@torch.no_grad()
def _vp_abar(t, beta_min=0.1, beta_max=20.0):
    """abar_t = exp(-int_0^t beta). t: (K,) in [0,1] -> (K,)."""
    integral = beta_min * t + 0.5 * (beta_max - beta_min) * t * t
    return torch.exp(-integral)


# ===========================================================================
# (1) diffusion_ig : VP-SDE analytic path, lightweight, expectation tren lich nhieu.
#
# Ta dung blur_baseline nhu "thanh phan mat thong tin" (dung tinh than blur=heat):
#   x_t(image) = sqrt(abar_t) * x  +  (1 - sqrt(abar_t)) * x0
# t: 1 (=> x0, mat thong tin) -> 0 (=> x, data). Coefficient sqrt(abar) tang don dieu
# tu 0 len 1 => day la 1 de-noising path noi x0 -> x nhung CONG theo lich VP thay vi
# tuyen tinh deu. Diem sac cua transition duoc lich VP dat vao vung abar chuyen nhanh.
#
# LIGHTWEIGHT: chi la reweight thoi gian analytic, KHONG score-net, KHONG stochastic
#   (rieng dong nay). Neu P>1 & jitter>0 => lay ky vong tren cac lich nhieu lech nhau
#   (khac lam mem endpoint), van dong-dang.
# Ngan sach: P * T = N grad eval.
# ===========================================================================
def diffusion_ig(x, blur_baseline, grad_fn, N,
                 beta_min=0.1, beta_max=20.0,
                 P=1, jitter=0.0, gen=None):
    """
    VP-SDE analytic de-noising path (blur_baseline -> x), Ito, ky vong tren P lich.

    x, blur_baseline : (3,H,W) da chuan hoa.
    N                : ngan sach grad; P*T = N.
    beta_min/max     : lich beta VP (Song et al. continuous). Dieu khien do cong.
    P                : so lich nhieu lay ky vong (P=1 => 1 path analytic thuan).
    jitter           : lech ngau nhien nho cua luoi thoi gian giua cac lich (>0 => P
                       lich khac nhau; van deterministic-per-lich, on-manifold).
    Tra ve attribution (3,H,W).
    """
    device = x.device
    x0 = blur_baseline
    dx = (x - x0)
    T = max(2, N // max(1, P))

    acc = torch.zeros_like(x)
    for p in range(max(1, P)):
        with torch.no_grad():
            # luoi t tren [t_hi -> t_lo], t=1 ~ x0, t=0 ~ x. Chua endpoint.
            base = torch.linspace(1.0, 0.0, T + 1, device=device)
            if jitter > 0 and gen is not None:
                # lech nho, giu don dieu & endpoint (khong pha noi x0->x)
                noise = torch.randn(T + 1, generator=gen, device=device) * jitter
                noise[0] = 0.0; noise[-1] = 0.0
                base = (base + noise).clamp(0.0, 1.0)
                base, _ = torch.sort(base, descending=True)
            abar = _vp_abar(base, beta_min, beta_max)          # (T+1,)
            coef = abar.sqrt().view(-1, 1, 1, 1)               # sqrt(abar): 0(x0)..1(x)
            # coef(t=1)=sqrt(abar_1)~0 -> gan x0 ; coef(t=0)=1 -> x
            path = x0[None] + coef * dx[None]                  # (T+1,3,H,W)
            path[0] = x0; path[-1] = x
        g = grad_fn(path[:-1])                                  # left-point Ito
        with torch.no_grad():
            acc += (g * (path[1:] - path[:-1])).sum(dim=0)
    return acc / max(1, P)


# ===========================================================================
# (2) diffusion_pf : Probability-Flow-ODE-style DETERMINISTIC path.
#
# PF-ODE cua VP-SDE:  dx = [ f(x,t) - 1/2 g(t)^2 * score ] dt,  score = grad log p_t.
# Score that can 1 mang. O day dung SCORE-PROXY nhat quan voi repo:
#   blur = heat diffusion => huong "ve manifold data" xap xi bang (x_deblur - x_t),
#   tuc keo trang thai hien tai ve phia anh sac hon (de-blur residual). Day la mot
#   xap xi score-like KHONG can train, giu path DETERMINISTIC va on-manifold.
#
# Path xay bang tich phan Euler PF-ODE-style tu x0(noise-mo) -> x(data), moi buoc:
#   x_{k+1} = x_k + drift_k,  drift_k huong theo (i) de-noising VP (keo ve x) +
#             (ii) score-proxy = de-blur residual (keo ve vung sac hon MANIFOLD),
#   ep endpoint x0->x de completeness sach.
# Sau khi co path DETERMINISTIC, tich phan Ito left-point.
#
# Option LIG-measure mu_k ∝ |d_k| (giong blur_bridge_lig) de don ngan sach vao
# dung buoc f chuyen. Completeness rescale ve f(x)-f(x0).
#
# Ngan sach: T buoc path -> T grad (proxy score KHONG ton grad, chi conv). = N.
# ===========================================================================
@torch.no_grad()
def _score_proxy_deblur(x_t, x, ksize, sigma_pix):
    """
    Score-proxy 've manifold': residual giua trang thai va ban de-blur cua NO huong x.
    Tra ve huong (3,H,W) da chuan hoa L2 toan cuc (chi lay CHIEU, do lon do drift dat).
    Y: neu x_t con mo/lech, (x - blur(x_t)) chi ve phia data sac hon.
    """
    k1d = _heat_kernel1d(ksize, sigma_pix, x_t.device, x_t.dtype)
    xt_blur = _heat_smooth(x_t[None], k1d)[0]         # ban mo cua trang thai
    resid = (x - xt_blur)                             # keo ve data sac hon
    return resid


def diffusion_pf(x, blur_baseline, grad_fn, N,
                 beta_min=0.1, beta_max=20.0,
                 score_scale=0.15, ksize=31, sigma_pix=None,
                 use_lig=True, ito=True,
                 model=None, target=None, score="logit"):
    """
    PF-ODE-style DETERMINISTIC diffusion path + Ito (+ optional LIG-measure).

    x, blur_baseline : (3,H,W).
    N                : ngan sach grad (= so buoc tich phan T).
    beta_min/max     : lich VP dinh do-cong cua thanh phan de-noising.
    score_scale      : cuong do score-proxy (de-blur residual). 0 => path VP thuan
                       (gan diffusion_ig 1 lich), >0 => cong ve manifold sac.
    ksize/sigma_pix  : heat kernel cho score-proxy (blur=heat).
    use_lig          : True => do bang LIG-measure mu_k ∝ |d_k| (don ngan sach).
    ito              : True => Ito left-point; False => trapezoid (ton them grad).
    model,target,score: de completeness rescale ve f(x)-f(x0) (tuy chon).

    Tra ve attribution (3,H,W).
    """
    device = x.device
    x0 = blur_baseline
    dx = (x - x0)
    T = max(2, N)
    if sigma_pix is None:
        sigma_pix = ksize / 3.0

    # --- xay path DETERMINISTIC bang PF-ODE-style Euler (khong ton grad) ---
    with torch.no_grad():
        # backbone VP: sqrt(abar) di tu ~0 (x0) -> 1 (x), giong diffusion_ig
        t_grid = torch.linspace(1.0, 0.0, T + 1, device=device)   # 1~x0 .. 0~x
        abar = _vp_abar(t_grid, beta_min, beta_max)
        coef = abar.sqrt().view(-1, 1, 1, 1)                      # (T+1,1,1,1)
        anchor = 4.0 * (torch.linspace(0, 1, T + 1, device=device)
                        * (1 - torch.linspace(0, 1, T + 1, device=device)))  # 0 o 2 dau

        path = torch.empty(T + 1, *x.shape, device=device)
        path[0] = x0
        for k in range(1, T + 1):
            vp_state = x0 + coef[k] * dx                          # vi tri VP-backbone
            # score-proxy: keo ve vung data sac hon tren manifold (blur=heat)
            sp = _score_proxy_deblur(path[k - 1], x, ksize, sigma_pix)
            spn = sp.flatten().norm() + 1e-12
            drift = score_scale * anchor[k] * dx.norm() * (sp / spn)
            path[k] = vp_state + drift
        path[0] = x0; path[-1] = x                                # ep endpoint => completeness

    # --- tich phan doc path deterministic ---
    g = grad_fn(path[:-1])                                        # left-point (T,3,H,W)
    with torch.no_grad():
        dgamma = path[1:] - path[:-1]
        if ito:
            gd = g * dgamma
        else:
            g2 = grad_fn(path[1:])
            gd = 0.5 * (g + g2) * dgamma

        if use_lig:
            dk = gd.flatten(1).sum(dim=1)                         # (T,) grad-predicted change
            mu = dk.abs(); mu = mu / (mu.sum() + 1e-12)           # LIG-measure tau->0
            attr = (mu.view(-1, 1, 1, 1) * gd).sum(dim=0)
        else:
            attr = gd.sum(dim=0)                                  # uniform Ito

    # --- completeness rescale ve f(x)-f(x0) ---
    fvals_fn = _resolve_fvals(None, model, target, score, device)
    if fvals_fn is not None:
        with torch.no_grad():
            total = (fvals_fn(x[None]).reshape(()) - fvals_fn(x0[None]).reshape(()))
            attr = attr * (total / (attr.sum() + 1e-12))
    return attr