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
# blur_plain: de-blur reference, UNIFORM measure, Ito. Chinh la BlurIG viet lai
# trong dung khung nay (cung _heat_ref, cung convention Ito left-point) de o
# (de-blur x uniform) cua bang 2x2 khop chinh xac ngan sach/duong tich phan voi
# cac o con lai — khong bi lech do IG-blur di code path khac.
#   sigma=0, drift=0, uniform => day la goc BlurIG cua bang.
# ---------------------------------------------------------------------------
def blur_plain(x, blur_baseline, grad_fn, N):
    """De-blur reference + uniform quadrature + Ito. = BlurIG trong khung nay. (3,H,W)."""
    x0 = blur_baseline
    T = max(2, N)
    path = _heat_ref(x, x0, T)                 # (T+1,3,H,W)
    g = grad_fn(path[:-1])                      # left-point (T,3,H,W)
    with torch.no_grad():
        dgamma = path[1:] - path[:-1]
        return (g * dgamma).sum(dim=0)         # uniform Ito (moi buoc trong so 1)


# ===========================================================================
# blur_reparam: HAP THU measure vao PATH bang reparametrize thoi gian.
#
# Y: IG bat bien voi reparametrization. mu_k ∝ |d_k| (don trong so vao buoc |d| lon)
# TUONG DUONG voi di CHAM lai (nhieu node) o vung |d| lon, di NHANH o vung phang,
# doc CUNG vet de-blur. Sau khi phan bo lai node theo mat do ∝ |d|, quadrature
# UNIFORM tren vet moi == quadrature |d_k|-weighted tren vet cu. Measure bien mat,
# nuot vao lich trinh node s(t). => 1 method path thuan, khong con pha measure rieng.
#
# Thuat toan (arc-length theo |d|):
#   1. probe: di de-blur thang T_p buoc, do |d_k| = |g_k . dgamma_k| moi buoc.
#   2. CDF: C_k = cumsum(|d_k|)/sum(|d_k|)  (tang, 0->1, endpoint-preserving).
#   3. node moi: chon T_i vi tri s_new sao cho khoi luong |d| giua 2 node bang nhau
#      = noi suy nghich dao C tai cac muc deu {i/T_i}. Node dammy vao transition.
#   4. tich phan UNIFORM Ito tren cac node moi (di qua CUNG cac diem anh, chi doi mat do).
#
# Khong drift, khong stochastic. Ngan sach: T_probe + T_integ = N.
# Neu smooth_frac>0, lam muot |d| truoc khi lay CDF de node khong giat.
# ===========================================================================
def blur_reparam(x, blur_baseline, grad_fn, N, probe_frac=0.4, smooth_k=5, eps=1e-9):
    """
    Hap thu mu_k ∝ |d_k| vao path bang reparam thoi gian tren vet de-blur.
    N: ngan sach grad. probe_frac: ti le N danh cho buoc probe do |d_k|.
    smooth_k: cua so lam muot |d| (0 = khong). Tra ve attribution (3,H,W).
    """
    device = x.device
    x0 = blur_baseline
    dx = (x - x0)

    T_probe = max(2, int(N * probe_frac))
    T_integ = max(2, N - T_probe)

    # --- 1. probe de-blur thang, do |d_k| ---
    with torch.no_grad():
        s_probe = torch.linspace(0.0, 1.0, T_probe + 1, device=device).view(-1, 1, 1, 1)
        probe_path = x0[None] + s_probe * dx[None]          # (T_probe+1,3,H,W)
    g_probe = grad_fn(probe_path[:-1])                       # (T_probe,3,H,W)
    with torch.no_grad():
        dgamma_p = probe_path[1:] - probe_path[:-1]
        dk = (g_probe * dgamma_p).flatten(1).sum(dim=1).abs()   # (T_probe,) |d_k|

        # lam muot |d| de node khong giat (moving average)
        if smooth_k and smooth_k > 1:
            ker = torch.ones(1, 1, smooth_k, device=device) / smooth_k
            dk = F.conv1d(dk.view(1, 1, -1), ker, padding=smooth_k // 2).view(-1)[:dk.numel()]
        dk = dk + eps                                        # tranh 0

        # --- 2. CDF theo |d| tren luoi probe (mid-point cua moi buoc) ---
        # gan |d_k| cho khoang [s_k, s_{k+1}]; CDF tai node = cumsum khoi luong
        w = dk / dk.sum()                                    # (T_probe,)
        C = torch.cat([torch.zeros(1, device=device), w.cumsum(0)])   # (T_probe+1,), 0..1
        s_nodes_probe = torch.linspace(0.0, 1.0, T_probe + 1, device=device)  # s tuong ung C

        # --- 3. nghich dao C tai cac muc deu -> node moi dammy vao transition ---
        levels = torch.linspace(0.0, 1.0, T_integ + 1, device=device)        # {i/T_integ}
        # tim s_new sao cho C(s_new) = levels  (noi suy tuyen tinh nghich dao)
        idx = torch.searchsorted(C, levels.clamp(0, 1), right=False).clamp(1, T_probe)
        C_lo, C_hi = C[idx - 1], C[idx]
        s_lo, s_hi = s_nodes_probe[idx - 1], s_nodes_probe[idx]
        frac = ((levels - C_lo) / (C_hi - C_lo + eps)).clamp(0, 1)
        s_new = (s_lo + frac * (s_hi - s_lo)).clamp(0, 1)     # (T_integ+1,) tang, 0->1
        s_new[0] = 0.0; s_new[-1] = 1.0                       # neo endpoint

        # --- 4. path moi: cung vet de-blur, mat do node ∝ |d| ---
        path = x0[None] + s_new.view(-1, 1, 1, 1) * dx[None]  # (T_integ+1,3,H,W)

    g = grad_fn(path[:-1])                                    # left-point Ito (T_integ,3,H,W)
    with torch.no_grad():
        dgamma = path[1:] - path[:-1]
        return (g * dgamma).sum(dim=0)                        # UNIFORM Ito == |d|-weighted tren vet goc



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
                    drift_scale=0.0, drift_iters=0,
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


# ===========================================================================
# SELF-DIFFUSION blur bridge.
#
# Y tuong: blur CHINH LA heat diffusion (d L/d alpha = 1/4 lap L). Nen reference
# process R cua Schrodinger bridge o ngach nay khong phai chon bua Brownian nhu
# SBA — no la heat-semigroup, sinh ra tu chinh baseline. "Self-diffusion" nghia la:
#   - reference     = de-blur path (blur_baseline -> x),
#   - NOISE cua bridge cung nam trong ho scale-space: moi increment Brownian bridge
#     duoc LOC qua chinh heat kernel (spatially-correlated), khong phai noise trang.
# => moi trajectory trung gian van la 1 anh blur HOP LE (tren manifold scale-space),
#    khac han SBA (noise trang -> off-manifold, path lom chom).
#
# Khong co drift theo grad (drift ∇f da chung minh la lam phang |d_k| va tut diem).
# Self-diffusion chi kham pha QUANH path blur TRONG manifold, roi lay expectation (Ito).
# Measure mu_k ∝ |d_k| van con transition sac de bam.
#
#   gamma_t = deblur(t)  +  sigma * anchor(t) * (G_a * xi_t)
#   xi_t    = Brownian bridge trang (0 o 2 dau),  G_a * = heat/blur kernel
#   sigma=0 => BlurIG.  Ito left-point.
# ===========================================================================
@torch.no_grad()
def _heat_kernel1d(ksize, sigma_pix, device, dtype):
    """1D Gaussian kernel (chuan hoa) de lam noise correlated theo scale-space."""
    coords = torch.arange(ksize, device=device, dtype=dtype) - ksize // 2
    g = torch.exp(-(coords ** 2) / (2.0 * sigma_pix ** 2))
    return g / g.sum()


@torch.no_grad()
def _heat_smooth(v, k1d):
    """v: (M,3,H,W) -> blur tach chieu bang k1d (heat kernel). Giu M,3,H,W."""
    C = v.shape[1]
    k = k1d.numel()
    kx = k1d.view(1, 1, 1, k).repeat(C, 1, 1, 1)
    ky = k1d.view(1, 1, k, 1).repeat(C, 1, 1, 1)
    v = F.conv2d(v, kx, padding=(0, k // 2), groups=C)
    v = F.conv2d(v, ky, padding=(k // 2, 0), groups=C)
    return v


@torch.no_grad()
def _heat_bridge_noise(T, shape, sigma, ksize, sigma_pix, device, gen):
    """
    Brownian bridge (0 o 2 dau) nhung MOI increment duoc lam muot bang heat kernel
    => noise spatially-correlated, song tren manifold scale-space. Tra (T+1,3,H,W).
    """
    dt = 1.0 / T
    dW = torch.randn(T, *shape, generator=gen, device=device) * (dt ** 0.5)  # (T,3,H,W)
    dW = _heat_smooth(dW, _heat_kernel1d(ksize, sigma_pix, device, dW.dtype))  # correlated
    W = torch.cat([torch.zeros(1, *shape, device=device), dW.cumsum(dim=0)], dim=0)  # (T+1,...)
    t = torch.linspace(0.0, 1.0, T + 1, device=device).view(-1, 1, 1, 1)
    BB = W - t * W[-1:]                # bridge: 0 o t=0 va t=1
    return sigma * BB                 # (T+1,3,H,W)


def blur_selfdiff(x, blur_baseline, grad_fn, N,
                  sigma=0.3, P=1, ksize=31, sigma_pix=None, gen=None):
    """
    Self-diffusion blur bridge: de-blur reference + heat-correlated bridge noise, Ito.
    x, blur_baseline: (3,H,W). N: ngan sach grad. P: so trajectory (B*P*T ~ N).
    sigma: bien do noise (0 => BlurIG). ksize/sigma_pix: heat kernel lam muot noise.
    Tra ve attribution (3,H,W).
    """
    device = x.device
    x0 = blur_baseline
    T = max(2, N // max(1, P))
    if sigma_pix is None:
        sigma_pix = ksize / 3.0
    shape = x.shape

    ref = _heat_ref(x, x0, T)                              # (T+1,3,H,W) de-blur reference
    t = torch.linspace(0.0, 1.0, T + 1, device=device)
    anch = _bridge_anchor(t).view(-1, 1, 1, 1)            # 0 o 2 dau

    acc = torch.zeros_like(x)
    for _ in range(max(1, P)):
        with torch.no_grad():
            noise = _heat_bridge_noise(T, shape, sigma, ksize, sigma_pix, device, gen)
            path = ref + anch * noise                    # (T+1,3,H,W), neo 2 dau
            path[0] = x0; path[-1] = x
        g = grad_fn(path[:-1])                            # left-point Ito, (T,3,H,W)
        with torch.no_grad():
            acc += (g * (path[1:] - path[:-1])).sum(dim=0)
    return acc / max(1, P)


def blur_selfdiff_lig(x, blur_baseline, grad_fn, N,
                      sigma=0.3, P=1, ksize=31, sigma_pix=None, gen=None,
                      model=None, target=None, score="logit"):
    """
    Self-diffusion path (nhu blur_selfdiff) nhung do bang LIG-measure mu_k ∝ |d_k|,
    gop tren P trajectory (measure tinh tren path trung binh de co 1 mu_k on dinh).
    Rescale completeness ve f(x)-f(x0) neu co (model,target).
    """
    device = x.device
    x0 = blur_baseline
    T = max(2, N // max(1, P))
    if sigma_pix is None:
        sigma_pix = ksize / 3.0
    shape = x.shape

    ref = _heat_ref(x, x0, T)
    t = torch.linspace(0.0, 1.0, T + 1, device=device)
    anch = _bridge_anchor(t).view(-1, 1, 1, 1)

    # gop grad*dgamma va d_k tren P trajectory
    acc_gd = torch.zeros(T, *shape, device=device)        # sum_P g_k*dgamma_k (per step)
    acc_dk = torch.zeros(T, device=device)                # sum_P d_k
    for _ in range(max(1, P)):
        with torch.no_grad():
            noise = _heat_bridge_noise(T, shape, sigma, ksize, sigma_pix, device, gen)
            path = ref + anch * noise
            path[0] = x0; path[-1] = x
        g = grad_fn(path[:-1])                            # (T,3,H,W)
        with torch.no_grad():
            dgamma = path[1:] - path[:-1]
            gd = g * dgamma                               # (T,3,H,W)
            acc_gd += gd
            acc_dk += gd.flatten(1).sum(dim=1)            # d_k per step

    with torch.no_grad():
        mu = acc_dk.abs()
        mu = mu / (mu.sum() + 1e-12)                      # LIG-measure tren path trung binh
        attr = (mu.view(-1, 1, 1, 1) * (acc_gd / max(1, P))).sum(dim=0)

    fvals_fn = _resolve_fvals(None, model, target, score, device)
    if fvals_fn is not None:
        with torch.no_grad():
            total = (fvals_fn(x[None]).reshape(()) - fvals_fn(x0[None]).reshape(()))
            attr = attr * (total / (attr.sum() + 1e-12))
    return attr