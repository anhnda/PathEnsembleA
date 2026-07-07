"""
BlurLIG-Full — LIG day du (Algorithm 1) tren blur reference.

Khac blur_lig (chi Phase 1, path co dinh): o day chay ALTERNATING MINIMISATION
dung nhu Algorithm 1 cua LIG paper — Phase 1 update measure (convex QP), Phase 2
update PATH bang grouped-velocity heuristic. Lap T vong.

Khoi tao:
  - baseline x0 = blur reference (ban mo cua x). Day la "blur reference" cua ngach nay.
  - path ban dau = DUONG THANG x0 -> x  (velocity matrix V = all-ones => uniform delivery).
  - alternating minimisation be duong thang do dan (Phase 2), dong thoi don measure (Phase 1).

Objective (Definition 3, LIG):
  J(gamma, mu) = Var_nu(phi)  -  lam * sum_k mu_k |d_k|  +  (tau/2) ||mu||^2
  voi  d_k = g_k . dgamma_k,  df_k = f(gamma_{k+1}) - f(gamma_k),  phi_k = d_k/df_k,
       nu_k = mu_k df_k^2 / sum_j mu_j df_j^2   (effective measure, Def 1),
       phibar = sum_k nu_k phi_k,  Var_nu(phi) = sum_k nu_k (phi_k - phibar)^2.

Phase 1 (measure, exact-ish): update mu tren simplex.
  Bản blur_lig da chi ra gioi han tau->0 cho mu_k ∝ |d_k|. O day giu convex QP tong quat
  (projected-gradient tren simplex) de con so-sanh voi Phase 2. Neu muon exact thuan
  |d_k|, dat use_exact_measure=True.

Phase 2 (path, grouped-velocity heuristic — Algorithm 1 line 6):
  - partition feature thanh G group theo |grad_i f(x)| (grouped by gradient importance).
  - V in R^{G x N}, khoi tao all-ones. Delivery: dgamma_{k,i} = dx_i * V_{g(i),k}/sum_m V_{g(i),m}.
  - moi vong: pick 1 probe group ngau nhien, uoc luong dJ/dV_{g,k} bang stochastic
    finite-difference, buoc theo huong giam. => O(N) model-eval/vong thay vi O(G N).

Completeness: rescale A sao cho sum(A) = f(x) - f(x0)  (line 10).

API:
  grad_fn(states (M,3,H,W)) -> grads (M,3,H,W)         # grad logit target theo input
  fvals_fn(states (M,3,H,W)) -> f (M,)                 # gia tri f target (cho df_k, completeness)
Ca hai deu tu quan autograd/no_grad ben trong. Tra ve attribution (3,H,W).

GPU theo device cua x. KHONG chay gi o day.
"""

from __future__ import annotations
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# fvals_fn helper: can gia tri f (khong chi grad) de tinh df_k va completeness.
# ---------------------------------------------------------------------------
def make_fvals_fn(model, target, device, chunk=16, score="logit"):
    """(M,3,H,W)->(M,) gia tri f target. Dung CHUNG score voi grad_fn/metric."""
    @torch.no_grad()
    def fvals(states):
        M = states.shape[0]
        out = torch.empty(M, device=device)
        for i in range(0, M, chunk):
            sb = states[i:i + chunk].to(device)
            logits = model(sb)
            if score == "softmax":
                out[i:i + chunk] = F.log_softmax(logits, dim=1)[:, target]
            else:
                out[i:i + chunk] = logits[:, target]
        return out
    return fvals


# ---------------------------------------------------------------------------
# Xay path tu velocity matrix V (Algorithm 1): delivery theo group.
#   dgamma_{k,i} = dx_i * V_{g(i),k} / sum_m V_{g(i),m}
#   gamma_k = x0 + cumsum_{j<k} dgamma_j        (gamma_0 = x0, gamma_N = x)
# V: (G, N) >= 0. group_index: (D,) group cua moi feature (flatten 3*H*W).
# ---------------------------------------------------------------------------
@torch.no_grad()
def _path_from_V(x0, dx, V, group_index, shape):
    """Tra ve path (N+1, 3, H, W) tu velocity matrix V (G,N)."""
    G, N = V.shape
    Vpos = V.clamp_min(1e-8)                          # giu duong de delivery hop le
    frac = Vpos / Vpos.sum(dim=1, keepdim=True)       # (G, N): ty le giao moi buoc, sum_k = 1
    frac_feat = frac[group_index]                     # (D, N): moi feature lay frac cua group no
    dx_flat = dx.reshape(-1)                          # (D,)
    dgamma = dx_flat[:, None] * frac_feat             # (D, N): dgamma_{i,k}
    gamma = torch.zeros(N + 1, dx_flat.numel(), device=dx.device)
    gamma[0] = x0.reshape(-1)
    gamma[1:] = x0.reshape(-1)[None] + dgamma.cumsum(dim=1).T   # (N+1, D)
    return gamma.reshape(N + 1, *shape)


# ---------------------------------------------------------------------------
# Objective J(gamma, mu) tu {d_k, df_k} da tinh san.
# ---------------------------------------------------------------------------
@torch.no_grad()
def _objective(dk, dfk, mu, lam, tau, eps=1e-12):
    """J = Var_nu(phi) - lam sum mu|d| + (tau/2)||mu||^2. dk,dfk,mu: (N,)."""
    df2 = dfk * dfk + eps
    nu = mu * df2
    nu = nu / (nu.sum() + eps)
    phi = dk / torch.where(dfk.abs() > 0, dfk, torch.ones_like(dfk))
    phi = phi.clamp(-10, 10)
    phibar = (nu * phi).sum()
    var = (nu * (phi - phibar) ** 2).sum()
    return var - lam * (mu * dk.abs()).sum() + 0.5 * tau * (mu * mu).sum()


# ---------------------------------------------------------------------------
# Phase 1: update mu (measure) tren simplex.
# ---------------------------------------------------------------------------
@torch.no_grad()
def _update_measure(dk, dfk, lam, tau, iters=300, lr=1e-2, use_exact=False, eps=1e-12):
    """Tra ve mu (N,) tren simplex toi thieu J theo mu (gamma co dinh)."""
    N = dk.numel()
    if use_exact:
        mu = dk.abs()
        return mu / (mu.sum() + eps)                 # gioi han tau->0: mu_k ∝ |d_k|

    df2 = dfk * dfk + eps
    absd = dk.abs()
    phi = (dk / torch.where(dfk.abs() > 0, dfk, torch.ones_like(dfk))).clamp(-10, 10)
    mu = torch.full((N,), 1.0 / N, device=dk.device)
    for _ in range(iters):
        nu = mu * df2; nu = nu / (nu.sum() + eps)
        phibar = (nu * phi).sum()
        # grad xap xi cua Var_nu(phi) theo mu (bo coupling bac 2 cua nu) + phan signal/reg
        var_g = df2 * (phi - phibar) ** 2 / ((mu * df2).sum() + eps)
        grad = var_g - lam * absd + tau * mu
        mu = mu - lr * grad
        # chieu len simplex (Duchi projection)
        s, _ = torch.sort(mu, descending=True)
        css = (s.cumsum(0) - 1.0) / torch.arange(1, N + 1, device=mu.device)
        rho = (s > css).nonzero()
        if rho.numel() == 0:
            mu = torch.full((N,), 1.0 / N, device=dk.device); continue
        theta = css[rho.max()]
        mu = (mu - theta).clamp_min(0.0)
    return mu


# ---------------------------------------------------------------------------
# Danh gia d_k, df_k cho 1 path (dung grad_fn + fvals_fn).
# ---------------------------------------------------------------------------
def _eval_path(path, grad_fn, fvals_fn):
    """Tra ve (dk, dfk, g_left, dgamma). path: (N+1,3,H,W)."""
    g = grad_fn(path[:-1])                            # (N,3,H,W) left-point grad
    with torch.no_grad():
        dgamma = path[1:] - path[:-1]                 # (N,3,H,W)
        dk = (g * dgamma).flatten(1).sum(dim=1)       # (N,)
        fv = fvals_fn(path)                           # (N+1,)
        dfk = fv[1:] - fv[:-1]                         # (N,)
    return dk, dfk, g, dgamma


# ---------------------------------------------------------------------------
# Full LIG on blur reference.
# ---------------------------------------------------------------------------
def blur_lig_full(
    x, blur_baseline, grad_fn, fvals_fn,
    group_index, G, N,
    T=10, lam=1.0, tau=0.01,
    path_lr=0.5, fd_eps=0.05, use_exact_measure=False,
    generator=None, model=None, target=None, score="logit",
):
    """
    x, blur_baseline : (3,H,W). x0 = blur_baseline (blur reference).
    grad_fn          : (M,3,H,W)->(M,3,H,W).
    fvals_fn         : (M,3,H,W)->(M,). Neu None va co (model,target) thi tu tao.
    group_index      : (D,) long, group cua moi feature (dung make_patch_groups).
    G                : so group.
    N                : so buoc path (= so grad eval/vong tich phan).
    T                : so vong alternating minimisation.
    lam, tau         : he so signal / regulariser.
    path_lr          : buoc cap nhat velocity V (Phase 2).
    fd_eps           : do lon finite-difference probe cho dJ/dV.
    use_exact_measure: True -> Phase 1 dung mu_k ∝ |d_k| (gioi han tau->0) thay vi QP.

    Tra ve attribution (3,H,W).
    """
    device = x.device
    shape = x.shape
    x0 = blur_baseline
    dx = (x - x0)
    if fvals_fn is None:
        assert model is not None and target is not None, "can fvals_fn hoac (model,target)"
        fvals_fn = make_fvals_fn(model, target, device, score=score)

    # khoi tao velocity: all-ones => path ban dau la DUONG THANG x0->x (uniform delivery)
    V = torch.ones(G, N, device=device)

    mu = torch.full((N,), 1.0 / N, device=device)

    for s in range(T):
        # ---- xay path tu V hien tai ----
        path = _path_from_V(x0, dx, V, group_index, shape)
        dk, dfk, g, dgamma = _eval_path(path, grad_fn, fvals_fn)

        # ---- Phase 1: update measure (fix path) ----
        mu = _update_measure(dk, dfk, lam, tau, use_exact=use_exact_measure)

        # ---- Phase 2: update path (fix measure), grouped-velocity heuristic ----
        if s < T - 1:
            J0 = _objective(dk, dfk, mu, lam, tau)
            # pick 1 probe group ngau nhien (Algorithm 1: 1 group/vong)
            gsel = int(torch.randint(0, G, (1,), generator=generator, device=device).item())
            gradV_row = torch.zeros(N, device=device)
            # stochastic finite-difference: uoc luong dJ/dV_{gsel,k} cho tung k
            # (probe rieng tung buoc; van O(N) model-eval vi chi 1 group thay doi delivery)
            for k in range(N):
                Vp = V.clone()
                Vp[gsel, k] = Vp[gsel, k] + fd_eps
                path_p = _path_from_V(x0, dx, Vp, group_index, shape)
                dk_p, dfk_p, _, _ = _eval_path(path_p, grad_fn, fvals_fn)
                with torch.no_grad():
                    Jp = _objective(dk_p, dfk_p, mu, lam, tau)
                    gradV_row[k] = (Jp - J0) / fd_eps
            with torch.no_grad():
                V[gsel] = (V[gsel] - path_lr * gradV_row).clamp_min(1e-8)

    # ---- attribution cuoi tu (path*, mu*) ----
    path = _path_from_V(x0, dx, V, group_index, shape)
    g = grad_fn(path[:-1])
    with torch.no_grad():
        dgamma = path[1:] - path[:-1]
        attr = (mu.view(-1, 1, 1, 1) * g * dgamma).sum(dim=0)   # sum_k mu_k g_k ⊙ dgamma_k
        # completeness rescale (line 10)
        total = (fvals_fn(x[None]).reshape(()) - fvals_fn(x0[None]).reshape(()))
        attr = attr * (total / (attr.sum() + 1e-12))
    return attr