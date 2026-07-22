"""
budget_probe_image.py — do C_path va Sigma_base cho IMAGE (ResNet), dung KHOP
convention grad_fn/ig_single cua pea.methods.

Cam vao e1_batch_image.py trong khoi --tau_diag (co grad_fn, diag_pool, ref).
Muc dich: lay diem C_path CAO (image) de doi cuc voi tabular (C_path thap) —
diem quyet dinh cho luat K* ∝ (Sigma_base N^4 / C_path^2)^(1/5).

DOC LAP voi faithfulness: chi dung gradient hinh hoc, KHONG insertion/deletion.
"""
from __future__ import annotations
import torch
from pea.methods import ig_single


@torch.no_grad()
def _noop():  # placeholder to keep import side-effect free
    pass


def c_path_image(x, b, grad_fn, hi=128):
    """
    Path curvature proxy = ||IG_2 - IG_hi|| / ||IG_hi||.
    Bias luong tu hoa midpoint ~ C_path / m^2, nen hieu 2-step vs hi-step la
    proxy TRUC TIEP cho do cong C_path doc doan b->x. Khong dung faithfulness.
    """
    ig2 = ig_single(x, b, grad_fn, T=2)
    ighi = ig_single(x, b, grad_fn, T=hi)
    denom = ighi.norm().clamp_min(1e-8)
    return ((ig2 - ighi).norm() / denom).item()


def sigma_base_image(x, pool, grad_fn, K_probe=16, hi=32):
    """
    Sigma_base = E_b || IG(x,b) - mean_b IG ||^2 tren K_probe baseline that tu pool.
    Spread cua attribution qua baseline — cang lon, EG (diversity) cang co loi.
    hi step vua phai vi anh dat; K_probe nho de kiem soat chi phi.
    """
    B = pool.shape[0]
    idx = torch.randperm(B)[:min(K_probe, B)]
    igs = []
    for i in idx:
        igs.append(ig_single(x, pool[i], grad_fn, T=hi).flatten())
    IG = torch.stack(igs)                       # (K, D)
    mu = IG.mean(0)
    return (IG - mu[None]).pow(2).sum(1).mean().item()


def probe_budget_law(diag_pool, ref, grad_fn_factory, target_default=None,
                     N=64, n_eval=8, shrink_fn=None, tag="image"):
    """
    diag_pool     : list[(x, target)] anh da giu lai.
    ref           : GaussRef (co shrinkage). shrink_fn(x, ref, tau) -> baseline.
    grad_fn_factory(target) -> grad_fn  (vi grad_fn gan voi target cu the).
    Neu grad_fn dung chung 1 target, truyen grad_fn_factory = lambda t: grad_fn.

    In C_path, Sigma_base, K* va DU DOAN winner (EG neu K*>4, else shrinkage).
    KHONG tinh insertion/deletion o day — winner that lay tu bang chinh cua script.
    """
    import numpy as np
    Cs, Sbs = [], []
    pool = torch.stack([p for p, _ in diag_pool])   # (M,3,H,W)
    for x, tgt in diag_pool[:n_eval]:
        t = tgt if target_default is None else target_default
        grad_fn = grad_fn_factory(t)
        b_sh = shrink_fn(x, ref, 1.0)               # representative shrinkage baseline
        Cs.append(c_path_image(x, b_sh, grad_fn))
        Sbs.append(sigma_base_image(x, pool, grad_fn))
    C = float(np.mean(Cs)); Sig = float(np.mean(Sbs))
    Kstar = (Sig * N**4 / max(C, 1e-8)**2) ** 0.2
    pred = "EG" if Kstar > 4 else "shrinkage"
    print(f"\n=== BUDGET-LAW PROBE [{tag}] (n={min(n_eval,len(diag_pool))}, N={N}) ===")
    print(f"[i] C_path (path curvature)   = {C:.4f}   <- KY VONG CAO cho image")
    print(f"[i] Sigma_base (attr spread)  = {Sig:.4f}")
    print(f"[i] K* ∝ (Sigma_base N^4 / C_path^2)^(1/5) = {Kstar:.2f}")
    print(f"[i] DU DOAN: {'DIVERSITY (EG)' if pred=='EG' else 'RESOLUTION (shrinkage)'}")
    print(f"[i]   So voi TABULAR (C_path~0.01, K*~370 => EG). Neu image C_path >> 0.01")
    print(f"[i]   va K* nho => luat GIAI THICH duoc vi sao shrinkage thang o image.")
    print(f"[i] DOI CHIEU: winner THAT o image (tu bang I-D chinh) la shrinkage/blur.")
    print(f"[i]   Neu pred == shrinkage => K* du doan DUNG => luat SONG.")
    print(f"[i]   Neu pred == EG nhung that ra shrinkage thang => luat SAI o image. [FALSIFIED]")
    return dict(C_path=C, Sigma_base=Sig, Kstar=Kstar, pred=pred)
