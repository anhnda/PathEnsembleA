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
    Bias luong tu hoa TUYET DOI (giu phu thuoc do dai path L=||x-b||):
    ||IG_2 - IG_hi||  — KHONG chuan hoa theo ||IG_hi|| (chuan hoa se BO MAT phu
    thuoc L, ma L moi la bien phan biet image vs tabular). Tra ve (bias_abs, L).

    Ly do (sua tu ban truoc, da bi falsify): remainder cau phuong ~ C^2 * L^2 / m^2.
    EG thua o image vi baseline la anh ngau nhien XA (L lon 400-1000) => path dai,
    coarse => bias no. Shrinkage thang vi L nho (~150) + du step. Phai giu L.
    """
    L = (x - b).norm().item()
    ig2 = ig_single(x, b, grad_fn, T=2)
    ighi = ig_single(x, b, grad_fn, T=hi)
    bias_abs = (ig2 - ighi).norm().item()
    return bias_abs, L


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
                     N=64, n_eval=8, shrink_fn=None, pool_baselines=None,
                     eg_pool_fn=None, tag="image"):
    """
    So sanh TRUC TIEP hai chinh sach ngan sach bang BIAS DO DUOC (khong dung cong
    thuc K* da bi falsify). Voi cung N:
      - shrinkage: 1 baseline GAN (L nho), N step  -> bias thap.
      - EG-K     : K baseline XA (anh ngau nhien, L lon), N/K step -> bias cao.

    Do bias = ||IG_coarse - IG_hi|| cho moi chinh sach, roi bao chinh sach nao bias
    thap hon => du doan chinh sach do thang. Doi chieu voi winner THAT (bang I-D).
    """
    import numpy as np
    pool = pool_baselines if pool_baselines is not None else \
           torch.stack([p for p, _ in diag_pool])
    L_sh, L_eg = [], []
    bias_sh, bias_eg = [], []
    Sbs = []
    for x, tgt in diag_pool[:n_eval]:
        t = tgt if target_default is None else target_default
        grad_fn = grad_fn_factory(t)

        # --- shrinkage policy: gan, full N step ---
        b_sh = shrink_fn(x, ref, 1.0)
        L_sh.append((x - b_sh).norm().item())
        # bias cua path shrinkage o do phan giai N: ||IG_N - IG_hi||
        ig_full = ig_single(x, b_sh, grad_fn, T=min(N, 128))
        ig_hi   = ig_single(x, b_sh, grad_fn, T=256)
        bias_sh.append((ig_full - ig_hi).norm().item())

        # --- EG policy: K=16 baseline (x + noise), N/16 step moi cai ---
        K = 16; m = max(2, N // K)
        if eg_pool_fn is not None:
            egp = eg_pool_fn(x)                     # (K,3,H,W) = x + N(0,noise)
        else:
            egp = pool[torch.randperm(len(pool))[:K]]
        per_bias = []
        Ls = []
        for j in range(egp.shape[0]):
            b = egp[j]
            Ls.append((x - b).norm().item())
            igc = ig_single(x, b, grad_fn, T=m)
            igh = ig_single(x, b, grad_fn, T=128)
            per_bias.append((igc - igh).norm().item())
        L_eg.append(float(np.mean(Ls)))
        bias_eg.append(float(np.mean(per_bias)))

    L_sh_m, L_eg_m = float(np.mean(L_sh)), float(np.mean(L_eg))
    b_sh_m, b_eg_m = float(np.mean(bias_sh)), float(np.mean(bias_eg))
    pred = "shrinkage" if b_sh_m < b_eg_m else "EG"

    print(f"\n=== BUDGET-LAW PROBE v2 [{tag}] (n={min(n_eval,len(diag_pool))}, N={N}) ===")
    print(f"[i] SHRINKAGE policy: L=||b-x||={L_sh_m:8.1f}  (gan)   bias@Nstep = {b_sh_m:.4f}")
    print(f"[i] EG-16    policy: L=||b-x||={L_eg_m:8.1f}  (xa)    bias@(N/16)= {b_eg_m:.4f}")
    print(f"[i] Bias ~ C^2 * L^2 / m^2. EG co L LON + m NHO (N/16) => bias no.")
    print(f"[i] DU DOAN (bias thap hon thang): {pred.upper()}")
    print(f"[i] DOI CHIEU winner THAT o image = shrinkage/blur (bang I-D: shr@4=5.01 vs EG-16=3.41).")
    if pred == "shrinkage":
        print(f"[OK] v2 du doan DUNG: bias-do-duoc chon shrinkage, khop winner that.")
        print(f"[OK]  => bien phan biet la L (do dai path) x m (step), KHONG phai Sigma_base.")
    else:
        print(f"[!!] v2 VAN SAI: bias-do-duoc chon EG nhung shrinkage thang. Con thieu so hang khac.")
    return dict(L_shrink=L_sh_m, L_eg=L_eg_m, bias_shrink=b_sh_m, bias_eg=b_eg_m, pred=pred)
