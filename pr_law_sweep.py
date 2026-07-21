"""
pr_law_sweep.py — KIEM LUAT 'PR du doan tau-sensitivity' tren NHIEU dataset.

Muc dich DUY NHAT: bien bang LUAT tu 1 diem thanh >= 4 diem trai PR/D, de xem
sensitivity do duoc co CHUYEN TIEP theo PR khong (flat khi PR thap, sharp khi PR cao).

Moi dataset:
  1. standardise, train MLP nho (differentiable — IG can grad).
  2. uoc luong Sigma (standardised features) + ridge floor -> pho s_k -> PR.
  3. sweep tau: b_tau = mu + Sigma(Sigma+tau I)^-1 (x-mu); IG tu b_tau -> x.
  4. faithfulness = insertion - deletion (remove-to-marginal, KHONG tai dung b_tau).
  5. dump curve -> chay tau_regime.check_match -> MATCH/MISMATCH.

TRUNG THUC: day la SANITY thi nghiem tren toy data (sklearn), KHONG phai ket qua
paper. Muc dich la kiem TRUC PR co dung khong TRUOC khi dau tu chay full. Neu
ngay ca o toy data ma PR khong monotonic voi sensitivity => truc sai, dung som.

Torch. Chay: python pr_law_sweep.py
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

import tau_regime as tr

torch.manual_seed(0)
np.random.seed(0)
DEV = "cpu"


# ---------------------------------------------------------------------------
# Model: MLP nho, differentiable
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, d, k, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(),
                                 nn.Linear(h, k))

    def forward(self, x):
        return self.net(x)


def train_mlp(Xtr, ytr, d, k, epochs=300, lr=1e-2):
    m = MLP(d, k).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    Xtr, ytr = Xtr.to(DEV), ytr.to(DEV)
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(m(Xtr), ytr)
        loss.backward(); opt.step()
    return m


# ---------------------------------------------------------------------------
# Sigma + pho
# ---------------------------------------------------------------------------
def fit_reference(X, lam=1e-3):
    mu = X.mean(0)
    Xc = X - mu
    S = (Xc.T @ Xc) / (X.shape[0] - 1)
    S = 0.5 * (S + S.T) + lam * torch.eye(S.shape[1], dtype=X.dtype)
    s, V = torch.linalg.eigh(S)              # tang dan
    s = s.clamp_min(0)
    return mu, s, V, S


def shrink_baseline(x, mu, S, tau):
    """b_tau = mu + S (S + tau I)^-1 (x - mu). x: (d,) hoac (M,d)."""
    d = S.shape[0]
    A = S + tau * torch.eye(d, dtype=S.dtype)
    W = S @ torch.linalg.inv(A)              # (d,d)
    single = x.dim() == 1
    X = x[None] if single else x
    B = mu[None] + (X - mu[None]) @ W.T
    return B[0] if single else B


# ---------------------------------------------------------------------------
# IG (straight line, autograd) + insertion/deletion remove-to-marginal
# ---------------------------------------------------------------------------
def ig_attr(model, x, base, target, steps=32):
    x = x.to(DEV); base = base.to(DEV)
    alphas = torch.linspace(0, 1, steps, device=DEV).view(-1, 1)
    path = base[None] + alphas * (x - base)[None]
    path.requires_grad_(True)
    out = model(path)[:, target]
    g = torch.autograd.grad(out.sum(), path)[0]
    return (x - base) * g.mean(0)


@torch.no_grad()
def insertion_deletion(model, x, attr, target, ref_pool, steps=None):
    """
    remove-to-marginal: thay feature bang gia tri THAT tu ref_pool (giu in-distribution).
    insertion: bat dau tu nen marginal, them dan feature quan trong nhat.
    deletion  : bat dau tu x, xoa dan feature quan trong nhat.
    Tra ve (ins_auc, del_auc).
    """
    d = x.numel()
    steps = steps or d
    order = torch.argsort(attr.abs(), descending=True)
    # nen marginal = mot mau ngau nhien tu pool (in-distribution)
    fill = ref_pool[torch.randint(len(ref_pool), (1,))][0].to(DEV)

    def prob(v):
        return torch.softmax(model(v[None])[0], 0)[target].item()

    # insertion
    cur = fill.clone(); ins = [prob(cur)]
    for j in range(d):
        cur[order[j]] = x[order[j]]
        ins.append(prob(cur))
    # deletion
    cur = x.clone(); dele = [prob(cur)]
    for j in range(d):
        cur[order[j]] = fill[order[j]]
        dele.append(prob(cur))
    _trap = getattr(np, "trapezoid", getattr(np, "trapz", None))
    ins_auc = _trap(ins) / (len(ins) - 1)
    del_auc = _trap(dele) / (len(dele) - 1)
    return ins_auc, del_auc


# ---------------------------------------------------------------------------
# Curve cho MOT dataset: sweep tau, dump long-format CSV cho tau_regime
# ---------------------------------------------------------------------------
def build_curve_csv(model, Xte, yte, mu, s, V, S, target, ref_pool,
                    taus, out_csv, n_eval=None):
    import csv as _csv
    idx = (yte == target).nonzero().flatten()
    if n_eval:
        idx = idx[:n_eval]
    with open(out_csv, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=["i", "tau", "f_x", "f_b", "delta_f",
                                            "dist", "rate", "maha_x", "maha_b",
                                            "valid", "id_gap"])
        w.writeheader()
        for ii, i in enumerate(idx):
            x = Xte[i].to(DEV)
            with torch.no_grad():
                fx = torch.softmax(model(x[None])[0], 0)[target].item()
            valid = int(fx >= 0.05)
            prev_df = None
            for t in taus:
                b = shrink_baseline(x, mu, S, t)
                with torch.no_grad():
                    fb = torch.softmax(model(b[None])[0], 0)[target].item()
                df = fx - fb
                dist = (x - b).norm().item()
                attr = ig_attr(model, x, b, target)
                ins, dele = insertion_deletion(model, x, attr, target, ref_pool)
                idg = ins - dele
                w.writerow({"i": ii, "tau": t, "f_x": fx, "f_b": fb,
                            "delta_f": df, "dist": dist, "rate": 0.0,
                            "maha_x": 0.0, "maha_b": 0.0,
                            "valid": valid, "id_gap": idg})
    return out_csv


# ---------------------------------------------------------------------------
# MAIN: chay nhieu dataset, trai PR
# ---------------------------------------------------------------------------
def run_dataset(name, loader, n_eval=15, taus=None):
    from sklearn.model_selection import train_test_split
    data = loader()
    X = torch.tensor(data.data, dtype=torch.float64)
    y = torch.tensor(data.target, dtype=torch.long)
    X = (X - X.mean(0)) / X.std(0).clamp_min(1e-8)          # standardise
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0,
                                          stratify=y)
    d, k = X.shape[1], int(y.max()) + 1
    Xtr32 = Xtr.float(); model = train_mlp(Xtr32, ytr, d, k)
    model = model.double()
    acc = (model(Xte).argmax(1) == yte).float().mean().item()

    mu, s, V, S = fit_reference(Xtr)
    PR = tr.participation_ratio(s)
    target = int(torch.bincount(yte).argmax())            # lop pho bien nhat

    if taus is None:
        sb = s.mean().item()
        taus = [float(g) for g in torch.logspace(
            np.log10(1e-2 * sb), np.log10(1e2 * sb), 24)]

    out_csv = f"/home/claude/curve_{name}.csv"
    build_curve_csv(model, Xte, yte, mu, s, V, S, target,
                    ref_pool=Xtr, taus=taus, out_csv=out_csv, n_eval=n_eval)

    res = tr.check_match(ref_s=s, curve_path=out_csv, metric="id_gap",
                         tag=f"{name}")
    res["_acc"] = acc; res["_D"] = d
    return res


def main():
    from sklearn.datasets import (load_iris, load_wine, load_breast_cancer,
                                   load_digits)
    jobs = [
        ("iris", load_iris),            # D=4,  pho co the tap trung
        ("wine", load_wine),            # D=13, PR/D~0.39 (da biet: flat)
        ("bcancer", load_breast_cancer),# D=30, features tuong quan cao
        ("digits", load_digits),        # D=64, pixel — pho TRAI (ky vong PR/D cao)
    ]
    entries = []
    for name, loader in jobs:
        print(f"\n########## {name} ##########")
        try:
            res = run_dataset(name, loader)
            print(f"[i] test acc = {res['_acc']:.3f}, D = {res['_D']}")
            tr.print_regime_check(res)
            entries.append(res)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[!] {name} loi: {e}")

    tr.print_pr_sensitivity_law(entries)


if __name__ == "__main__":
    main()
