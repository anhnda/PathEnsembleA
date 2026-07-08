"""
synthetic_e0.py — E0 (semantic go/no-go) + E1 (baseline comparison) tren TABULAR synthetic.

Xay theo draft LaTeX (Sec. Experiments):

  E0 — semantic go/no-go (chay TRUOC moi benchmark, moi modality rieng).
    Du lieu synthetic + model kha vi + BA regime khop nhau:
      (a) two-equal : hai feature du bao NGANG NHAU nhung khac phuong sai.
      (b) high-var  : feature du bao la feature PHUONG SAI CAO.
      (c) low-var   : feature du bao la feature PHUONG SAI THAP.
    Cau hoi: bias tau/(s_k+tau) (Eq. residual) co PHUC HOI dung driver that khong,
    hay OVER-CREDIT feature phuong-sai-thap BAT KE tinh du bao?
    Neu over-credit bat ke => method la "deviation-from-prior attribution", bao cao dung the.
    Bao cao: AUPRC / rank-corr(attribution, |true-driver|) theo tung regime.

  E1 — baseline comparison (theo compare.py, tabular).
    IG tu: zero, mean, median, random-sample, EG(K in {1,4,16,64}),
    Shrinkage-IG (swept tau), PM-IG-PPCA (psi uoc luong).
    Metric: conditional-imputation insertion/deletion (imputer fit tren split ROI,
    KHONG tai dung baseline), + stability. Bootstrap CI tren tap test.

Uu tien torch GPU (device theo --device). KHONG train benchmark ngoai, KHONG smoketest.
Ban tu chay:
    python synthetic_e0.py                         # E0 ca 3 regime + E1
    python synthetic_e0.py --only e0
    python synthetic_e0.py --only e1 --regime high-var
    python synthetic_e0.py --tau_sweep 0.1 1 10 100 --device cuda

Cong thuc baseline (draft, Prop. closed form / Eq. residual / Eq. lowrank):
    b_tau(x) = mu + Sigma (Sigma + tau I)^{-1} (x - mu)
    x - b_tau(x) = tau (Sigma + tau I)^{-1} (x - mu)
  eigenbasis Sigma = V diag(s) V^T:
    [b_tau(x) - mu]_k = s_k/(s_k+tau) * (x-mu)_k        (Wiener gain, Cor. Wiener)
  PM-IG-PPCA (Cor. MMSE): fit Sigma_Z = W W^T, tau = psi (FA/PPCA), khong ridge floor.
"""

from __future__ import annotations
import argparse
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# (0) Reference (mu, Sigma) va cac dang baseline shrinkage.
#   Lam viec o eigenbasis cua Sigma cho re va sach (D nho o tabular).
# ===========================================================================
@dataclass
class GaussRef:
    """Prior tham chieu N(mu, Sigma) da phan ra eigen: Sigma = V diag(s) V^T."""
    mu: torch.Tensor      # (D,)
    V: torch.Tensor       # (D,D) cot = eigenvector
    s: torch.Tensor       # (D,)  eigenvalue (phuong sai theo truc PCA), giam dan

    @property
    def device(self):
        return self.mu.device


@torch.no_grad()
def fit_reference(X_ref: torch.Tensor, floor: float = 1e-6) -> GaussRef:
    """
    Uoc luong (mu, Sigma) tu tap tham chieu X_ref (N,D) + ridge floor lambda I
    (draft Sec. Covariance Estimation): Sigma = 1/2(S+S^T) + lambda I, roi eigh.
    floor = lambda (dam bao Sigma > 0, giu Tikhonov reading khi tau->0).
    """
    mu = X_ref.mean(dim=0)
    Xc = X_ref - mu[None]
    N = X_ref.shape[0]
    S = (Xc.transpose(0, 1) @ Xc) / max(1, N - 1)          # (D,D) sample cov
    S = 0.5 * (S + S.transpose(0, 1))
    S = S + floor * torch.eye(S.shape[0], device=S.device, dtype=S.dtype)
    s, V = torch.linalg.eigh(S)                             # tang dan
    order = torch.argsort(s, descending=True)
    return GaussRef(mu=mu, V=V[:, order].contiguous(), s=s[order].contiguous())


@torch.no_grad()
def shrinkage_baseline(x: torch.Tensor, ref: GaussRef, tau: float) -> torch.Tensor:
    """
    b_tau(x) = mu + V diag(s/(s+tau)) V^T (x - mu).  x: (D,) hoac (M,D).
    tau -> 0 : b -> x (contrast -> 0).  tau -> inf : b -> mu (mean baseline).
    """
    single = (x.dim() == 1)
    Xq = x[None] if single else x                          # (M,D)
    d = Xq - ref.mu[None]                                   # (M,D)
    coeff = d @ ref.V                                       # (M,D) toa do eigen
    g = ref.s / (ref.s + tau)                              # (D,) Wiener gain
    b = ref.mu[None] + (coeff * g[None]) @ ref.V.transpose(0, 1)
    return b[0] if single else b


@torch.no_grad()
def fit_ppca(X_ref: torch.Tensor, q: int) -> tuple[GaussRef, float]:
    """
    PPCA/FA (Cor. MMSE): Sigma_Z = W W^T (rank q), psi = phuong sai nhieu con lai.
    Tra ve (ref_ppca, psi) voi ref_ppca.s = eigenvalue cua Sigma_Z (co truc q),
    psi = tau exact-MMSE (khong sweep, khong ridge floor rieng).
    ML PPCA (Tipping-Bishop): psi = mean cua (D-q) eigenvalue nho cua sample cov;
    s_k(W) = eigval_k - psi (k<=q), clamp >=0.
    """
    mu = X_ref.mean(dim=0)
    Xc = X_ref - mu[None]
    N, D = X_ref.shape
    S = (Xc.transpose(0, 1) @ Xc) / max(1, N - 1)
    S = 0.5 * (S + S.transpose(0, 1))
    lam, U = torch.linalg.eigh(S)                          # tang dan
    order = torch.argsort(lam, descending=True)
    lam = lam[order]; U = U[:, order].contiguous()
    q = min(q, D - 1)
    psi = float(lam[q:].mean().clamp_min(1e-8).item())     # phuong sai nhieu ML
    s_z = (lam[:q] - psi).clamp_min(0.0)                   # factor strengths
    s_full = torch.zeros(D, device=X_ref.device, dtype=X_ref.dtype)
    s_full[:q] = s_z                                        # con lai = 0 (chi psi giu)
    ref = GaussRef(mu=mu, V=U, s=s_full)
    return ref, psi


# ===========================================================================
# (1) Du lieu synthetic + model kha vi cho BA regime E0.
#   Moi feature j la Gaussian voi phuong sai sigma_j^2 khac nhau.
#   Nhan y = f(x) chi phu thuoc TAP DRIVER that (khai bao ro), qua MLP kha vi.
#   Ta huan luyen NHE MLP de fit ham target that -> model co gradient co y nghia.
# ===========================================================================
@dataclass
class Regime:
    name: str
    D: int
    sigmas: torch.Tensor      # (D,) do lech chuan moi feature
    driver_idx: torch.Tensor  # (k,) chi so feature LA driver that
    driver_w: torch.Tensor    # (k,) trong so trong ham target (predictiveness)


def make_regime(name: str, D: int, device, seed: int) -> Regime:
    """
    Ba regime khop nhau (draft E0):
      two-equal : 2 driver du bao NGANG (|w| bang nhau) nhung sigma khac nhau.
      high-var  : 1 driver = feature phuong sai CAO.
      low-var   : 1 driver = feature phuong sai THAP.
    Cac feature con lai la nhieu (khong vao ham target). sigma trai deu tren log-scale.
    """
    g = torch.Generator(device="cpu"); g.manual_seed(seed)
    # sigma trai tu 0.2 .. 3.0 tren log-scale (co ca low & high variance)
    lo, hi = 0.2, 3.0
    ranks = torch.linspace(0, 1, D)
    sigmas = torch.exp(torch.log(torch.tensor(lo)) +
                       ranks * (math.log(hi) - math.log(lo)))       # tang dan
    sigmas = sigmas[torch.randperm(D, generator=g)]                 # xao tron vi tri

    order = torch.argsort(sigmas)                # thap -> cao
    if name == "two-equal":
        lo_idx = order[0].item()                 # phuong sai thap nhat
        hi_idx = order[-1].item()                # phuong sai cao nhat
        driver_idx = torch.tensor([lo_idx, hi_idx])
        # du bao NGANG NHAU: chuan hoa theo sigma de dong gop phuong sai-output bang nhau
        # y ~ w_lo * x_lo + w_hi * x_hi ; muon Var(w x) bang nhau -> w ~ 1/sigma
        w = torch.tensor([1.0 / sigmas[lo_idx], 1.0 / sigmas[hi_idx]])
    elif name == "high-var":
        hi_idx = order[-1].item()
        driver_idx = torch.tensor([hi_idx])
        w = torch.tensor([1.0 / sigmas[hi_idx]])
    elif name == "low-var":
        lo_idx = order[0].item()
        driver_idx = torch.tensor([lo_idx])
        w = torch.tensor([1.0 / sigmas[lo_idx]])
    else:
        raise ValueError(name)

    return Regime(
        name=name, D=D,
        sigmas=sigmas.to(device),
        driver_idx=driver_idx.to(device),
        driver_w=w.to(device),
    )


@torch.no_grad()
def sample_regime(reg: Regime, n: int, device, seed: int):
    """
    X ~ N(0, diag(sigma^2)) doc lap (mac dinh khong tuong quan o E0 — E0 test bias
    phuong-sai, khong phai tuong quan; tuong quan de cho E3).
    y_true = sum_k w_k x_driver_k  (ham TUYEN TINH that, la ground truth driver).
    Tra ve X (n,D), y (n,) — y la gia tri lien tuc (regression target).
    """
    g = torch.Generator(device="cpu"); g.manual_seed(seed)
    Z = torch.randn(n, reg.D, generator=g)
    X = (Z * reg.sigmas.cpu()[None]).to(device)
    y = (X[:, reg.driver_idx] * reg.driver_w[None]).sum(dim=1)      # (n,)
    return X, y


class MLP(nn.Module):
    """MLP kha vi nho — chi de model HOC ham target that roi lay gradient IG."""
    def __init__(self, D, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_model(reg: Regime, device, seed: int, n_train=8000, epochs=300, lr=1e-3):
    """
    Fit MLP -> y_true tren regime. Tra ve model.eval() da .to(device).
    Chi de co model kha vi khop ham that; day KHONG phai benchmark, chi chuan bi E0.
    """
    Xtr, ytr = sample_regime(reg, n_train, device, seed + 1)
    # chuan hoa target de loss on dinh (khong doi driver ranking)
    ymu, ysd = ytr.mean(), ytr.std().clamp_min(1e-6)
    ytr_n = (ytr - ymu) / ysd
    model = MLP(reg.D).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    bs = 512
    for ep in range(epochs):
        perm = torch.randperm(Xtr.shape[0], device=device)
        for i in range(0, Xtr.shape[0], bs):
            idx = perm[i:i + bs]
            pred = model(Xtr[idx])
            loss = F.mse_loss(pred, ytr_n[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    # gan meta chuan hoa de score nhat quan (khong bat buoc dung)
    model._ymu, model._ysd = float(ymu), float(ysd)
    return model


# ===========================================================================
# (2) IG tren tabular (dung CHUNG midpoint rule voi methods.py: ig_single).
#   grad_fn(states (M,D)) -> grads (M,D) qua autograd cua model (scalar output).
# ===========================================================================
def make_tabular_gradfn(model, device, chunk=4096):
    """grad d model(x)/dx cho batch (M,D). Model output scalar-per-row."""
    def grad_fn(states: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(states)
        for i in range(0, states.shape[0], chunk):
            sb = states[i:i + chunk].to(device).clone().requires_grad_(True)
            score = model(sb).sum()
            grad, = torch.autograd.grad(score, sb)
            out[i:i + chunk] = grad.detach()
            del sb, score, grad
        return out
    return grad_fn


def ig_tabular(x, x0, grad_fn, T=64):
    """
    IG midpoint (khop pea.methods.ig_single, chi khac shape (D,)):
      phi = (x - x0) * mean_t grad(x0 + t(x-x0)).  x,x0: (D,). Tra ve (D,).
    """
    device = x.device
    alphas = ((torch.arange(T, device=device) + 0.5) / T).view(-1, 1)   # (T,1) midpoint
    states = x0[None] + alphas * (x - x0)[None]                          # (T,D)
    g = grad_fn(states)                                                  # (T,D)
    return g.mean(dim=0) * (x - x0)                                      # (D,)


def eg_tabular(x, baselines, grad_fn, N):
    """EG: trung binh IG tren pool baseline (baselines: (K,D)), ngan sach chia deu."""
    K = baselines.shape[0]
    T = max(1, N // K)
    acc = torch.zeros_like(x)
    for k in range(K):
        acc += ig_tabular(x, baselines[k], grad_fn, T=T)
    return acc / K


# ===========================================================================
# (3) E0 metrics: attribution co phuc hoi driver that khong?
#   Ground truth: mask nhi phan (feature j la driver hay khong).
#   Cham diem tam quan feature = |attribution| gom |.| tren cac chieu (o tabular
#   moi feature 1 chieu, nen chi la |phi_j|). AUPRC & rank-corr(|phi|, mask).
# ===========================================================================
@torch.no_grad()
def _auprc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """AUPRC (average precision) cho diem scores vs nhan nhi phan labels (1=driver)."""
    order = torch.argsort(scores, descending=True)
    lab = labels[order].float()
    tp = torch.cumsum(lab, dim=0)
    precision = tp / torch.arange(1, len(lab) + 1, device=lab.device).float()
    recall = tp / lab.sum().clamp_min(1)
    # average precision = sum (R_k - R_{k-1}) * P_k
    recall_prev = torch.cat([torch.zeros(1, device=lab.device), recall[:-1]])
    ap = ((recall - recall_prev) * precision).sum().item()
    return ap


@torch.no_grad()
def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Spearman rank-corr giua hai vector."""
    ra = torch.argsort(torch.argsort(a)).float()
    rb = torch.argsort(torch.argsort(b)).float()
    ra = ra - ra.mean(); rb = rb - rb.mean()
    denom = (ra.norm() * rb.norm()).clamp_min(1e-12)
    return float((ra @ rb) / denom)


@torch.no_grad()
def e0_scores(phi_matrix: torch.Tensor, driver_mask: torch.Tensor):
    """
    phi_matrix : (n_eval, D) attribution moi mau.
    driver_mask: (D,) 1 neu feature la driver that.
    Cham tren tam quan TRUNG BINH |phi| qua mau (importance toan cuc theo E0).
    Tra ve dict(auprc, spearman, per_feature_importance).
    """
    imp = phi_matrix.abs().mean(dim=0)                     # (D,) importance
    auprc = _auprc(imp, driver_mask)
    spear = _spearman(imp, driver_mask.float())
    return {"auprc": auprc, "spearman": spear, "importance": imp.cpu()}


# ===========================================================================
# (4) E1: conditional-imputation insertion/deletion (KHONG tai dung baseline).
#   Imputer: Gaussian conditional mean fit tren SPLIT RIENG (draft: disjoint imputer).
#   Xoa feature = thay bang E[x_j | x_{-j}] uoc luong tu (mu_imp, Sigma_imp).
# ===========================================================================
@dataclass
class GaussImputer:
    mu: torch.Tensor          # (D,)
    Sigma: torch.Tensor       # (D,D)

    @torch.no_grad()
    def impute(self, x: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
        """
        Dat feature NGOAI keep_mask ve E[x_miss | x_keep] (conditional Gaussian mean).
        x: (D,). keep_mask: (D,) bool (True=giu gia tri that). Tra ve (D,).
        """
        D = x.shape[0]
        keep = keep_mask
        miss = ~keep
        if miss.sum() == 0:
            return x.clone()
        if keep.sum() == 0:
            out = x.clone(); out[miss] = self.mu[miss]; return out
        mu_m = self.mu[miss]; mu_k = self.mu[keep]
        S_mk = self.Sigma[miss][:, keep]
        S_kk = self.Sigma[keep][:, keep]
        S_kk = S_kk + 1e-6 * torch.eye(S_kk.shape[0], device=x.device, dtype=x.dtype)
        sol = torch.linalg.solve(S_kk, (x[keep] - mu_k))
        cond = mu_m + S_mk @ sol
        out = x.clone(); out[miss] = cond
        return out


@torch.no_grad()
def fit_imputer(X: torch.Tensor) -> GaussImputer:
    mu = X.mean(dim=0)
    Xc = X - mu[None]
    S = (Xc.transpose(0, 1) @ Xc) / max(1, X.shape[0] - 1)
    S = 0.5 * (S + S.transpose(0, 1)) + 1e-6 * torch.eye(X.shape[1], device=X.device, dtype=X.dtype)
    return GaussImputer(mu=mu, Sigma=S)


@torch.no_grad()
def insertion_deletion_tabular(model, x, phi, imputer: GaussImputer, steps=None):
    """
    Conditional-imputation insertion/deletion tren 1 mau (tabular).
    - order feature theo |phi| giam dan.
    - insertion: bat dau tat ca feature bi impute (conditional mean voi keep rong),
      lan luot LO ra feature quan trong nhat (dat lai gia tri that), do model output.
    - deletion : bat dau tu x day du, lan luot XOA feature quan trong nhat (impute).
    AUC = trung binh output tren cac buoc. I-D gap = ins_auc - del_auc.
    Output dung score model tho (nhat quan voi attribution backward).
    """
    D = x.shape[0]
    steps = steps or D
    order = torch.argsort(phi.abs(), descending=True)      # (D,)
    device = x.device

    ks = torch.linspace(0, D, steps + 1, device=device).round().long()

    # INSERTION: keep = top-k feature quan trong; con lai impute
    ins_imgs = []
    for k in ks:
        keep = torch.zeros(D, dtype=torch.bool, device=device)
        keep[order[:k]] = True
        ins_imgs.append(imputer.impute(x, keep))
    ins = model(torch.stack(ins_imgs))                     # (steps+1,)

    # DELETION: xoa dan top-k quan trong (keep = phan con lai)
    del_imgs = []
    for k in ks:
        keep = torch.ones(D, dtype=torch.bool, device=device)
        keep[order[:k]] = False
        del_imgs.append(imputer.impute(x, keep))
    dele = model(torch.stack(del_imgs))                    # (steps+1,)

    return {"insertion_auc": ins.mean().item(),
            "deletion_auc": dele.mean().item(),
            "id_gap": (ins.mean() - dele.mean()).item()}


# ===========================================================================
# (5) Runners
# ===========================================================================
def run_e0(reg: Regime, model, device, args):
    """
    E0 tren 1 regime: quet tau, cham AUPRC/rank-corr(attribution vs driver that).
    Ket luan go/no-go: neu tau nho (gan x) van over-credit feature phuong-sai-thap
    BAT KE regime => deviation-from-prior; neu recover dung driver theo regime => OK.
    """
    print(f"\n{'='*70}\nE0  regime = {reg.name}   D={reg.D}")
    drv = reg.driver_idx.tolist()
    print(f"     driver feature idx = {drv}   sigma(driver) = "
          f"{[round(reg.sigmas[i].item(),3) for i in drv]}   "
          f"w = {[round(w,3) for w in reg.driver_w.tolist()]}")

    # tap tham chieu (uoc luong mu,Sigma) va tap danh gia — split RIENG
    X_ref, _ = sample_regime(reg, args.n_ref, device, args.seed + 100)
    X_eval, _ = sample_regime(reg, args.n_eval, device, args.seed + 200)
    ref = fit_reference(X_ref, floor=args.floor)
    grad_fn = make_tabular_gradfn(model, device, chunk=args.chunk)

    driver_mask = torch.zeros(reg.D, dtype=torch.long, device=device)
    driver_mask[reg.driver_idx] = 1

    print(f"\n{'tau':>10}{'AUPRC↑':>10}{'rank-corr↑':>12}{'top-feat':>10}"
          f"{'  (driver rank in importance)'}")
    print("-" * 70)
    results = {}
    for tau in args.tau_sweep:
        phis = []
        for i in range(X_eval.shape[0]):
            x = X_eval[i]
            b = shrinkage_baseline(x, ref, tau=tau)
            phis.append(ig_tabular(x, b, grad_fn, T=args.T))
        phi_mat = torch.stack(phis)                        # (n_eval, D)
        sc = e0_scores(phi_mat, driver_mask)
        imp = sc["importance"].to(device)
        top = int(torch.argmax(imp).item())
        # thu hang cua driver that trong bang importance (0 = duoc credit cao nhat)
        imp_order = torch.argsort(imp, descending=True)
        drv_ranks = [int((imp_order == d).nonzero().item()) for d in reg.driver_idx.tolist()]
        results[tau] = {"auprc": sc["auprc"], "spearman": sc["spearman"],
                        "top": top, "driver_ranks": drv_ranks}
        print(f"{tau:>10.3g}{sc['auprc']:>10.4f}{sc['spearman']:>12.4f}"
              f"{top:>10}{'   ' + str(drv_ranks)}")

    # chan doan go/no-go
    print("-" * 70)
    best_tau = max(results, key=lambda t: results[t]["auprc"])
    print(f"[i] best AUPRC @ tau={best_tau:.3g} : {results[best_tau]['auprc']:.4f}")
    # over-credit test: driver co bi day khoi top-k khi la high-var?
    if reg.name == "high-var":
        ranks_small_tau = results[min(args.tau_sweep)]["driver_ranks"]
        verdict = ("OVER-CREDIT low-var (driver bi tut hang du la high-var predictive)"
                   if max(ranks_small_tau) >= reg.D // 2
                   else "recover high-var driver OK")
        print(f"[go/no-go] high-var regime -> {verdict}")
    return results


def run_e1(reg: Regime, model, device, args):
    """
    E1 baseline comparison tren tabular (theo compare.py):
      zero, mean, median, random-sample, EG(K in {1,4,16,64}),
      Shrinkage-IG(swept tau), PM-IG-PPCA(psi).
    Metric: conditional-imputation insertion/deletion (imputer split RIENG) + I-D gap,
    trung binh tren tap eval + bootstrap CI.
    """
    print(f"\n{'='*70}\nE1  baseline comparison   regime = {reg.name}   D={reg.D}")

    X_ref, _ = sample_regime(reg, args.n_ref, device, args.seed + 100)
    X_eval, _ = sample_regime(reg, args.n_eval_e1, device, args.seed + 300)
    X_imp, _ = sample_regime(reg, args.n_ref, device, args.seed + 400)   # imputer split RIENG
    ref = fit_reference(X_ref, floor=args.floor)
    ppca_ref, psi = fit_ppca(X_ref, q=args.ppca_q)
    imputer = fit_imputer(X_imp)
    grad_fn = make_tabular_gradfn(model, device, chunk=args.chunk)

    mu = X_ref.mean(dim=0)
    med = X_ref.median(dim=0).values
    zero = torch.zeros(reg.D, device=device)
    g = torch.Generator(device="cpu"); g.manual_seed(args.seed + 5)
    rand_pool = X_ref[torch.randperm(X_ref.shape[0], generator=g)[:64]]   # pool random-sample

    N = args.N_e1                                           # ngan sach grad/mau

    def attr_for(name, x):
        if name == "IG-zero":
            return ig_tabular(x, zero, grad_fn, T=N)
        if name == "IG-mean":
            return ig_tabular(x, mu, grad_fn, T=N)
        if name == "IG-median":
            return ig_tabular(x, med, grad_fn, T=N)
        if name == "IG-random":
            return ig_tabular(x, rand_pool[0], grad_fn, T=N)
        if name.startswith("EG-"):
            K = int(name.split("-")[1])
            return eg_tabular(x, rand_pool[:K], grad_fn, N=N)
        if name.startswith("Shrinkage-"):
            tau = float(name.split("@")[1])
            b = shrinkage_baseline(x, ref, tau=tau)
            return ig_tabular(x, b, grad_fn, T=N)
        if name == "PM-IG-PPCA":
            b = shrinkage_baseline(x, ppca_ref, tau=psi)
            return ig_tabular(x, b, grad_fn, T=N)
        raise ValueError(name)

    methods = ["IG-zero", "IG-mean", "IG-median", "IG-random",
               "EG-1", "EG-4", "EG-16", "EG-64"]
    methods += [f"Shrinkage-IG@{t:g}" for t in args.tau_sweep]
    methods += ["PM-IG-PPCA"]
    print(f"[i] PM-IG-PPCA psi(estimated) = {psi:.4f}   (rank q={min(args.ppca_q, reg.D-1)})")
    print(f"[i] ngan sach N={N} grad/mau, n_eval={X_eval.shape[0]}, "
          f"imputer split RIENG (n={X_imp.shape[0]})\n")

    print(f"{'method':<20}{'insertion↑':>12}{'deletion↓':>12}{'I-D↑':>10}"
          f"{'  (mean ± boot-SE)'}")
    print("-" * 66)
    table = {}
    for nm in methods:
        gaps, ins, dels = [], [], []
        for i in range(X_eval.shape[0]):
            x = X_eval[i]
            phi = attr_for(nm, x)
            r = insertion_deletion_tabular(model, x, phi, imputer, steps=args.insdel_steps)
            ins.append(r["insertion_auc"]); dels.append(r["deletion_auc"]); gaps.append(r["id_gap"])
        ins_t = torch.tensor(ins); del_t = torch.tensor(dels); gap_t = torch.tensor(gaps)
        # bootstrap SE cua I-D gap
        se = _bootstrap_se(gap_t, n_boot=args.n_boot, seed=args.seed)
        table[nm] = {"insertion_auc": ins_t.mean().item(),
                     "deletion_auc": del_t.mean().item(),
                     "id_gap": gap_t.mean().item(), "id_se": se}
        print(f"{nm:<20}{ins_t.mean().item():>12.4f}{del_t.mean().item():>12.4f}"
              f"{gap_t.mean().item():>10.4f}   ± {se:.4f}")

    print("-" * 66)
    best = max(table, key=lambda k: table[k]["id_gap"])
    print(f"[i] best I-D gap: {best} = {table[best]['id_gap']:.4f} ± {table[best]['id_se']:.4f}")
    return table


@torch.no_grad()
def _bootstrap_se(v: torch.Tensor, n_boot=1000, seed=0) -> float:
    g = torch.Generator(device="cpu"); g.manual_seed(seed + 999)
    n = v.numel()
    if n <= 1:
        return 0.0
    means = torch.empty(n_boot)
    for b in range(n_boot):
        idx = torch.randint(0, n, (n,), generator=g)
        means[b] = v[idx].mean()
    return float(means.std().item())


# ===========================================================================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default="all", choices=["all", "e0", "e1"])
    ap.add_argument("--regime", type=str, default="all",
                    choices=["all", "two-equal", "high-var", "low-var"])
    ap.add_argument("--D", type=int, default=20, help="so feature tabular")
    ap.add_argument("--tau_sweep", type=float, nargs="+",
                    default=[0.01, 0.1, 1.0, 10.0, 100.0], help="dai tau quet")
    ap.add_argument("--T", type=int, default=64, help="so buoc IG midpoint (E0)")
    ap.add_argument("--N_e1", type=int, default=64, help="ngan sach grad/mau cho E1")
    ap.add_argument("--n_ref", type=int, default=6000, help="tap uoc luong mu,Sigma / imputer")
    ap.add_argument("--n_eval", type=int, default=400, help="so mau danh gia E0")
    ap.add_argument("--n_eval_e1", type=int, default=200, help="so mau danh gia E1")
    ap.add_argument("--floor", type=float, default=1e-6, help="ridge floor lambda cho Sigma")
    ap.add_argument("--ppca_q", type=int, default=5, help="rank q cho PM-IG-PPCA")
    ap.add_argument("--insdel_steps", type=int, default=None, help="so buoc ins/del (mac dinh D)")
    ap.add_argument("--n_boot", type=int, default=1000, help="so lan bootstrap CI")
    ap.add_argument("--epochs", type=int, default=300, help="epoch fit MLP")
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[!] cuda khong san sang -> cpu"); device = "cpu"
    torch.manual_seed(args.seed)

    regimes = (["two-equal", "high-var", "low-var"] if args.regime == "all"
               else [args.regime])

    for rn in regimes:
        reg = make_regime(rn, args.D, device, args.seed)
        model = train_model(reg, device, args.seed, epochs=args.epochs)
        # bao cao chat luong fit de biet gradient co y nghia
        with torch.no_grad():
            Xte, yte = sample_regime(reg, 2000, device, args.seed + 777)
            pred = model(Xte)
            yn = (yte - torch.tensor(model._ymu, device=device)) / torch.tensor(model._ysd, device=device)
            r2 = 1 - F.mse_loss(pred, yn).item() / yn.var().item()
        print(f"\n[model fit] regime={rn}  R^2(test)={r2:.4f}")

        if args.only in ("all", "e0"):
            run_e0(reg, model, device, args)
        if args.only in ("all", "e1"):
            run_e1(reg, model, device, args)


if __name__ == "__main__":
    main()