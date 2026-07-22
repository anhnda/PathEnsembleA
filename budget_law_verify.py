"""
budget_law_verify.py — KIEM LUAT PHAN BO NGAN SACH GRADIENT.

Luat (dan xuat giai tich):
    E_EG(K) ≈ Sigma_base / K  +  a * C_path^2 * K^4 / N^4
    => K* ∝ ( Sigma_base * N^4 / C_path^2 )^(1/5)

    C_path LON  (path cong)      => K*->1 => RESOLUTION (+ shrinkage)   [ky vong: image]
    C_path NHO, Sigma_base LON   => K* lon => DIVERSITY (EG)            [ky vong: tabular/nlp]

MUC TIEU: do C_path va Sigma_base DOC LAP voi diem faithfulness, roi kiem xem
K* co DU DOAN dung method thang (EG vs shrinkage) o moi modality khong.

TRUNG THUC / CHONG VONG LAP:
  - C_path, Sigma_base = tu HINH HOC model (gradient doc path, spread attribution qua
    baseline). KHONG dung insertion/deletion.
  - winner = tu insertion/deletion (marginal removal).
  - Neu hai cai nay TU hai nguon khac nhau ma K* van doan dung => co tin.
  - Neu C_path KHONG tach duoc image khoi tabular => luat SAI, in [FALSIFIED].

Chay tren toy tabular (chay duoc o day). Cau truc de cam sang image/nlp model that.
Torch.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0); np.random.seed(0)
DEV = "cpu"


# ------------------------------------------------------------------ model
class MLP(nn.Module):
    def __init__(self, d, k, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(), nn.Linear(h, k))
    def forward(self, x): return self.net(x)


def train_mlp(X, y, d, k, epochs=300, lr=1e-2):
    m = MLP(d, k); opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4)
    lf = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad(); lf(m(X), y).backward(); opt.step()
    return m.double()


# ------------------------------------------------------------------ reference
def fit_ref(X, lam=1e-3):
    mu = X.mean(0); Xc = X - mu
    S = (Xc.T @ Xc) / (X.shape[0]-1); S = 0.5*(S+S.T) + lam*torch.eye(X.shape[1], dtype=X.dtype)
    return mu, S

def shrink(x, mu, S, tau):
    d = S.shape[0]; W = S @ torch.linalg.inv(S + tau*torch.eye(d, dtype=S.dtype))
    return mu + (x - mu) @ W.T


# ------------------------------------------------------------------ IG core
def prob(model, x, target):
    return torch.softmax(model(x[None])[0], 0)[target]

def ig(model, x, b, target, steps):
    a = ((torch.arange(steps, dtype=x.dtype)+0.5)/steps).view(-1,1)
    path = b[None] + a*(x-b)[None]; path.requires_grad_(True)
    out = model(path)[:, target]
    g = torch.autograd.grad(out.sum(), path)[0]
    return (x-b)*g.mean(0)


# ============================================================ QUANTITY 1: C_path
# Path curvature = do cong cua F doc doan b->x. Do DOC LAP voi faithfulness:
# lay ||IG_2step - IG_highres|| chuan hoa — bias luong tu hoa ~ C_path/m^2, nen
# hieu giua 2-step va 256-step la proxy truc tiep cho C_path.
def c_path_proxy(model, x, b, target, hi=256):
    ig2  = ig(model, x, b, target, 2)
    ighi = ig(model, x, b, target, hi)
    denom = ighi.norm().clamp_min(1e-8)
    return (ig2 - ighi).norm().item() / denom.item()   # relative curvature-induced error


# ============================================================ QUANTITY 2: Sigma_base
# Spread cua attribution qua cac baseline p_ref. DOC LAP voi faithfulness:
# lay K_probe baseline that, tinh IG(highres) tu moi cai, do phuong sai quanh trung binh.
def sigma_base(model, x, pool, target, K_probe=32, hi=64):
    idx = torch.randperm(len(pool))[:K_probe]
    igs = torch.stack([ig(model, x, pool[i], target, hi) for i in idx])  # (K,d)
    mu_ig = igs.mean(0)
    return (igs - mu_ig[None]).pow(2).sum(1).mean().item()   # E||IG(b)-mean||^2


# ============================================================ winner: EG vs shrinkage at fixed N
def insdel(model, x, attr, target, pool, n_draw=6):
    d = x.numel(); order = torch.argsort(attr.abs(), descending=True)
    _trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    ins_a, del_a = [], []
    g = torch.Generator().manual_seed(0)
    for _ in range(n_draw):
        fill = pool[torch.randint(len(pool),(1,),generator=g)][0]
        cur = fill.clone(); ins=[prob(model,cur,target).item()]
        for j in order: cur[j]=x[j]; ins.append(prob(model,cur,target).item())
        cur = x.clone(); de=[prob(model,cur,target).item()]
        for j in order: cur[j]=fill[j]; de.append(prob(model,cur,target).item())
        ins_a.append(_trap(ins)/(len(ins)-1)); del_a.append(_trap(de)/(len(de)-1))
    return float(np.mean(ins_a)-np.mean(del_a))

def eg_attr(model, x, pool, target, N, K):
    m = max(2, N//K); idx = torch.randperm(len(pool))[:K]
    return torch.stack([ig(model,x,pool[i],target,m) for i in idx]).mean(0)


def run(name, loader, N=64, n_eval=15):
    from sklearn.model_selection import train_test_split
    data = loader()
    X = torch.tensor(data.data, dtype=torch.float64); y = torch.tensor(data.target)
    X = (X - X.mean(0))/X.std(0).clamp_min(1e-8)
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=.3,random_state=0,stratify=y)
    d,k = X.shape[1], int(y.max())+1
    model = train_mlp(Xtr.float(), ytr, d, k)
    mu,S = fit_ref(Xtr); pool = Xtr
    tgt = int(torch.bincount(yte).argmax())
    idx = (yte==tgt).nonzero().flatten()[:n_eval]

    Cs, Sb = [], []
    eg_scores = {1:[],4:[],16:[]}; shr_scores=[]
    for i in idx:
        x = Xte[i]
        b_sh = shrink(x, mu, S, tau=1.0)                     # a representative shrinkage baseline
        Cs.append(c_path_proxy(model, x, b_sh, tgt))
        Sb.append(sigma_base(model, x, pool, tgt))
        # winner contest at fixed N
        shr_scores.append(insdel(model, x, ig(model,x,b_sh,tgt,N), tgt, pool))
        for K in eg_scores:
            eg_scores[K].append(insdel(model, x, eg_attr(model,x,pool,tgt,N,K), tgt, pool))

    C = float(np.mean(Cs)); Sig = float(np.mean(Sb))
    shr = float(np.mean(shr_scores))
    egbest_K = max(eg_scores, key=lambda K: np.mean(eg_scores[K]))
    egbest = float(np.mean(eg_scores[egbest_K]))
    # predicted K* (relative, up to constant a): (Sig * N^4 / C^2)^(1/5)
    Kstar = (Sig * N**4 / max(C,1e-8)**2) ** 0.2
    winner = "EG" if egbest > shr else "shrinkage"
    pred = "EG" if Kstar > 4 else "shrinkage"   # Kstar>~few => diversity; ~1 => resolution
    return dict(name=name, C_path=C, Sigma_base=Sig, Kstar=Kstar,
                shr=shr, eg=egbest, eg_K=egbest_K, winner=winner, pred=pred)


def main():
    from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
    jobs = [("iris",load_iris),("wine",load_wine),
            ("bcancer",load_breast_cancer),("digits",load_digits)]
    rows = []
    for nm, ld in jobs:
        print(f"[.] {nm} ...", flush=True)
        rows.append(run(nm, ld))

    print(f"\n{'dataset':<9}{'C_path':>9}{'Sig_base':>10}{'K*':>8}"
          f"{'shr':>8}{'EG':>8}{'EGk':>5}{'winner':>11}{'K*pred':>9}{'ok':>5}")
    print("-"*82)
    ok_all = True
    for r in rows:
        ok = (r['winner']==r['pred']); ok_all &= ok
        print(f"{r['name']:<9}{r['C_path']:>9.3f}{r['Sigma_base']:>10.4f}{r['Kstar']:>8.2f}"
              f"{r['shr']:>8.3f}{r['eg']:>8.3f}{r['eg_K']:>5}{r['winner']:>11}{r['pred']:>9}"
              f"{'  Y' if ok else '  N':>5}")
    print("-"*82)
    # key falsification: does C_path SEPARATE at all? does K* predict winner?
    Cs = [r['C_path'] for r in rows]
    print(f"[i] C_path range across datasets: [{min(Cs):.3f}, {max(Cs):.3f}]  "
          f"spread={max(Cs)-min(Cs):.3f}")
    if max(Cs)-min(Cs) < 0.05:
        print("[!!] C_path KHONG tach duoc cac dataset => khong the la truc phan biet. LUAT NGHI NGO.")
    if ok_all:
        print("[OK] K* du doan dung winner o MOI dataset (toy). Dang khich le — kiem tiep tren")
        print("     image (C_path ky vong CAO) + nlp de co diem doi cuc.")
    else:
        print("[!!] K* SAI o it nhat 1 dataset => cong thuc phan bo ngan sach CHUA dung nhu la.")
        print("[!!]  Co the do: (a) toy data deu low-curvature nen khong co diem tuong phan,")
        print("[!!]  (b) nguong K*>4 sai, hoac (c) luat thieu so hang. Xem cot C_path/K*.")
    print("\n[i] LUU Y: toy tabular co the DEU low-C_path => khong doi cuc duoc. Diem quyet dinh")
    print("[i]   la IMAGE (CNN, path cong manh). Cam ham nay vao e1_batch_image de do C_path that.")


if __name__ == "__main__":
    main()
