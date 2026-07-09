"""
E1 tabular — DOC LAP HOAN TOAN. Khong import synthetic_e0.
Tu chua: model, IG, shrinkage baseline, insertion/deletion (da SUA).

Diem sua so voi ban cu:
  1. AUC bo diem k=0 (chua can thiep) -> khong con bi keo len 1.0 gia tao.
  2. In thang logit/softmax cua input all-zero va all-mean -> thay tan mat
     model tra gi khi "xoa het". Neu prob giu nguyen ~1 la BANG CHUNG OOD-artifact.
  3. Them che do xoa 'marginal' (boc gia tri tu phan bo cot) -> pha thong tin
     ma van in-distribution, khong nhay ra goc hop.

torch GPU mac dinh (device theo --device). Ban tu chay. KHONG smoketest.
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

DATASETS = {
    "breast_cancer": datasets.load_breast_cancer,
    "wine": datasets.load_wine,
    "digits": datasets.load_digits,
}


# --------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, D, hidden=128, n_out=2):
        super().__init__()
        self.n_out = n_out
        self.net = nn.Sequential(
            nn.Linear(D, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, n_out),
        )
    def forward(self, x):
        return self.net(x)


def score_target(model, x, target, score="softmax"):
    single = (x.dim() == 1)
    out = model(x[None] if single else x)
    s = out[:, target] if score == "logit" else F.softmax(out, 1)[:, target]
    return s[0] if single else s


# --------------------------------------------------------------------------
# Shrinkage baseline: b_tau(x) = mu + V diag(s/(s+tau)) V^T (x - mu)
# --------------------------------------------------------------------------
@torch.no_grad()
def fit_reference(X, floor=1e-6):
    mu = X.mean(0)
    Xc = X - mu[None]
    S = (Xc.T @ Xc) / max(1, X.shape[0] - 1)
    S = 0.5 * (S + S.T) + floor * torch.eye(X.shape[1], device=X.device)
    s, V = torch.linalg.eigh(S)
    order = torch.argsort(s, descending=True)
    return mu, V[:, order].contiguous(), s[order].contiguous()


@torch.no_grad()
def shrinkage_baseline(x, ref, tau):
    mu, V, s = ref
    d = (x - mu)
    coeff = d @ V
    g = s / (s + tau)
    return mu + (coeff * g) @ V.T


# --------------------------------------------------------------------------
# IG (midpoint)
# --------------------------------------------------------------------------
def make_gradfn(model, target, score, device, chunk=4096):
    def grad_fn(states):
        out = torch.empty_like(states)
        for i in range(0, states.shape[0], chunk):
            sb = states[i:i+chunk].to(device).clone().requires_grad_(True)
            s = score_target(model, sb, target, score).sum()
            g, = torch.autograd.grad(s, sb)
            out[i:i+chunk] = g.detach()
        return out
    return grad_fn


def ig(x, x0, grad_fn, T=64):
    a = ((torch.arange(T, device=x.device) + 0.5) / T).view(-1, 1)
    states = x0[None] + a * (x - x0)[None]
    g = grad_fn(states)
    return g.mean(0) * (x - x0)


# --------------------------------------------------------------------------
# Insertion / deletion — SUA: bo k=0, them mode 'marginal'
# --------------------------------------------------------------------------
@torch.no_grad()
def ins_del(model, x, phi, target, score, baseline_vec, mode, Xtr, steps=None, gen=None):
    D = x.shape[0]
    steps = steps or D
    order = torch.argsort(phi.abs(), descending=True)
    dev = x.device

    def fill(keep_mask):
        if mode == "marginal":
            out = x.clone()
            miss = ~keep_mask
            idx = torch.where(miss)[0]
            for j in idx.tolist():
                r = torch.randint(0, Xtr.shape[0], (1,), generator=gen).item()
                out[j] = Xtr[r, j]
            return out
        return torch.where(keep_mask, x, baseline_vec)

    ks = torch.linspace(0, D, steps + 1, device=dev).round().long()

    ins_imgs, del_imgs = [], []
    for k in ks:
        ki = int(k)
        keep_ins = torch.zeros(D, dtype=torch.bool, device=dev); keep_ins[order[:ki]] = True
        keep_del = torch.ones(D, dtype=torch.bool, device=dev);  keep_del[order[:ki]] = False
        ins_imgs.append(fill(keep_ins)); del_imgs.append(fill(keep_del))
    ins = score_target(model, torch.stack(ins_imgs), target, score)
    dele = score_target(model, torch.stack(del_imgs), target, score)

    # SUA: AUC bo diem k=0 (chua can thiep) o CA HAI dau
    ins_auc = ins[1:].mean().item()
    del_auc = dele[1:].mean().item()
    return {"insertion_auc": ins_auc, "deletion_auc": del_auc,
            "id_gap": ins_auc - del_auc,
            "del_curve": dele.tolist()}   # tra ca duong de nhin dinh/day


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="breast_cancer", choices=list(DATASETS))
    ap.add_argument("--scaler", default="minmax", choices=["minmax", "standard"])
    ap.add_argument("--del_mode", default="marginal", choices=["zero", "marginal", "conditional"])
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--tau_sweep", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0, 100.0])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--score", default="softmax", choices=["logit", "softmax"])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[!] cuda khong san sang -> cpu"); device = "cpu"
    torch.manual_seed(args.seed)

    d = DATASETS[args.dataset]()
    X, y = d.data, d.target
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=args.seed, stratify=y)
    Scaler = MinMaxScaler if args.scaler == "minmax" else StandardScaler
    sc = Scaler().fit(Xtr)
    Xtr = torch.tensor(sc.transform(Xtr), dtype=torch.float32, device=device)
    Xte = torch.tensor(sc.transform(Xte), dtype=torch.float32, device=device)
    ytr = torch.tensor(ytr, dtype=torch.long, device=device)
    yte = torch.tensor(yte, dtype=torch.long, device=device)
    D = Xtr.shape[1]; n_class = int(y.max()) + 1
    target = 1 if n_class == 2 else int(torch.bincount(ytr).argmax())

    model = MLP(D, n_out=n_class).to(device)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    model.train()
    for ep in range(args.epochs):
        p = torch.randperm(Xtr.shape[0], device=device)
        for i in range(0, Xtr.shape[0], 256):
            idx = p[i:i+256]
            loss = F.cross_entropy(model(Xtr[idx]), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()

    with torch.no_grad():
        acc = (model(Xte).argmax(1) == yte).float().mean().item()
    print(f"[i] dataset={args.dataset} scaler={args.scaler} D={D} target={target} test_acc={acc:.4f}")

    # ---- SANITY: model tra gi khi input = all-zero / all-mean / all-min / all-max ----
    with torch.no_grad():
        mu_in = Xtr.mean(0, keepdim=True)
        probes = {
            "all-zero": torch.zeros(1, D, device=device),
            "all-mean": mu_in,
            "all-min":  Xtr.min(0).values[None],
            "all-max":  Xtr.max(0).values[None],
        }
        print("\n[SANITY] softmax(target) tren input dong nhat (xoa het feature = 1 trong cac diem nay):")
        for k, v in probes.items():
            sm = F.softmax(model(v), 1)[0]
            print(f"   {k:<10} softmax={sm.cpu().numpy().round(4)}  -> p(target={target})={sm[target].item():.4f}")
        print("   (Neu p(target) o 'all-zero'/'all-mean' ~ giong sample that => 'xoa het' KHONG ha prediction => metric vo nghia)\n")

    ref = fit_reference(Xtr)
    grad_fn = make_gradfn(model, target, args.score, device)
    mu = Xtr.mean(0); med = Xtr.median(0).values; zero = torch.zeros(D, device=device)
    gen = torch.Generator(device="cpu"); gen.manual_seed(args.seed)

    with torch.no_grad():
        is_tgt = model(Xte).argmax(1) == target
    Xev = Xte[is_tgt]
    if args.limit: Xev = Xev[:args.limit]
    print(f"[i] danh gia {Xev.shape[0]} mau lop target, del_mode={args.del_mode}\n")

    baseline_vec = zero  # cho mode 'zero'
    methods = {
        "IG-zero":   lambda x: ig(x, zero, grad_fn, args.N),
        "IG-mean":   lambda x: ig(x, mu,   grad_fn, args.N),
        "IG-median": lambda x: ig(x, med,  grad_fn, args.N),
    }
    for t in args.tau_sweep:
        methods[f"Shrinkage@{t:g}"] = (lambda x, t=t: ig(x, shrinkage_baseline(x, ref, t), grad_fn, args.N))

    print(f"{'method':<18}{'insertion↑':>12}{'deletion↓':>12}{'I-D↑':>10}")
    print("-" * 52)
    results = {}
    for nm, fn in methods.items():
        ins, dele, gap = [], [], []
        for i in range(Xev.shape[0]):
            x = Xev[i]; phi = fn(x)
            r = ins_del(model, x, phi, target, args.score, baseline_vec,
                        args.del_mode, Xtr, gen=gen)
            ins.append(r["insertion_auc"]); dele.append(r["deletion_auc"]); gap.append(r["id_gap"])
        mi = sum(ins)/len(ins); mdl = sum(dele)/len(dele); mg = sum(gap)/len(gap)
        results[nm] = mg
        print(f"{nm:<18}{mi:>12.4f}{mdl:>12.4f}{mg:>10.4f}")
    print("-" * 52)
    best = max(results, key=results.get)
    print(f"[i] best I-D: {best} = {results[best]:.4f}")


if __name__ == "__main__":
    main()