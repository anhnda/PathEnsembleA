"""
E1 (baseline comparison) tren REAL TABULAR — sklearn standard dataset.

LUU Y (draft + kinh nghiem): tabular don gian thuong TUYEN TINH, khong co interaction
ro, nen tau->0 (baseline = x) da gan toi uu — Shrinkage KHONG tach manh o day. File nay
la SANITY-CHECK modality de bang E1 day du ca ba, KHONG phai trong tam. Trong tam la
CV (e1_batch_image.py, FFT) va NLP (embedding, lam sau).

Chi giu IG + BASELINE (path thang, dung E1):
    IG-zero / IG-mean / IG-median / IG-random
    EG-K  (pool K random-sample, ngan sach chia deu)               K in {1,4,16,64}
    Shrinkage-IG@tau  (covariance-PCA, quet tau)                   Eq. closed-form
    PM-IG-PPCA        (psi uoc luong, Cor. MMSE)

Dung LAI estimator/IG/metric tu synthetic_e0.py (khong lap code):
    fit_reference, shrinkage_baseline, fit_ppca, ig_tabular, eg_tabular,
    fit_imputer, insertion_deletion_tabular, score_target, make_tabular_gradfn, MLP-train.

Dataset: sklearn offline (breast_cancer / wine / digits). Nhan da co san -> classif,
softmax AUC in [0,1] (khop insdel.py). Metric: conditional-imputation insertion/deletion,
imputer fit tren SPLIT RIENG (train), danh gia tren test. Paired test Shrinkage vs baseline.

Chay (torch GPU mac dinh, tu chay lay):
    python e1_batch_tabular.py --dataset breast_cancer --N 64
    python e1_batch_tabular.py --dataset wine --tau_sweep 0.1 1 10 100
    python e1_batch_tabular.py --dataset digits --target_class 3 --limit 100

KHONG smoketest.
"""

import argparse
import csv
import math
import torch
import torch.nn.functional as F

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# dung lai toan bo may moc tu synthetic_e0.py
from synthetic_e0 import (
    fit_reference, shrinkage_baseline, fit_ppca,
    ig_tabular, eg_tabular, MLP, score_target,
    make_tabular_gradfn, fit_imputer, insertion_deletion_tabular,
    soft_faith_tabular, _bootstrap_se,
)


DATASETS = {
    "breast_cancer": datasets.load_breast_cancer,
    "wine": datasets.load_wine,
    "digits": datasets.load_digits,
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="breast_cancer", choices=list(DATASETS))
    ap.add_argument("--target_class", type=int, default=None,
                    help="lop can giai thich; mac dinh = lop 1 (nhi phan) hoac lop pho bien nhat")
    ap.add_argument("--N", type=int, default=64, help="ngan sach grad/mau")
    ap.add_argument("--tau_sweep", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0, 100.0])
    ap.add_argument("--eg_K", type=int, nargs="+", default=[1, 4, 16, 64])
    ap.add_argument("--ppca_q", type=int, default=5, help="rank q cho PM-IG-PPCA")
    ap.add_argument("--floor", type=float, default=1e-6, help="ridge floor lambda cho Sigma")
    ap.add_argument("--test_size", type=float, default=0.3)
    ap.add_argument("--limit", type=int, default=None, help="chi danh gia N mau test dau")
    ap.add_argument("--metric", type=str, default="soft", choices=["soft", "insdel"],
                    help="soft = Soft-Faith (khong li voi feature tuong quan); insdel = conditional ins/del")
    ap.add_argument("--n_soft", type=int, default=20, help="so mau Bernoulli cho soft metric")
    ap.add_argument("--insdel_steps", type=int, default=None, help="so buoc ins/del (mac dinh D)")
    ap.add_argument("--score", type=str, default="softmax", choices=["logit", "softmax"])
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def load_tabular(name, test_size, seed, device):
    """Load sklearn dataset, standardise theo TRAIN, tra tensors tren device."""
    d = DATASETS[name]()
    X, y = d.data, d.target
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)
    sc = StandardScaler().fit(Xtr)                         # chuan hoa theo TRAIN
    Xtr = torch.tensor(sc.transform(Xtr), dtype=torch.float32, device=device)
    Xte = torch.tensor(sc.transform(Xte), dtype=torch.float32, device=device)
    ytr = torch.tensor(ytr, dtype=torch.long, device=device)
    yte = torch.tensor(yte, dtype=torch.long, device=device)
    n_class = int(y.max()) + 1
    return Xtr, ytr, Xte, yte, n_class


def train_classifier(Xtr, ytr, n_class, args, device):
    """MLP classifier tren tabular that. Tra ve model.eval()."""
    model = MLP(Xtr.shape[1], hidden=args.hidden, n_out=n_class).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()
    bs = 256
    for ep in range(args.epochs):
        perm = torch.randperm(Xtr.shape[0], device=device)
        for i in range(0, Xtr.shape[0], bs):
            idx = perm[i:i + bs]
            loss = F.cross_entropy(model(Xtr[idx]), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    model.n_out = n_class
    return model


def main():
    args = parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[!] cuda khong san sang -> cpu"); device = "cpu"
    torch.manual_seed(args.seed)

    Xtr, ytr, Xte, yte, n_class = load_tabular(args.dataset, args.test_size, args.seed, device)
    D = Xtr.shape[1]
    target = args.target_class if args.target_class is not None else (1 if n_class == 2 else int(torch.bincount(ytr).argmax().item()))
    print(f"[i] dataset={args.dataset}  D={D}  n_class={n_class}  target_class={target}")
    print(f"[i] train={Xtr.shape[0]}  test={Xte.shape[0]}  N={args.N}  score={args.score}")

    model = train_classifier(Xtr, ytr, n_class, args, device)
    with torch.no_grad():
        acc = (model(Xte).argmax(1) == yte).float().mean().item()
    print(f"[i] test acc = {acc:.4f}")

    # reference (mu,Sigma) + PPCA + imputer — tat ca fit tren TRAIN (split rieng voi test)
    ref = fit_reference(Xtr, floor=args.floor)
    ppca_ref, psi = fit_ppca(Xtr, q=min(args.ppca_q, D - 1))
    imputer = fit_imputer(Xtr)
    grad_fn = make_tabular_gradfn(model, device, chunk=args.chunk, target=target, score=args.score)

    mu = Xtr.mean(dim=0)
    med = Xtr.median(dim=0).values
    zero = torch.zeros(D, device=device)
    g = torch.Generator(device="cpu"); g.manual_seed(args.seed + 5)
    rand_pool = Xtr[torch.randperm(Xtr.shape[0], generator=g)[:max(args.eg_K)]]

    # chi giai thich mau THUOC LOP target (ins/del don dieu)
    with torch.no_grad():
        is_tgt = model(Xte).argmax(1) == target
    X_eval = Xte[is_tgt]
    if args.limit:
        X_eval = X_eval[:args.limit]
    if X_eval.shape[0] == 0:
        print("[!] khong co mau lop target trong test"); return
    print(f"[i] danh gia tren {X_eval.shape[0]} mau lop target\n")

    N = args.N

    def attr_for(name, x):
        if name == "IG-zero":   return ig_tabular(x, zero, grad_fn, T=N)
        if name == "IG-mean":   return ig_tabular(x, mu, grad_fn, T=N)
        if name == "IG-median": return ig_tabular(x, med, grad_fn, T=N)
        if name == "IG-random": return ig_tabular(x, rand_pool[0], grad_fn, T=N)
        if name.startswith("EG-"):
            K = int(name.split("-")[1])
            return eg_tabular(x, rand_pool[:K], grad_fn, N=N)
        if name.startswith("Shrinkage-IG@"):
            tau = float(name.split("@")[1])
            return ig_tabular(x, shrinkage_baseline(x, ref, tau=tau), grad_fn, T=N)
        if name == "PM-IG-PPCA":
            return ig_tabular(x, shrinkage_baseline(x, ppca_ref, tau=psi), grad_fn, T=N)
        raise ValueError(name)

    methods = ["IG-zero", "IG-mean", "IG-median", "IG-random"]
    methods += [f"EG-{K}" for K in args.eg_K]
    methods += [f"Shrinkage-IG@{t:g}" for t in args.tau_sweep]
    methods += ["PM-IG-PPCA"]
    print(f"[i] PM-IG-PPCA psi(estimated) = {psi:.4f}  (rank q={min(args.ppca_q, D-1)})")
    print(f"[i] metric = {args.metric}"
          + ("  (Soft-Faith: khong li voi feature tuong quan, khong can group)\n"
             if args.metric == "soft"
             else "  (conditional insertion/deletion — co the LI khi feature tuong quan manh)\n"))

    table, gaps_by_method = {}, {}
    if args.metric == "soft":
        print(f"{'method':<20}{'Soft-NC↑':>12}{'Soft-NS↑':>12}{'Soft-gap↑':>12}{'  (gap ± boot-SE)'}")
        print("-" * 68)
        for nm in methods:
            ncs, nss, gaps = [], [], []
            for i in range(X_eval.shape[0]):
                x = X_eval[i]
                phi = attr_for(nm, x)
                r = soft_faith_tabular(model, x, phi, mu, target=target,
                                       score=args.score, n_samples=args.n_soft)
                ncs.append(r["soft_nc"]); nss.append(r["soft_ns"]); gaps.append(r["soft_gap"])
            gap_t = torch.tensor(gaps)
            se = _bootstrap_se(gap_t, n_boot=args.n_boot, seed=args.seed)
            gaps_by_method[nm] = gaps
            table[nm] = {"soft_nc": sum(ncs) / len(ncs), "soft_ns": sum(nss) / len(nss),
                         "id_gap": gap_t.mean().item(), "id_se": se}
            print(f"{nm:<20}{table[nm]['soft_nc']:>12.4f}{table[nm]['soft_ns']:>12.4f}"
                  f"{table[nm]['id_gap']:>12.4f}   ± {se:.4f}")
    else:
        print(f"{'method':<20}{'insertion↑':>12}{'deletion↓':>12}{'I-D↑':>10}{'  (mean ± boot-SE)'}")
        print("-" * 66)
        for nm in methods:
            ins, dels, gaps = [], [], []
            for i in range(X_eval.shape[0]):
                x = X_eval[i]
                phi = attr_for(nm, x)
                r = insertion_deletion_tabular(model, x, phi, imputer, steps=args.insdel_steps,
                                               target=target, score=args.score)
                ins.append(r["insertion_auc"]); dels.append(r["deletion_auc"]); gaps.append(r["id_gap"])
            gap_t = torch.tensor(gaps)
            se = _bootstrap_se(gap_t, n_boot=args.n_boot, seed=args.seed)
            gaps_by_method[nm] = gaps
            table[nm] = {"insertion": sum(ins) / len(ins), "deletion": sum(dels) / len(dels),
                         "id_gap": gap_t.mean().item(), "id_se": se}
            print(f"{nm:<20}{table[nm]['insertion']:>12.4f}{table[nm]['deletion']:>12.4f}"
                  f"{table[nm]['id_gap']:>10.4f}   ± {se:.4f}")

    gap_label = "Soft-gap" if args.metric == "soft" else "I-D"
    print("-" * 68)
    best = max(table, key=lambda k: table[k]["id_gap"])
    print(f"[i] best {gap_label}: {best} = {table[best]['id_gap']:.4f} ± {table[best]['id_se']:.4f}")

    # ---- Paired test: Shrinkage-IG(gap tot nhat) vs baseline (ghep cap per-sample) ----
    shr = [m for m in methods if m.startswith("Shrinkage-IG")]
    refm = max(shr, key=lambda m: table[m]["id_gap"]) if shr else best
    print(f"\n=== PAIRED TEST: {refm} vs baseline (n={X_eval.shape[0]} mau, metric={gap_label}) ===")
    print(f"{'vs method':<20}{'mean_diff':>12}{'t':>9}{'p(t)':>11}{'z(W)':>9}{'p(Wilcox)':>12}")
    print("-" * 73)
    from e1_batch_image import paired_t, wilcoxon     # dung chung stat tu luc
    stat_rows = []
    a = gaps_by_method[refm]
    for m in methods:
        if m == refm: continue
        b = gaps_by_method[m]
        md, t, pt = paired_t(a, b)
        W, z, pw = wilcoxon(a, b)
        print(f"{m:<20}{md:>12.4f}{t:>9.3f}{pt:>11.4g}{z:>9.3f}{pw:>12.4g}")
        stat_rows.append({"ref": refm, "vs": m, "metric": gap_label,
                          "mean_diff": md, "t": t, "p_t": pt, "z_wilcoxon": z, "p_wilcoxon": pw})

    with open(f"e1_tabular_{args.dataset}_paired.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stat_rows[0].keys()))
        w.writeheader(); w.writerows(stat_rows)
    with open(f"e1_tabular_{args.dataset}_summary.csv", "w", newline="") as f:
        cols = (["method", "soft_nc", "soft_ns", "id_gap", "id_se"] if args.metric == "soft"
                else ["method", "insertion", "deletion", "id_gap", "id_se"])
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for m in methods:
            w.writerow({"method": m, **{k: table[m][k] for k in cols if k != "method"}})
    print(f"\n[i] da luu -> e1_tabular_{args.dataset}_summary.csv, e1_tabular_{args.dataset}_paired.csv")


if __name__ == "__main__":
    main()