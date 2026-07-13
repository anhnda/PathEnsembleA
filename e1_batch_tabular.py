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
from sklearn.preprocessing import MinMaxScaler

# dung lai toan bo may moc tu synthetic_e0.py
from synthetic_e0 import (
    fit_reference, shrinkage_baseline, fit_ppca,
    ig_tabular, eg_tabular, MLP, score_target,
    make_tabular_gradfn, fit_imputer, insertion_deletion_tabular,
    soft_faith_tabular, _bootstrap_se,
)
from pea.baselines_rival import (
    mlp_penultimate, ig2_tabular, sample_cf_ref_tabular,
    max_entropy_baseline_tab, ig_from_baseline_tab, fringe_tabular,
)
import tau_diag
import tau_star as taustar


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
    # --- doi trong: IG2 / Max-Entropy / FRInGe ---
    ap.add_argument("--rivals", action="store_true", help="bat IG2 / Max-Entropy / FRInGe")
    ap.add_argument("--ig2_steps", type=int, default=40)
    ap.add_argument("--me_steps", type=int, default=100)
    ap.add_argument("--test_size", type=float, default=0.3)
    ap.add_argument("--limit", type=int, default=None, help="chi danh gia N mau test dau")
    ap.add_argument("--metric", type=str, default="insdel", choices=["insdel", "soft"],
                    help="insdel = insertion/deletion remove-to-zero (mac dinh, hop tabular); "
                         "soft = Soft-Faith (KHONG hop tabular 1-chieu du thua, chi de tham khao)")
    ap.add_argument("--insdel_mode", type=str, default="marginal",
                    choices=["marginal", "conditional", "zero"],
                    help="marginal (MAC DINH) = boc gia tri that tu cot, pha thong tin nhung "
                         "GIU in-distribution (dung cho tabular duong); conditional = Gaussian "
                         "cond mean (co the li); zero = remove ve 0 (CANH BAO: OOD, deletion dinh tran ~1)")
    ap.add_argument("--rand_seeds", type=int, default=5,
                    help="so seed cho baseline ngau nhien (IG-random/EG) — bao cao mean±std de kh khoi may rui")
    ap.add_argument("--n_soft", type=int, default=20, help="so mau Bernoulli cho soft metric")
    ap.add_argument("--insdel_steps", type=int, default=None, help="so buoc ins/del (mac dinh D)")
    ap.add_argument("--score", type=str, default="softmax", choices=["logit", "softmax"])
    # --- TAU-DIAGNOSTIC (dense sweep, per-input, KHONG dung ins/del de chon) ---
    ap.add_argument("--tau_star", action="store_true",
                    help="tau* CLOSED FORM tu pho evidence (1 backward pass, KHONG sweep). "
                         "log tau* = sum_k |c_k| log s_k / sum_k |c_k|, c_k = <grad F, v_k><x-mu, v_k>")
    ap.add_argument("--tau_diag", action="store_true",
                    help="quet DENSE tau, log Δf/|b-x|₂/TI GIA BIEN per-input, tinh tau_rate "
                         "(rule chinh) + doi chung ORACLE(I-D). Xuat CSV long-format.")
    ap.add_argument("--diag_n", type=int, default=25, help="so diem tren log-grid tau (4-5 la KHONG DU)")
    ap.add_argument("--diag_lo", type=float, default=None, help="tau min (mac dinh: gamma=1e-2 * s_bar)")
    ap.add_argument("--diag_hi", type=float, default=None, help="tau max (mac dinh: gamma=1e2 * s_bar)")
    ap.add_argument("--diag_gamma", action="store_true",
                    help="dung grid SCALE-FREE tau = gamma*s_bar (de so sanh duoc cross-modality)")
    ap.add_argument("--diag_oracle", action="store_true",
                    help="chay ins/del tai MOI tau tren grid de lay ORACLE tau (DAT — chi de doi chung)")
    ap.add_argument("--diag_eps", type=float, default=0.01,
                    help="nguong TUONG DOI cho ti gia bien: dung khi r(tau) < eps*max(r)")
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
    sc = MinMaxScaler().fit(Xtr)                           # normalize [0,1] theo TRAIN
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


def _fmt_strength(s):
    """
    BON cot, GIONG HET nhau o ca ba modality:  f(x)  f(b)  Δf  |b-x|₂   (+P2)

    Δf = f(x) - f(b) = NGAN SACH COMPLETENESS (= sum_i phi_i).
    |b-x|₂ = QUANG DUONG (L2, khong phai .abs().mean() = L1/D nhu code cu).

    Hai cot nay du de doc moi thu: cung Δf, ai di xa hon thi te hon.
    Ti gia bien d(Δf)/d|b-x| chi tinh duoc giua cac hang CUNG TRUC tau
    -> in o bang --tau_diag.

    DA BO:
      ratio = f(b)/f(x)      -> bo mat f(x). Δf da co f(x) ben trong.
      SNR   = Δf/|b-x|²      -> hong o D lon (vision: mau so ~1e4, in ra 0.0000).
      |b-x|/|x-mu|           -> o vision mu = TENSOR 0 => ||x-mu|| = ||x||, vo nghia.
      amp                    -> can K => modality-specific.
    Moi modality mot measure thi bang cross-modality vo nghia.
    """
    if s is None:
        return f"{'-':>8}{'-':>8}{'-':>9}{'-':>10}{'-':>6}"
    g = lambda k: (s[k].mean().item() if k in s else float("nan"))
    f = lambda v, w, p=4: (f"{v:>{w}.{p}f}" if v == v else f"{'-':>{w}}")
    p2 = s["P2_ok"].mean().item() * 100 if "P2_ok" in s else float("nan")
    p2s = f"{p2:>5.0f}%" if p2 == p2 else f"{'-':>6}"
    return f(g("f_x"),8) + f(g("f_b"),8) + f(g("delta_f"),9) + f(g("dist"),10) + p2s


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

    # --- doi trong: rep layer + CF pool + n_class ---
    rep_fn = mlp_penultimate(model) if args.rivals else None
    cf_pool = Xtr if args.rivals else None
    n_class = model.n_out if hasattr(model, "n_out") else 2
    if args.rivals:
        print(f"[i] RIVALS bat: IG2 (ig2_steps={args.ig2_steps}) / MaxEnt / FRInGe "
              f"(me_steps={args.me_steps})")

    def rand_pool_for(rng_seed):
        gg = torch.Generator(device="cpu"); gg.manual_seed(rng_seed)
        idx = torch.randperm(Xtr.shape[0], generator=gg)[:max(args.eg_K)]
        return Xtr[idx]

    def attr_for(name, x, rng_seed=None):
        if name == "IG-zero":   return ig_tabular(x, zero, grad_fn, T=N)
        if name == "IG-mean":   return ig_tabular(x, mu, grad_fn, T=N)
        if name == "IG-median": return ig_tabular(x, med, grad_fn, T=N)
        if name == "IG-random":
            pool = rand_pool_for(rng_seed if rng_seed is not None else args.seed + 5)
            return ig_tabular(x, pool[0], grad_fn, T=N)
        if name.startswith("EG-"):
            K = int(name.split("-")[1])
            pool = rand_pool_for(rng_seed if rng_seed is not None else args.seed + 5)
            return eg_tabular(x, pool[:K], grad_fn, N=N)
        if name.startswith("Shrinkage-IG@"):
            tau = float(name.split("@")[1])
            return ig_tabular(x, shrinkage_baseline(x, ref, tau=tau), grad_fn, T=N)
        if name == "PM-IG-PPCA":
            return ig_tabular(x, shrinkage_baseline(x, ppca_ref, tau=psi), grad_fn, T=N)
        # --- doi trong ---
        if name == "IG-MaxEnt":
            b = max_entropy_baseline_tab(model, x, n_class, steps=args.me_steps)
            return ig_from_baseline_tab(model, x, b, target, T=N, score=args.score)
        if name == "FRInGe":
            return fringe_tabular(model, x, target, n_class, steps=N,
                                  me_steps=args.me_steps, score=args.score)
        if name == "IG2":
            xr = sample_cf_ref_tabular(model, x, target, cf_pool, score=args.score)
            return ig2_tabular(model, x, xr, target, rep_fn,
                               steps=args.ig2_steps, score=args.score)
        raise ValueError(name)

    def is_stochastic(name):
        return name == "IG-random" or name.startswith("EG-")

    def baseline_vec_for(name, x):
        """Baseline DIEM cho method (khop attr_for). None neu khong 1-diem (EG/random pool)."""
        if name == "IG-zero":   return zero
        if name == "IG-mean":   return mu
        if name == "IG-median": return med
        if name.startswith("Shrinkage-IG@"):
            return shrinkage_baseline(x, ref, tau=float(name.split("@")[1]))
        if name == "PM-IG-PPCA":
            return shrinkage_baseline(x, ppca_ref, tau=psi)
        # --- doi trong: baseline DIEM (de debug strength) ---
        if name == "IG-MaxEnt":
            return max_entropy_baseline_tab(model, x, n_class, steps=args.me_steps)
        if name == "FRInGe":
            return max_entropy_baseline_tab(model, x, n_class, steps=args.me_steps)  # ref cua FRInGe
        if name == "IG2":
            xr = sample_cf_ref_tabular(model, x, target, cf_pool, score=args.score)
            # GradCF = endpoint cua GradPath; tinh nhanh bang chinh ig2 path (chi lay baseline)
            with torch.enable_grad():
                eta = x.abs().mean().item() * 2.0 + 1e-3
                x_ref_rep = rep_fn(xr).detach()
                delta = torch.zeros_like(x)
                for _ in range(args.ig2_steps):
                    xd = (x + delta).clone().requires_grad_(True)
                    d = (rep_fn(xd) - x_ref_rep).pow(2).sum()
                    g, = torch.autograd.grad(d, xd)
                    delta = (delta - eta * g / (g.norm() + 1e-12)).detach()
                return (x + delta).detach()               # GradCF
        return None            # IG-random, EG-*: pool nhieu diem -> bo qua debug

    @torch.no_grad()
    def baseline_strength(name):
        """
        Per-input, KHONG trung binh som. Tra ve dict cac tensor (M,):
            f_x, f_b, rho, delta_f = f(x)-f(b)   <- NGAN SACH COMPLETENESS
            dist  = ||b-x||_2   (SUA: truoc day la (b-x).abs().mean() = L1/D,
                                 KHONG phai quang duong Euclid, dung cho SNR la sai)
            dist2 = ||b-x||_2^2 <- meo mo IG ~ O(L * dist2), khong phu thuoc f(x)
            (khong co dist_norm/amp: chung modality-specific, xem tau_diag.py)
            maha_b, P2_ok       <- kiem tra (P2) contraction cho tung baseline
        """
        if baseline_vec_for(name, X_eval[0]) is None:
            return None
        return tau_diag.fixed_baseline_diag(
            X_eval,
            lambda Z: score_target(model, Z, target=target, score="softmax"),
            lambda x: baseline_vec_for(name, x),
            ref_s=ref.s, ref_V=ref.V, ref_mu=ref.mu,
        )

    methods = ["IG-zero", "IG-mean", "IG-median", "IG-random"]
    methods += [f"EG-{K}" for K in args.eg_K]
    methods += [f"Shrinkage-IG@{t:g}" for t in args.tau_sweep]
    methods += ["PM-IG-PPCA"]
    if args.rivals:
        methods += ["IG-MaxEnt", "FRInGe", "IG2"]
    print(f"[i] PM-IG-PPCA psi(estimated) = {psi:.4f}  (rank q={min(args.ppca_q, D-1)})")
    print(f"[i] metric = {args.metric}"
          + (f"  (insertion/deletion remove-to-{args.insdel_mode})\n" if args.metric == "insdel"
             else "  (Soft-Faith — KHONG hop tabular 1-chieu du thua, chi tham khao)\n"))

    # =====================================================================
    # TAU-DIAGNOSTIC: quet DENSE, per-input. Chay TRUOC bang E1.
    # Muc tieu: chon tau bang mot tieu chi NOI TAI (chi forward pass), khong
    # cham insertion/deletion — de tranh dung toi ma draft dang buoc cho BEE
    # ("optimises a learned baseline against a metric").
    # =====================================================================
    if args.tau_diag:
        score_fn = lambda Z: score_target(model, Z, target=target, score="softmax")

        # (0) Pho cua Sigma: knee co SAC khong?
        PR = tau_diag.participation_ratio(ref.s)
        s_bar = tau_diag.effective_tau_scale(ref.s, "mean")
        print(f"\n=== PHO CUA SIGMA ===")
        print(f"[i] D = {D},  participation ratio PR = {PR:.2f}  ({PR/D*100:.1f}% cua D)")
        print(f"[i] s_bar (mean eigval) = {s_bar:.6g}   s_max = {ref.s.max():.6g}  s_min = {ref.s.min():.6g}")
        if PR < 0.2 * D:
            print("[i] PR NHO => pho tap trung => duong cong doc, gay SAC => tau_rate on dinh.")
        else:
            print("[!] PR LON => pho trai dai => knee MO => moi rule chon tau se BAT DINH.")
            print("[!] Dung ky vong rule sac net o modality nay (vd anh tu nhien, pho 1/f^2).")

        # (1) Grid
        if args.diag_gamma:
            taus_d, sb = tau_diag.gamma_grid(ref.s, lo=1e-2, hi=1e2, n=args.diag_n)
            print(f"[i] grid SCALE-FREE: tau = gamma * s_bar, gamma in [1e-2, 1e2], s_bar={sb:.6g}")
        else:
            lo = args.diag_lo if args.diag_lo else 1e-2 * s_bar
            hi = args.diag_hi if args.diag_hi else 1e2 * s_bar
            taus_d = tau_diag.log_tau_grid(lo, hi, args.diag_n)
            print(f"[i] grid tau: [{lo:.4g}, {hi:.4g}], {args.diag_n} diem log-spaced")

        # (2) Quet dense
        curve = tau_diag.sweep_curve(
            X_eval, score_fn,
            lambda x, t: shrinkage_baseline(x, ref, tau=t),
            taus_d, mu=mu, ref_s=ref.s, ref_V=ref.V, ref_mu=ref.mu,
        )
        tau_diag.print_curve_table(curve, tag=f"[tabular/{args.dataset}]")

        # (3) Cac baseline CO DINH — xep chung LEN CUNG TRUC, cung don vi
        tau_diag.print_fixed_header()
        for nm in ["IG-zero", "IG-mean", "IG-median"] + (["IG-MaxEnt", "IG2"] if args.rivals else []):
            d = baseline_strength(nm)
            if d is not None:
                tau_diag.print_fixed_row(nm, d)
        print("[i] P2 ok% = ti le input ma |b-mu|_S-1 <= |x-mu|_S-1 (P2 contraction).")
        print("[i] Neu IG-zero co P2 ok% CAO => hang 'zero ✗' trong Table 1 la SAI o modality nay.")

        # (4) ORACLE: tau tot nhat theo chinh I-D (CHI de doi chung, KHONG dung de chon)
        oracle = None
        if args.diag_oracle:
            print(f"\n[i] dang chay ORACLE: ins/del tai {len(taus_d)} tau x {X_eval.shape[0]} mau (DAT)...")
            og = torch.Generator(device="cpu"); og.manual_seed(args.seed + 7)
            id_per_tau = torch.zeros(X_eval.shape[0], len(taus_d), device=device)
            for t_i, t in enumerate(taus_d):
                for i in range(X_eval.shape[0]):
                    x = X_eval[i]
                    b = shrinkage_baseline(x, ref, tau=t)
                    phi = ig_tabular(x, b, grad_fn, T=N)
                    r = insertion_deletion_tabular(model, x, phi, imputer, steps=args.insdel_steps,
                                                   target=target, score=args.score,
                                                   mode=args.insdel_mode, X_pool=Xtr, gen=og)
                    id_per_tau[i, t_i] = r["id_gap"]
            oracle = tau_diag.oracle_tau(curve, id_per_tau)
            print(f"[i] ORACLE I-D theo tau (aggregate): "
                  f"argmax tai tau={taus_d[int(id_per_tau.mean(0).argmax())]:.4g}")
            curve["_id_per_tau"] = id_per_tau

        # (5) Selection rules — deu CHI dung forward pass
        rules, valid_m = tau_diag.selection_rules(curve, eps=args.diag_eps)
        tau_diag.print_rules_table(rules, oracle=oracle, valid=valid_m)

        # (6) CSV long-format — de fit rule offline, dung suy dien tu 4 diem nua
        extra = {"id_gap": curve["_id_per_tau"]} if "_id_per_tau" in curve else None
        tau_diag.dump_curve_csv(curve, f"e1_tabular_{args.dataset}_taucurve.csv", extra_cols=extra)

    # =====================================================================
    # tau* CLOSED FORM — bo tau khoi danh sach sieu tham so.
    #   g_k(tau) = tau/(s_k+tau) = sigmoid(log tau - log s_k)
    #   => Δf(log tau) = tong sigmoid, moi eigen-direction mot bac thang tai log s_k
    #   => knee (Δf''=0) = tam khoi eigenvalue mang nhieu evidence nhat
    #   => tau* = trung binh hinh hoc co trong so cua s_k, trong so |c_k|
    # Mot backward pass. Khong sweep, khong grid, per-input tu dong.
    # =====================================================================
    if args.tau_star:
        G = grad_fn(X_eval)                                   # (M,D) grad tai chinh x
        tstar, tdiag = taustar.tau_star(X_eval, G, ref)       # (M,)

        sp = taustar.evidence_spectrum(X_eval, G, ref)
        taustar.print_evidence_spectrum(sp, tag=f"[tabular/{args.dataset}]")

        q = torch.tensor([0.25, 0.5, 0.75], device=tstar.device).double()
        qq = torch.quantile(tstar.double(), q)
        print(f"\n=== tau* CLOSED FORM (per-input, n={X_eval.shape[0]}) ===")
        print(f"[i] median {qq[1]:.6g}   mean {tstar.mean():.6g}   IQR [{qq[0]:.6g}, {qq[2]:.6g}]")
        print(f"[i] k_eff per-input: median {tdiag['k_eff'].median():.1f} / D={D}")
        print(f"[i] sd(log s) per-input: median {tdiag['sd_log_s'].median():.3f}")
        print(f"[i] tau_sweep hien tai = {args.tau_sweep}")
        print("[i] So sanh tau* voi tau_sweep: neu tau* roi vao vung best cua sweep")
        print("[i]   => tau khong con la sieu tham so, no la CLOSED FORM tu (F, x, Sigma).")

        # doi chieu voi sweep (neu co --tau_diag thi dung chung curve; neu khong, quet nhanh)
        taus_cmp = sorted(args.tau_sweep)
        with torch.no_grad():
            Dm = torch.zeros(X_eval.shape[0], len(taus_cmp), device=device)
            Lm = torch.zeros(X_eval.shape[0], len(taus_cmp), device=device)
            fx = score_target(model, X_eval, target=target, score="softmax")
            for j, t in enumerate(taus_cmp):
                B = shrinkage_baseline(X_eval, ref, tau=t)
                Dm[:, j] = fx - score_target(model, B, target=target, score="softmax")
                Lm[:, j] = (X_eval - B).norm(dim=1)
        taustar.compare_rules(torch.tensor(taus_cmp, device=device, dtype=torch.float),
                              Lm, Dm, tstar)

    # baseline strength f(x) vs f(baseline) cho moi method (None neu pool nhieu diem)
    bl_str = {nm: baseline_strength(nm) for nm in methods}

    table, gaps_by_method = {}, {}
    if args.metric == "soft":
        print(f"{'method':<20}{'Soft-NC↑':>12}{'Soft-NS↑':>12}{'Soft-gap↑':>12}"
              f"{'f(x)':>8}{'f(b)':>8}{'Δf':>9}{'|b-x|₂':>10}{'P2':>6}"
              f"{'  (gap±SE)'}")
        print("-" * 101)
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
                         "id_gap": gap_t.mean().item(), "id_se": se,
                         "rank_key": sum(ncs) / len(ncs),          # xep hang theo Soft-NC
                         "nc_list": ncs}                            # cho paired test theo NC
            scol = _fmt_strength(bl_str.get(nm))
            print(f"{nm:<20}{table[nm]['soft_nc']:>12.4f}{table[nm]['soft_ns']:>12.4f}"
                  f"{table[nm]['id_gap']:>12.4f}{scol}   ± {se:.4f}")
    else:
        print(f"{'method':<20}{'insertion↑':>12}{'deletion↓':>12}{'I-D↑':>10}"
              f"{'f(x)':>8}{'f(b)':>8}{'Δf':>9}{'|b-x|₂':>10}{'P2':>6}"
              f"{'  (mean±SE[±seed-std])'}")
        print("-" * 105)
        idg = torch.Generator(device="cpu"); idg.manual_seed(args.seed + 7)   # boc marginal
        for nm in methods:
            seeds = list(range(args.rand_seeds)) if is_stochastic(nm) else [None]
            # gap per-sample TRUNG BINH qua seed (de paired test), + seed-level I-D de bao variance
            per_sample_gap = None
            seed_id_means = []
            ins_all, del_all = [], []
            for sd in seeds:
                rs = (args.seed + 1000 + sd) if sd is not None else None
                ins, dels, gaps = [], [], []
                for i in range(X_eval.shape[0]):
                    x = X_eval[i]
                    phi = attr_for(nm, x, rng_seed=rs)
                    r = insertion_deletion_tabular(model, x, phi, imputer, steps=args.insdel_steps,
                                                   target=target, score=args.score, mode=args.insdel_mode,
                                                   X_pool=Xtr, gen=idg)
                    ins.append(r["insertion_auc"]); dels.append(r["deletion_auc"]); gaps.append(r["id_gap"])
                gaps_t = torch.tensor(gaps)
                per_sample_gap = gaps_t if per_sample_gap is None else per_sample_gap + gaps_t
                seed_id_means.append(gaps_t.mean().item())
                ins_all.append(sum(ins) / len(ins)); del_all.append(sum(dels) / len(dels))
            per_sample_gap = per_sample_gap / len(seeds)          # trung binh qua seed
            se = _bootstrap_se(per_sample_gap, n_boot=args.n_boot, seed=args.seed)
            seed_std = float(torch.tensor(seed_id_means).std().item()) if len(seeds) > 1 else 0.0
            gaps_by_method[nm] = per_sample_gap.tolist()
            table[nm] = {"insertion": sum(ins_all) / len(ins_all),
                         "deletion": sum(del_all) / len(del_all),
                         "id_gap": per_sample_gap.mean().item(), "id_se": se,
                         "seed_std": seed_std, "n_seeds": len(seeds)}
            tail = f"   ± {se:.4f}" + (f"  [seed-std {seed_std:.4f}, n={len(seeds)}]" if len(seeds) > 1 else "")
            scol = _fmt_strength(bl_str.get(nm))
            print(f"{nm:<20}{table[nm]['insertion']:>12.4f}{table[nm]['deletion']:>12.4f}"
                  f"{table[nm]['id_gap']:>10.4f}{scol}{tail}")

    gap_label = "Soft-gap" if args.metric == "soft" else "I-D"
    print("-" * 68)
    best = max(table, key=lambda k: table[k]["id_gap"])
    print(f"[i] best {gap_label}: {best} = {table[best]['id_gap']:.4f} ± {table[best]['id_se']:.4f}"
          f"   <-- best")
    if bl_str.get(best) is not None:
        s = bl_str[best]
        print(f"[i] best: f(x)={s['f_x'].mean():.4f} f(b)={s['f_b'].mean():.4f} "
              f"Δf={s['delta_f'].mean():.4f} |b-x|₂={s['dist'].mean():.4f}")
    print("[i] Δf = f(x)-f(b) = ngan sach Completeness (= sum phi_i, Completeness).")
    print("[i] Chay --tau_diag de co TI GIA BIEN d(Δf)/d|b-x| va tau_rate (rule chon tau).")

    # ---- Paired test: Shrinkage-IG(gap tot nhat) vs baseline (ghep cap per-sample) ----
    shr = [m for m in methods if m.startswith("Shrinkage-IG")]
    refm = max(shr, key=lambda m: table[m]["id_gap"]) if shr else best
    print(f"\n=== PAIRED TEST: {refm} vs baseline (n={X_eval.shape[0]} mau, metric={gap_label}) ===")
    print(f"{'vs method':<20}{'mean_diff':>12}{'t':>9}{'p(t)':>11}{'z(W)':>9}{'p(Wilcox)':>12}")
    print("-" * 73)
    from stats_utils import paired_t, wilcoxon     # module stat doc lap (khong keo torchvision)
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