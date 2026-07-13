"""
E1 (baseline comparison) tren REAL ANH — trung binh tren mot folder benchmark.

Bam sat compare_batch.py NHUNG chi giu IG + cac BASELINE (bo PEA/SBA/Diffusion/BlurLIG
path-methods). Trong tam E1 draft: so DIEM BASELINE, path giu thang.

Methods (moi cai IG THANG tu 1 baseline, cung ngan sach N buoc):
    IG-black / IG-white / IG-noise / IG-blur / IG-mean            (baseline co dinh)
    IG-random-K   (EG: pool K baseline sample, N chia deu)         K in {1,4,16,64}
    Shrinkage-IG@sigma  (Wiener low-pass FFT per-image, quet sigma)
        -> day la Shrinkage-IG cho ANH: eigenbasis Sigma = Fourier (prior 1/f^2),
           gain Wiener s_k/(s_k+tau). Blur-IG la truong hop LOW-PASS CO DINH; ta
           quet muc cat sigma => Blur-IG la MOT diem tren truc nay (Cor. Wiener).
           KHONG dung reference set, KHONG dung Sigma D×D — covariance nam trong pho.

Metric: insertion / deletion / I-D (insdel.py, RISE-style, substrate blur/black).
  KHONG tai dung baseline lam mask (substrate doc lap voi baseline cua method).
Sau MOI anh in bang TICH LUY (running mean±SE + win%). Cuoi cung: paired test
  (Wilcoxon/paired-t) Shrinkage-IG(sigma tot nhat) vs tung baseline. Luu CSV.

Chay (torch GPU mac dinh, tu chay lay):
    python e1_batch_image.py benchmark_50 --N 500 --chunk 16
    python e1_batch_image.py benchmark_50 --N 500 --substrate black --glob '*.JPEG'
    python e1_batch_image.py imgs --sigma_sweep 2 4 8 16 --eg_K 1 4 16

KHONG train, KHONG smoketest.
"""

import argparse
import glob
import os
import csv
import math
import torch

from pea.resnet50_gradfn import (
    load_resnet50, make_resnet50_gradfn, preprocess,
    IMAGENET_MEAN, IMAGENET_STD,
)
from pea.insdel import insertion_deletion
from pea.methods import ig_single, eg
from pea.spectral_reference import spectral_reference_fft
from pea.baselines_rival import (
    resnet50_penultimate, ig2_attribution, sample_counterfactual_ref,
    max_entropy_baseline, ig_from_baseline, fringe_attribution,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="thu muc chua anh, vd benchmark_50")
    ap.add_argument("--glob", type=str, default="*.JPEG", help="mau ten file")
    ap.add_argument("--target", type=int, default=None, help="ep target chung; mac dinh top-1 moi anh")
    ap.add_argument("--N", type=int, default=500, help="ngan sach: so gradient eval/anh")
    # --- Shrinkage-IG (FFT Wiener low-pass), quet muc cat sigma (pixel) ---
    ap.add_argument("--sigma_sweep", type=float, nargs="+", default=[2.0, 4.0, 8.0, 16.0],
                    help="dai sigma (pixel) cho Shrinkage-IG FFT. Blur-IG cu ~ sigma=k/3~10.")
    # --- EG (random-sample pool) ---
    ap.add_argument("--eg_K", type=int, nargs="+", default=[1, 4, 16, 64],
                    help="so baseline pool cho EG (random sample quanh anh trong khong gian chuan hoa)")
    ap.add_argument("--eg_noise", type=float, default=1.0,
                    help="do lech chuan mau EG (khong gian da chuan hoa)")
    ap.add_argument("--no_fixed", action="store_true", help="bo cac baseline co dinh (chi Shrinkage + EG)")
    # --- doi trong: IG2 / Max-Entropy / FRInGe ---
    ap.add_argument("--rivals", action="store_true",
                    help="bat cac doi trong IG2 / Max-Entropy / FRInGe")
    ap.add_argument("--ig2_steps", type=int, default=30, help="so buoc GradPath cua IG2")
    ap.add_argument("--me_steps", type=int, default=100, help="so buoc toi uu Max-Entropy baseline")
    ap.add_argument("--rival_T", type=int, default=50, help="so buoc IG cho ME/FRInGe path")
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--insdel_steps", type=int, default=224)
    ap.add_argument("--substrate", type=str, default="blur", choices=["blur", "black"],
                    help="nen cho insertion / gia tri xoa deletion (doc lap voi baseline)")
    ap.add_argument("--score", type=str, default="logit", choices=["logit", "softmax"],
                    help="dung CHUNG cho attribution backward va metric")
    ap.add_argument("--limit", type=int, default=None, help="chi chay N anh dau (debug)")
    ap.add_argument("--report_every", type=int, default=1,
                    help="in bang tich luy moi bao nhieu anh (0 = tat, chi in cuoi)")
    ap.add_argument("--paired_ref", type=str, default=None,
                    help="method lam moc paired-test; mac dinh = Shrinkage-IG sigma tot nhat")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Baseline co dinh (black/white/noise/blur/mean) — deu trong khong gian chuan hoa.
# ---------------------------------------------------------------------------
def make_fixed_baselines(x, device, seed):
    C, H, W = x.shape
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(3, 1, 1)
    black = (torch.zeros(3, H, W, device=device) - mean) / std
    white = (torch.ones(3, H, W, device=device) - mean) / std
    g = torch.Generator(device=device); g.manual_seed(seed + 7)
    noise = torch.randn(3, H, W, generator=g, device=device)
    # mean baseline = anh xam trung binh ImageNet (0 trong khong gian chuan hoa)
    mean_bl = torch.zeros(3, H, W, device=device)
    # blur cua chinh anh (baseline blur co dinh = Blur-IG endpoint, sigma~k/3)
    import torch.nn.functional as F
    k = 31; coords = torch.arange(k, device=device).float() - k // 2
    g1d = torch.exp(-(coords ** 2) / (2 * (k / 3) ** 2)); g1d = g1d / g1d.sum()
    xb = x[None]
    xb = F.conv2d(xb, g1d.view(1, 1, 1, k).repeat(C, 1, 1, 1), padding=(0, k // 2), groups=C)
    xb = F.conv2d(xb, g1d.view(1, 1, k, 1).repeat(C, 1, 1, 1), padding=(k // 2, 0), groups=C)
    blur = xb[0]
    names = ["black", "white", "noise", "mean", "blur"]
    return names, torch.stack([black, white, noise, mean_bl, blur])


def make_eg_pool(x, K, noise, device, seed):
    """Pool K baseline random-sample: x + N(0, noise^2) trong khong gian chuan hoa."""
    C, H, W = x.shape
    g = torch.Generator(device=device); g.manual_seed(seed + 31)
    eps = torch.randn(K, C, H, W, generator=g, device=device) * noise
    return x[None] + eps                                   # (K,3,H,W)


def collect_baselines_for_image(x, args, device, seed):
    """
    Tra ve dict {method: baseline_tensor(3,H,W)} — DUNG cach attribution tao baseline,
    de debug f(x) vs f(baseline). Shrinkage-IG@sigma -> Wiener low-pass FFT;
    IG-<fixed> -> black/white/noise/mean/blur. (EG bo qua: baseline la pool, khong 1 diem.)
    """
    out = {}
    for sig in args.sigma_sweep:
        out[f"Shrinkage-IG@{sig:g}"] = spectral_reference_fft(x, sigma=sig)
    if not args.no_fixed:
        names, fixed = make_fixed_baselines(x, device, seed)
        for nm, b in zip(names, fixed):
            out[f"IG-{nm}"] = b
    return out


@torch.no_grad()
def baseline_strength_row(model, x, baseline, target):
    """p_full=f(x), p_base=f(baseline) (softmax target), ratio, mean|b-x|."""
    import torch.nn.functional as F
    p_full = F.softmax(model(x[None]), 1)[0, target].item()
    p_base = F.softmax(model(baseline[None]), 1)[0, target].item()
    ratio = p_base / p_full if p_full > 1e-9 else float("nan")
    shift = (baseline - x).abs().mean().item()
    return p_full, p_base, ratio, shift


def attributions_for_image(x, grad_fn, args, device, seed,
                           model=None, target=None, rep_fn=None, ref_pool=None, n_class=1000):
    """
    Chi IG + baseline (theo E1, path thang). Tra ve dict {method: attr(3,H,W)}.
      - Shrinkage-IG@sigma : baseline = Wiener low-pass FFT (blur), IG thang toi x.
      - IG-<fixed>         : black/white/noise/mean/blur.
      - EG-K               : trung binh IG tren K random-sample baseline, ngan sach chia deu.
    """
    out = {}
    # --- Shrinkage-IG (FFT), quet sigma. Blur-IG = mot diem tren truc nay. ---
    for sig in args.sigma_sweep:
        ref = spectral_reference_fft(x, sigma=sig)         # (3,H,W) Wiener low-pass
        out[f"Shrinkage-IG@{sig:g}"] = ig_single(x, ref, grad_fn, T=args.N)

    # --- baseline co dinh ---
    if not args.no_fixed:
        names, fixed = make_fixed_baselines(x, device, seed)
        for nm, b in zip(names, fixed):
            out[f"IG-{nm}"] = ig_single(x, b, grad_fn, T=args.N)

    # --- EG (random-sample pool), moi K mot dong ---
    for K in args.eg_K:
        pool = make_eg_pool(x, K, args.eg_noise, device, seed)
        out[f"EG-{K}"] = eg(x, pool, grad_fn, N=args.N)

    # --- DOI TRONG: IG2 / Max-Entropy / FRInGe ---
    if args.rivals and model is not None:
        # Max-Entropy: baseline output ~ uniform, roi IG thang
        b_me = max_entropy_baseline(model, x, n_class, steps=args.me_steps, device=device)
        out["IG-MaxEnt"] = ig_from_baseline(x, b_me, grad_fn, T=args.rival_T)
        # FRInGe: max-ent reference + Fisher-Rao geodesic path
        out["FRInGe"] = fringe_attribution(model, x, target, n_class, grad_fn,
                                           steps=args.rival_T, me_steps=args.me_steps, device=device)
        # IG2: GradCF + GradPath (can reference lop khac + rep layer)
        if rep_fn is not None and ref_pool is not None:
            x_ref = sample_counterfactual_ref(model, x, target, ref_pool, device=device)
            out["IG2"] = ig2_attribution(model, x, x_ref, target, rep_fn,
                                         steps=args.ig2_steps, device=device)
    return out


# ===========================================================================
# Thong ke: mean_se + paired stats tu luc (khong scipy) — sao y compare_batch.py.
# ===========================================================================
def mean_se(vals):
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(vals) / n
    if n == 1:
        return m, 0.0
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    return m, math.sqrt(var / n)


def _norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _t_sf(t, df):
    if df <= 0 or not math.isfinite(t):
        return float("nan")
    x = df / (df + t * t)
    a, b = df / 2.0, 0.5
    bt = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log(1.0 - x)
    ) if 0.0 < x < 1.0 else 0.0

    def _betacf(x, a, b):
        MAXIT, EPS, FPMIN = 200, 3e-12, 1e-300
        qab, qap, qam = a + b, a + 1.0, a - 1.0
        c = 1.0
        d = 1.0 - qab * x / qap
        if abs(d) < FPMIN: d = FPMIN
        d = 1.0 / d; h = d
        for m in range(1, MAXIT + 1):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            if abs(d) < FPMIN: d = FPMIN
            c = 1.0 + aa / c
            if abs(c) < FPMIN: c = FPMIN
            d = 1.0 / d; h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            if abs(d) < FPMIN: d = FPMIN
            c = 1.0 + aa / c
            if abs(c) < FPMIN: c = FPMIN
            d = 1.0 / d; de = d * c; h *= de
            if abs(de - 1.0) < EPS: break
        return h
    if x < (a + 1.0) / (a + b + 2.0):
        ix = bt * _betacf(x, a, b) / a
    else:
        ix = 1.0 - bt * _betacf(1.0 - x, b, a) / b
    return max(0.0, min(1.0, ix))


def paired_t(a, b):
    d = [ai - bi for ai, bi in zip(a, b)]
    n = len(d)
    if n < 2:
        return (float("nan"), float("nan"), float("nan"))
    md = sum(d) / n
    var = sum((x - md) ** 2 for x in d) / (n - 1)
    sd = math.sqrt(var)
    if sd == 0.0:
        return (md, float("inf") if md != 0 else 0.0, 0.0 if md != 0 else 1.0)
    t = md / (sd / math.sqrt(n))
    return (md, t, _t_sf(t, n - 1))


def wilcoxon(a, b):
    d = [ai - bi for ai, bi in zip(a, b) if ai - bi != 0.0]
    n = len(d)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    Wp = sum(ranks[i] for i in range(n) if d[i] > 0)
    Wm = sum(ranks[i] for i in range(n) if d[i] < 0)
    W = min(Wp, Wm)
    mu = n * (n + 1) / 4.0
    from collections import Counter
    tie_counts = Counter(abs(x) for x in d)
    tie_term = sum(t ** 3 - t for t in tie_counts.values())
    sigma2 = n * (n + 1) * (2 * n + 1) / 24.0 - tie_term / 48.0
    if sigma2 <= 0:
        return (W, float("nan"), float("nan"))
    z = (W - mu + 0.5 * (1 if W < mu else -1)) / math.sqrt(sigma2)
    p = 2.0 * _norm_cdf(-abs(z))
    return (W, z, p)


def _print_summary_table(metric_names, acc, id_per_image, n_img, title, bl_strength=None):
    win_count = {m: 0 for m in metric_names}
    for d in id_per_image:
        win_count[max(d, key=d.get)] += 1
    # tim best I-D truoc de danh dau
    id_means = {m: mean_se(acc[m]["id"])[0] for m in metric_names}
    best_m = max(id_means, key=id_means.get)

    print(f"\n--- {title} (mean±SE tren {n_img} anh) ---")
    print(f"{'method':<20}{'insertion↑':>16}{'deletion↓':>16}{'I-D↑':>16}{'win%':>7}"
          f"{'f(x)':>8}{'f(xt)':>8}{'ratio':>8}{'|b-x|':>8}")
    print("-" * 107)
    for m in metric_names:
        im, ise = mean_se(acc[m]["ins"])
        dm, dse = mean_se(acc[m]["del"])
        idm, idse = mean_se(acc[m]["id"])
        winp = 100.0 * win_count[m] / n_img if n_img else 0.0
        # f(x)=p_full, f(xt)=p_base, ratio, shift |b-x| (EG khong co -> "-")
        fx, fxt, rtxt, stxt = "   -  ", "   -  ", "   -  ", "   -  "
        if bl_strength and m in bl_strength:
            d = bl_strength[m]
            if d["pf"]:   fx  = f"{sum(d['pf'])/len(d['pf']):>6.4f}"
            if d["pb"]:   fxt = f"{sum(d['pb'])/len(d['pb']):>6.4f}"
            rr = [r for r in d["ratio"] if r == r]
            if rr:        rtxt = f"{sum(rr)/len(rr):>6.4f}"
            if d["shift"]:stxt = f"{sum(d['shift'])/len(d['shift']):>6.4f}"
        mark = "  <-- best" if m == best_m else ""
        print(f"{m:<20}{im:>8.4f}±{ise:<6.4f}{dm:>8.4f}±{dse:<6.4f}{idm:>8.4f}±{idse:<6.4f}"
              f"{winp:>6.1f}%{fx:>8}{fxt:>8}{rtxt:>8}{stxt:>8}{mark}")
    print("-" * 107)
    print(f"[i] dan dau I-D: {best_m} = {id_means[best_m]:.4f}")
    return best_m


def main():
    args = parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[!] cuda khong san sang -> cpu"); device = "cpu"
    torch.manual_seed(args.seed)

    from PIL import Image
    paths = sorted(glob.glob(os.path.join(args.folder, args.glob)))
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        raise FileNotFoundError(f"khong thay anh: {os.path.join(args.folder, args.glob)}")
    print(f"[i] {len(paths)} anh, N={args.N}, substrate={args.substrate}, device={device}")
    print(f"[i] Shrinkage-IG sigma_sweep={args.sigma_sweep}  EG_K={args.eg_K}\n")

    model = load_resnet50(device)

    # --- chuan bi cho doi trong (rivals): rep layer + pool anh lam counterfactual ref ---
    rep_fn = ref_pool = None
    n_class = 1000
    if args.rivals:
        rep_fn = resnet50_penultimate(model)
        from PIL import Image as _Img
        pool_paths = paths[:min(8, len(paths))]     # dung vai anh dau lam pool CF
        ref_pool = [preprocess(_Img.open(p), size=224, device=device) for p in pool_paths]
        print(f"[i] RIVALS bat: IG2 (ig2_steps={args.ig2_steps}) / MaxEnt / FRInGe "
              f"(me_steps={args.me_steps}), CF-pool={len(ref_pool)}\n")

    metric_names = None
    acc = {}
    per_image_rows = []
    id_per_image = []
    bl_strength = {}          # {method: {"pf":[], "pb":[], "ratio":[], "shift":[]}}

    for ip, path in enumerate(paths):
        x = preprocess(Image.open(path), size=224, device=device)
        if args.target is None:
            with torch.no_grad():
                target = model(x[None]).argmax(1).item()
        else:
            target = args.target
        grad_fn = make_resnet50_gradfn(model, target, device, chunk=args.chunk, score=args.score)

        # --- DEBUG: baseline strength f(x) vs f(baseline) ---
        for m, b in collect_baselines_for_image(x, args, device, args.seed).items():
            pf, pb, ratio, shift = baseline_strength_row(model, x, b, target)
            d = bl_strength.setdefault(m, {"pf": [], "pb": [], "ratio": [], "shift": []})
            d["pf"].append(pf); d["pb"].append(pb); d["ratio"].append(ratio); d["shift"].append(shift)

        attrs = attributions_for_image(x, grad_fn, args, device, args.seed,
                                       model=model, target=target, rep_fn=rep_fn,
                                       ref_pool=ref_pool, n_class=n_class)
        if metric_names is None:
            metric_names = list(attrs.keys())
            for m in metric_names:
                acc[m] = {"ins": [], "del": [], "id": []}

        img_ids = {}
        for m, a in attrs.items():
            r = insertion_deletion(model, x, a, target, device=device,
                                   steps=args.insdel_steps, substrate=args.substrate,
                                   batch=args.chunk, score=args.score)
            acc[m]["ins"].append(r["insertion_auc"])
            acc[m]["del"].append(r["deletion_auc"])
            acc[m]["id"].append(r["id_gap"])
            img_ids[m] = r["id_gap"]
            per_image_rows.append({
                "image": os.path.basename(path), "target": target, "method": m,
                "insertion": r["insertion_auc"], "deletion": r["deletion_auc"],
                "id_gap": r["id_gap"],
            })
        id_per_image.append(img_ids)
        best_m = max(img_ids, key=img_ids.get)
        print(f"[{ip+1}/{len(paths)}] {os.path.basename(path):<24} best={best_m} ({img_ids[best_m]:.3f})")
        if args.report_every > 0 and ((ip + 1) % args.report_every == 0 or ip + 1 == len(paths)):
            _print_summary_table(metric_names, acc, id_per_image, ip + 1, bl_strength=bl_strength,
                                 title=f"TICH LUY sau {ip+1} anh")

    n_img = len(id_per_image)
    win_count = {m: 0 for m in metric_names}
    for d in id_per_image:
        win_count[max(d, key=d.get)] += 1

    print("\n" + "=" * 80)
    best_overall = _print_summary_table(metric_names, acc, id_per_image, n_img, bl_strength=bl_strength,
                                        title=f"KET QUA CUOI CUNG tren {n_img} anh")

    # ---- DEBUG: bang baseline strength f(x) vs f(baseline), trung binh tren anh ----
    if bl_strength:
        print("\n[DEBUG] baseline strength: f(x) vs f(baseline)  (softmax target, TB tren anh)")
        print(f"{'method':<20}{'p_full':>9}{'p_base':>9}{'ratio p_base/p_full':>22}{'mean|b-x|':>12}")
        print("-" * 72)
        for m in metric_names:
            if m not in bl_strength:      # EG bo qua (khong 1 baseline diem)
                continue
            d = bl_strength[m]
            mpf = sum(d["pf"]) / len(d["pf"])
            mpb = sum(d["pb"]) / len(d["pb"])
            rr = [r for r in d["ratio"] if r == r]
            mr = sum(rr) / len(rr) if rr else float("nan")
            msh = sum(d["shift"]) / len(d["shift"])
            print(f"{m:<20}{mpf:>9.4f}{mpb:>9.4f}{mr:>22.4f}{msh:>12.4f}")
        print("-" * 72)
        print("[i] ratio~1 => baseline chua xoa gi (sigma nho); ratio thap => trung tinh/lat lop (vd black OOD).\n")

    summary_rows = []
    for m in metric_names:
        im, ise = mean_se(acc[m]["ins"])
        dm, dse = mean_se(acc[m]["del"])
        idm, idse = mean_se(acc[m]["id"])
        winp = 100.0 * win_count[m] / n_img
        summary_rows.append({
            "method": m, "n": n_img,
            "insertion_mean": im, "insertion_se": ise,
            "deletion_mean": dm, "deletion_se": dse,
            "id_mean": idm, "id_se": idse, "win_rate": winp,
        })

    # ---- Paired test: Shrinkage-IG(sigma tot nhat) vs cac baseline ----
    # moc = paired_ref neu chi dinh; neu khong -> Shrinkage-IG@sigma co I-D cao nhat.
    if args.paired_ref and args.paired_ref in acc:
        ref = args.paired_ref
    else:
        shr = [m for m in metric_names if m.startswith("Shrinkage-IG")]
        ref = max(shr, key=lambda m: mean_se(acc[m]["id"])[0]) if shr else best_overall

    if ref in acc:
        stat_rows = []
        print("\n=== PAIRED TEST: {} vs cac baseline (n={} anh, ghep cap per-image) ===".format(ref, n_img))
        print("Two-sided p. diff = {} - method. Ins/I-D: diff>0 co loi cho {}; Del: diff<0 co loi.".format(ref, ref))
        for metric, key, better in [("Ins", "ins", "higher"),
                                    ("Del", "del", "lower"),
                                    ("I-D", "id", "higher")]:
            print(f"\n-- {metric} ({'cao hon tot' if better=='higher' else 'thap hon tot'}) --")
            print(f"{'vs method':<20}{'mean_diff':>12}{'t':>9}{'p(t)':>11}{'z(W)':>9}{'p(Wilcox)':>12}")
            print("-" * 73)
            a = acc[ref][key]
            for m in metric_names:
                if m == ref:
                    continue
                b = acc[m][key]
                md, t, pt = paired_t(a, b)
                W, z, pw = wilcoxon(a, b)
                print(f"{m:<20}{md:>12.4f}{t:>9.3f}{pt:>11.4g}{z:>9.3f}{pw:>12.4g}")
                stat_rows.append({
                    "ref": ref, "vs": m, "metric": metric,
                    "mean_diff": md, "t": t, "p_t": pt, "z_wilcoxon": z, "p_wilcoxon": pw,
                })
        with open("e1_image_paired.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(stat_rows[0].keys()))
            w.writeheader(); w.writerows(stat_rows)
        print("\n[i] da luu -> e1_image_paired.csv")

    with open("e1_image_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "target", "method", "insertion", "deletion", "id_gap"])
        w.writeheader(); w.writerows(per_image_rows)
    with open("e1_image_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)
    print("[i] da luu -> e1_image_results.csv (tung anh), e1_image_summary.csv (tong hop)")


if __name__ == "__main__":
    main()