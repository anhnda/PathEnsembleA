"""
Trung binh ket qua tren mot folder anh benchmark.

Chay (torch GPU mac dinh, tu chay lay):
    python compare_batch.py benchmark_50 --N 500 --chunk 16
    python compare_batch.py benchmark_50 --N 500 --substrate black --glob '*.JPEG'

Voi moi anh: chay IG(4 baseline)/EG/SBA/SBA-D/BlurLIG/BlurLIG-Full/PEA/Tube-EG duoi
cung ngan sach N, do insertion/deletion/I-D. Sau MOI anh in bang thong ke TICH LUY
(running mean±SE + win%) — theo doi truc tiep ai dang dan dau khi so anh tang.
Cuoi cung in bang tong + paired test (Wilcoxon/paired-t) BlurLIG vs cac method.
Luu results.csv (tung anh), summary.csv (tong hop), paired_tests.csv.

Luu y: BlurLIG-Full rat dat (O(T*N*N) eval/anh); tat bang --no_lig_full neu chi can
so cac method re. Khong train, khong smoketest.
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
from pea.methods import ig_single, eg, sba, sba_d
from pea.blur_lig import blur_lig
from pea.blur_lig_full import blur_lig_full, make_fvals_fn
from pea.diffusion_path import diffusion_ig, diffusion_pf, diffusion_ig_multiref
from pea.estimator import path_ensemble_attribution
from pea.schedules import make_patch_groups


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="thu muc chua anh, vd benchmark_50")
    ap.add_argument("--glob", type=str, default="*.JPEG", help="mau ten file")
    ap.add_argument("--target", type=int, default=None, help="ep target chung; mac dinh top-1 moi anh")
    ap.add_argument("--N", type=int, default=500)
    ap.add_argument("--sba_sigma", type=float, default=0.3)
    ap.add_argument("--sba_P", type=int, default=4)
    ap.add_argument("--rho", type=float, default=0.2, help="schedule-strength cho PEA")
    ap.add_argument("--grid", type=int, default=14, help="patch grid cho PEA / group cho LIG-Full")
    ap.add_argument("--L", type=int, default=6, help="so mode cosine cho PEA")
    ap.add_argument("--pea_P", type=int, default=25, help="so path cho PEA/Tube-EG")
    ap.add_argument("--lig_T", type=int, default=10, help="so vong alternating cho BlurLIG-Full")
    ap.add_argument("--lig_N", type=int, default=50, help="so buoc path cho BlurLIG-Full (nho vi Phase2 dat)")
    ap.add_argument("--lig_exact_mu", action="store_true", help="Phase1 dung mu∝|d_k| thay vi QP")
    ap.add_argument("--lig_init", type=str, default="blurlig", choices=["blurlig", "straight"],
                    help="khoi tao BlurLIG-Full: 'blurlig' / 'straight'")
    ap.add_argument("--no_lig_full", action="store_true",
                    help="BO BlurLIG-Full (rat dat: O(T*N*N) eval/anh)")
    # --- Diffusion-path (VP-SDE forward analytic + PF-ODE-style) ---
    ap.add_argument("--diff_beta_min", type=float, default=0.1, help="beta_min lich VP cho Diffusion")
    ap.add_argument("--diff_beta_max", type=float, default=20.0, help="beta_max lich VP cho Diffusion")
    ap.add_argument("--diff_P", type=int, default=4, help="so lich nhieu lay ky vong cho Diffusion-IG")
    ap.add_argument("--diff_jitter", type=float, default=0.02, help="lech luoi thoi gian giua cac lich (Diffusion-IG)")
    ap.add_argument("--diff_score_scale", type=float, default=0.15, help="cuong do score-proxy de-blur cho Diffusion-PF (0 => VP thuan)")
    ap.add_argument("--diff_no_lig", action="store_true", help="Diffusion-PF dung uniform Ito thay vi LIG-measure")
    ap.add_argument("--no_diffusion", action="store_true", help="BO ca hai method Diffusion")
    ap.add_argument("--diff_R", type=int, default=4, help="so blur reference (nhiet do) cho Diffusion-MultiRef")
    ap.add_argument("--diff_sigma_min", type=float, default=1.0, help="blur nhe nhat (pixel) cho Diffusion-MultiRef")
    ap.add_argument("--diff_sigma_max", type=float, default=25.0, help="blur nang nhat (pixel) cho Diffusion-MultiRef")
    ap.add_argument("--no_multiref", action="store_true", help="BO Diffusion-MultiRef")
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--insdel_steps", type=int, default=224)
    ap.add_argument("--substrate", type=str, default="blur", choices=["blur", "black"])
    ap.add_argument("--score", type=str, default="logit", choices=["logit", "softmax"],
                    help="dung CHUNG cho attribution backward va metric")
    ap.add_argument("--limit", type=int, default=None, help="chi chay N anh dau (debug)")
    ap.add_argument("--report_every", type=int, default=1,
                    help="in bang thong ke tich luy moi bao nhieu anh (0 = tat, chi in cuoi)")
    return ap.parse_args()


def make_baselines(x, device, seed):
    C, H, W = x.shape
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(3, 1, 1)
    black = (torch.zeros(3, H, W, device=device) - mean) / std
    white = (torch.ones(3, H, W, device=device) - mean) / std
    g = torch.Generator(device=device); g.manual_seed(seed + 7)
    noise = torch.randn(3, H, W, generator=g, device=device)
    import torch.nn.functional as F
    k = 31; coords = torch.arange(k, device=device).float() - k // 2
    g1d = torch.exp(-(coords ** 2) / (2 * (k / 3) ** 2)); g1d = g1d / g1d.sum()
    xb = x[None]
    xb = F.conv2d(xb, g1d.view(1, 1, 1, k).repeat(C, 1, 1, 1), padding=(0, k // 2), groups=C)
    xb = F.conv2d(xb, g1d.view(1, 1, k, 1).repeat(C, 1, 1, 1), padding=(k // 2, 0), groups=C)
    blur = xb[0]
    return ["black", "white", "noise", "blur"], torch.stack([black, white, noise, blur])


def attributions_for_image(x, grad_fn, args, device, seed, model, target):
    """Tra ve dict {method_name: attr(3,H,W)}. Day du nhu compare.py."""
    gen = torch.Generator(device=device); gen.manual_seed(seed)
    names, baselines = make_baselines(x, device, seed)
    blur_baseline = baselines[-1]
    C, H, W = x.shape
    out = {}
    for nm, b in zip(names, baselines):
        out[f"IG-{nm}"] = ig_single(x, b, grad_fn, T=args.N)
    out["EG"] = eg(x, baselines, grad_fn, N=args.N)
    out["SBA"] = sba(x, baselines, grad_fn, N=args.N, sigma=args.sba_sigma, P=args.sba_P, gen=gen)
    out["SBA-D"] = sba_d(x, baselines, grad_fn, N=args.N, gen=gen)
    out["BlurLIG"] = blur_lig(x, blur_baseline, grad_fn, N=args.N,
                              model=model, target=target, score=args.score)

    # Diffusion-IG: VP-SDE ANALYTIC de-noising path (blur->x), lightweight, ky vong tren P lich nhieu.
    # Diffusion-PF: PF-ODE-style path DETERMINISTIC (score-proxy = de-blur, blur=heat) + LIG-measure.
    if not args.no_diffusion:
        out["Diffusion-IG"] = diffusion_ig(
            x, blur_baseline, grad_fn, N=args.N,
            beta_min=args.diff_beta_min, beta_max=args.diff_beta_max,
            P=args.diff_P, jitter=args.diff_jitter, gen=gen,
        )
        out["Diffusion-PF"] = diffusion_pf(
            x, blur_baseline, grad_fn, N=args.N,
            beta_min=args.diff_beta_min, beta_max=args.diff_beta_max,
            score_scale=args.diff_score_scale, use_lig=not args.diff_no_lig,
            model=model, target=target, score=args.score,
        )

    # Diffusion-MultiRef: HO blur reference theo nhiet do (tan cong truc BASELINE).
    if not args.no_diffusion and not args.no_multiref:
        out["Diffusion-MultiRef"] = diffusion_ig_multiref(
            x, grad_fn, N=args.N,
            R=args.diff_R, sigma_min=args.diff_sigma_min, sigma_max=args.diff_sigma_max,
            beta_min=args.diff_beta_min, beta_max=args.diff_beta_max,
            model=model, target=target, score=args.score,
        )

    # BlurLIG-Full (Algorithm 1 tren blur reference). Rat dat -> co the tat bang --no_lig_full.
    if not args.no_lig_full:
        gidx_lig = make_patch_groups(C, H, W, grid=args.grid).to(device)
        fvals_fn = make_fvals_fn(model, target, device, chunk=args.chunk, score=args.score)
        out["BlurLIG-Full"] = blur_lig_full(
            x, blur_baseline, grad_fn, fvals_fn,
            group_index=gidx_lig, G=args.grid * args.grid, N=args.lig_N,
            T=args.lig_T, lam=1.0, tau=0.01,
            use_exact_measure=args.lig_exact_mu, init=args.lig_init, generator=gen,
            model=model, target=target, score=args.score,
        )

    # PEA + Tube-EG (cung pool baseline blur, cung ngan sach N)
    P = args.pea_P
    reps = (P + baselines.shape[0] - 1) // baselines.shape[0]
    pea_baselines = blur_baseline.repeat(reps, 1, 1, 1)[:P]
    gidx = make_patch_groups(C, H, W, grid=args.grid).to(device)
    T_pea = max(2, args.N // P)
    phi_pea, phi_tube, _ = path_ensemble_attribution(
        x, pea_baselines, gidx, grad_fn, n_groups=args.grid * args.grid,
        L=args.L, rho=args.rho, T=T_pea, generator=gen, log_geometry=False,
    )
    out["PEA"] = phi_pea
    out["Tube-EG"] = phi_tube
    return out


def mean_se(vals):
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(vals) / n
    if n == 1:
        return m, 0.0
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    return m, math.sqrt(var / n)  # standard error


# ---------------------------------------------------------------------------
# Paired stats tu luc — khong phu thuoc scipy. Tinh tren hieu so per-image
# diff_k = a_k - b_k (a=BlurLIG, b=baseline), da ghep cap dung thu tu anh.
# ---------------------------------------------------------------------------
def _norm_cdf(z):
    """CDF chuan tac qua erf."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _t_sf(t, df):
    """P(T>|t|)*2 (two-sided) cho phan phoi Student-t, xap xi.
    Dung bien doi sang normal khi df lon; voi df nho dung cong thuc chuoi don gian.
    Tra ve p-value hai phia."""
    if df <= 0 or not math.isfinite(t):
        return float("nan")
    x = df / (df + t * t)
    # regularized incomplete beta I_x(df/2, 1/2) = two-sided tail cua t (Abramowitz-Stegun tinh than)
    # dung xap xi lien phan qua betacf; du chinh xac de bao cao p-value.
    a, b = df / 2.0, 0.5
    bt = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log(1.0 - x)
    ) if 0.0 < x < 1.0 else 0.0
    # lien phan Lentz cho I_x(a,b)
    def _betacf(x, a, b):
        MAXIT, EPS, FPMIN = 200, 3e-12, 1e-300
        qab, qap, qam = a + b, a + 1.0, a - 1.0
        c = 1.0
        d = 1.0 - qab * x / qap
        if abs(d) < FPMIN: d = FPMIN
        d = 1.0 / d
        h = d
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
    return max(0.0, min(1.0, ix))  # = two-sided p cua t


def paired_t(a, b):
    """Paired t-test tren diff = a-b. Tra ve (mean_diff, t, p_two_sided)."""
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
    """Wilcoxon signed-rank tren diff=a-b, normal approx co hieu chinh ties + zeros.
    Tra ve (W, z, p_two_sided). Loai cac diff==0 (Pratt bo qua)."""
    d = [ai - bi for ai, bi in zip(a, b) if ai - bi != 0.0]
    n = len(d)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    order = sorted(range(n), key=lambda i: abs(d[i]))
    # gan rank co xu ly ties (trung binh rank)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        avg = (i + 1 + j + 1) / 2.0  # rank 1-based trung binh
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    Wp = sum(ranks[i] for i in range(n) if d[i] > 0)
    Wm = sum(ranks[i] for i in range(n) if d[i] < 0)
    W = min(Wp, Wm)
    mu = n * (n + 1) / 4.0
    # hieu chinh ties cho phuong sai
    from collections import Counter
    tie_counts = Counter(abs(x) for x in d)
    tie_term = sum(t ** 3 - t for t in tie_counts.values())
    sigma2 = n * (n + 1) * (2 * n + 1) / 24.0 - tie_term / 48.0
    if sigma2 <= 0:
        return (W, float("nan"), float("nan"))
    z = (W - mu + 0.5 * (1 if W < mu else -1)) / math.sqrt(sigma2)  # continuity correction
    p = 2.0 * _norm_cdf(-abs(z))
    return (W, z, p)


def _print_summary_table(metric_names, acc, id_per_image, n_img, title):
    """In bang mean±SE (ins/del/id) + win% tren n_img anh da chay. Dung cho ca running lan final."""
    win_count = {m: 0 for m in metric_names}
    for d in id_per_image:
        win_count[max(d, key=d.get)] += 1
    print(f"\n--- {title} (mean±SE tren {n_img} anh) ---")
    print(f"{'method':<14}{'insertion↑':>16}{'deletion↓':>16}{'I-D↑':>16}{'win%':>8}")
    print("-" * 70)
    best_m, best_id = None, -float("inf")
    for m in metric_names:
        im, ise = mean_se(acc[m]["ins"])
        dm, dse = mean_se(acc[m]["del"])
        idm, idse = mean_se(acc[m]["id"])
        winp = 100.0 * win_count[m] / n_img if n_img else 0.0
        star = ""
        if idm > best_id:
            best_id, best_m = idm, m
        print(f"{m:<14}{im:>8.4f}±{ise:<6.4f}{dm:>8.4f}±{dse:<6.4f}{idm:>8.4f}±{idse:<6.4f}{winp:>7.1f}%")
    print("-" * 70)
    print(f"[i] dan dau I-D: {best_m} = {best_id:.4f}")


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

    model = load_resnet50(device)

    # tich luy: {method: {'ins': [...], 'del': [...], 'id': [...]}}
    metric_names = None
    acc = {}
    per_image_rows = []  # cho results.csv
    id_per_image = []    # de tinh win-rate: list dict {method: id_gap}

    for ip, path in enumerate(paths):
        x = preprocess(Image.open(path), size=224, device=device)
        if args.target is None:
            with torch.no_grad():
                target = model(x[None]).argmax(1).item()
        else:
            target = args.target
        grad_fn = make_resnet50_gradfn(model, target, device, chunk=args.chunk, score=args.score)

        attrs = attributions_for_image(x, grad_fn, args, device, args.seed, model, target)
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
        # ---- thong ke TICH LUY sau moi anh (running mean±SE + win%) ----
        if args.report_every > 0 and ((ip + 1) % args.report_every == 0 or ip + 1 == len(paths)):
            _print_summary_table(metric_names, acc, id_per_image, ip + 1,
                                 title=f"TICH LUY sau {ip+1} anh")

    # ---- tong hop CUOI CUNG ----
    n_img = len(id_per_image)
    win_count = {m: 0 for m in metric_names}
    for d in id_per_image:
        win_count[max(d, key=d.get)] += 1

    print("\n" + "=" * 74)
    _print_summary_table(metric_names, acc, id_per_image, n_img,
                         title=f"KET QUA CUOI CUNG tren {n_img} anh")

    # build summary_rows cho CSV (khong in lai, da in o bang tren)
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

    best = max(summary_rows, key=lambda r: r["id_mean"])

    # ---- Paired test: BlurLIG vs cac method con lai (tren hieu so per-image) ----
    ref = "BlurLIG"
    if ref in acc:
        stat_rows = []
        print("\n=== PAIRED TEST: {} vs cac method (n={} anh, ghep cap per-image) ===".format(ref, n_img))
        print("Two-sided p. diff = {} - method. Ins/I-D: diff>0 co loi cho {}; Del: diff<0 co loi.".format(ref, ref))
        for metric, key, better in [("Ins", "ins", "higher"),
                                    ("Del", "del", "lower"),
                                    ("I-D", "id", "higher")]:
            print(f"\n-- {metric} ({'cao hon tot' if better=='higher' else 'thap hon tot'}) --")
            print(f"{'vs method':<12}{'mean_diff':>12}{'t':>9}{'p(t)':>11}{'z(W)':>9}{'p(Wilcox)':>12}")
            print("-" * 65)
            a = acc[ref][key]
            for m in metric_names:
                if m == ref:
                    continue
                b = acc[m][key]
                md, t, pt = paired_t(a, b)
                W, z, pw = wilcoxon(a, b)
                print(f"{m:<12}{md:>12.4f}{t:>9.3f}{pt:>11.4g}{z:>9.3f}{pw:>12.4g}")
                stat_rows.append({
                    "ref": ref, "vs": m, "metric": metric,
                    "mean_diff": md, "t": t, "p_t": pt, "z_wilcoxon": z, "p_wilcoxon": pw,
                })
        with open("paired_tests.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(stat_rows[0].keys()))
            w.writeheader(); w.writerows(stat_rows)
        print("\n[i] da luu -> paired_tests.csv")

    with open("results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "target", "method", "insertion", "deletion", "id_gap"])
        w.writeheader(); w.writerows(per_image_rows)
    with open("summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)
    print("[i] da luu -> results.csv (tung anh), summary.csv (tong hop)")


if __name__ == "__main__":
    main()