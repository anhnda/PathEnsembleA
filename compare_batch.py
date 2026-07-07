"""
Trung binh ket qua tren mot folder anh benchmark.

Chay (torch GPU mac dinh, tu chay lay):
    python compare_batch.py benchmark_50 --N 500 --chunk 16
    python compare_batch.py benchmark_50 --N 500 --substrate black --glob '*.JPEG'

Voi moi anh: chay IG(4 baseline)/EG/SBA/SBA-D/BlurLIG duoi cung ngan sach N, do insertion/deletion/I-D.
Sau do in mean +- SE tren toan bo anh + win-rate (ty le anh method thang I-D cao nhat).
Luu chi tiet tung anh -> results.csv va tong hop -> summary.csv.
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


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="thu muc chua anh, vd benchmark_50")
    ap.add_argument("--glob", type=str, default="*.JPEG", help="mau ten file")
    ap.add_argument("--target", type=int, default=None, help="ep target chung; mac dinh top-1 moi anh")
    ap.add_argument("--N", type=int, default=500)
    ap.add_argument("--sba_sigma", type=float, default=0.3)
    ap.add_argument("--sba_P", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--insdel_steps", type=int, default=224)
    ap.add_argument("--substrate", type=str, default="blur", choices=["blur", "black"])
    ap.add_argument("--score", type=str, default="logit", choices=["logit", "softmax"],
                    help="dung CHUNG cho attribution backward va metric")
    ap.add_argument("--limit", type=int, default=None, help="chi chay N anh dau (debug)")
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
    """Tra ve dict {method_name: attr(3,H,W)}."""
    gen = torch.Generator(device=device); gen.manual_seed(seed)
    names, baselines = make_baselines(x, device, seed)
    blur_baseline = baselines[-1]
    out = {}
    for nm, b in zip(names, baselines):
        out[f"IG-{nm}"] = ig_single(x, b, grad_fn, T=args.N)
    out["EG"] = eg(x, baselines, grad_fn, N=args.N)
    out["SBA"] = sba(x, baselines, grad_fn, N=args.N, sigma=args.sba_sigma, P=args.sba_P, gen=gen)
    out["SBA-D"] = sba_d(x, baselines, grad_fn, N=args.N, gen=gen)
    out["BlurLIG"] = blur_lig(x, blur_baseline, grad_fn, N=args.N,
                              model=model, target=target, score=args.score)
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

    # ---- tong hop ----
    print("\n=== TRUNG BINH TREN {} ANH (mean +- SE) ===".format(len(paths)))
    print(f"{'method':<12}{'insertion↑':>18}{'deletion↓':>18}{'I-D↑':>18}{'win%':>8}")
    print("-" * 74)

    # win-rate
    win_count = {m: 0 for m in metric_names}
    for d in id_per_image:
        win_count[max(d, key=d.get)] += 1
    n_img = len(id_per_image)

    summary_rows = []
    for m in metric_names:
        im, ise = mean_se(acc[m]["ins"])
        dm, dse = mean_se(acc[m]["del"])
        idm, idse = mean_se(acc[m]["id"])
        winp = 100.0 * win_count[m] / n_img
        print(f"{m:<12}{im:>10.4f}±{ise:<6.4f}{dm:>10.4f}±{dse:<6.4f}"
              f"{idm:>10.4f}±{idse:<6.4f}{winp:>7.1f}%")
        summary_rows.append({
            "method": m, "n": n_img,
            "insertion_mean": im, "insertion_se": ise,
            "deletion_mean": dm, "deletion_se": dse,
            "id_mean": idm, "id_se": idse, "win_rate": winp,
        })

    best = max(summary_rows, key=lambda r: r["id_mean"])
    print("-" * 74)
    print(f"[i] best I-D trung binh: {best['method']} = {best['id_mean']:.4f} ± {best['id_se']:.4f}")

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