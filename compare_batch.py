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

    with open("results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "target", "method", "insertion", "deletion", "id_gap"])
        w.writeheader(); w.writerows(per_image_rows)
    with open("summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)
    print("[i] da luu -> results.csv (tung anh), summary.csv (tong hop)")


if __name__ == "__main__":
    main()