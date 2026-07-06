"""
Visualize saliency map cho cac phuong phap attribution -> xuat .png

Chay (torch GPU mac dinh, tu chay lay):
    python visualize.py img/church.JPEG --method IG --baseline blur
    python visualize.py img/church.JPEG --method SBA-D --baseline pool --N 500
    python visualize.py img/church.JPEG --method PEA --baseline gaussian --rho 0.2
    python visualize.py img/church.JPEG --method all --baseline pool --N 500 --out grid.png

Method: IG, EG, SBA, SBA-D, PEA, Tube-EG, all
Baseline: black, white, noise, blur, gaussian, pool (4 cai: black/white/noise/blur)
Output: file .png gom anh goc + heatmap + overlay (neu --method all thi luoi nhieu method)
"""

import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # khong can display
import matplotlib.pyplot as plt

from pea.resnet50_gradfn import (
    load_resnet50, make_resnet50_gradfn, preprocess,
    IMAGENET_MEAN, IMAGENET_STD,
)
from pea.methods import ig_single, eg, sba, sba_d
from pea.estimator import path_ensemble_attribution
from pea.schedules import make_patch_groups


ALL_METHODS = ["IG", "EG", "SBA", "SBA-D", "PEA", "Tube-EG"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--method", type=str, default="IG",
                    choices=ALL_METHODS + ["all"])
    ap.add_argument("--baseline", type=str, default="blur",
                    choices=["black", "white", "noise", "blur", "gaussian", "pool"],
                    help="'pool' = 4 baseline (black/white/noise/blur) cho EG/SBA/SBA-D")
    ap.add_argument("--target", type=int, default=None)
    ap.add_argument("--N", type=int, default=500, help="ngan sach gradient eval/anh")
    ap.add_argument("--rho", type=float, default=0.2, help="schedule-strength cho PEA")
    ap.add_argument("--grid", type=int, default=14, help="patch grid cho PEA")
    ap.add_argument("--L", type=int, default=6)
    ap.add_argument("--sba_sigma", type=float, default=0.3)
    ap.add_argument("--sba_P", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cmap", type=str, default="jet")
    ap.add_argument("--percentile", type=float, default=99.0,
                    help="clip attribution o percentile nay de bo outlier khi to mau")
    ap.add_argument("--score", type=str, default="logit", choices=["logit", "softmax"],
                    help="score cho attribution backward")
    ap.add_argument("--out", type=str, default=None)
    return ap.parse_args()


def denorm(x):
    """(3,H,W) chuan hoa -> (H,W,3) trong [0,1] de hien thi."""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(3, 1, 1)
    img = (x * std + mean).clamp(0, 1)
    return img.permute(1, 2, 0).cpu().numpy()


def make_single_baseline(x, kind, device, seed):
    C, H, W = x.shape
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(3, 1, 1)
    if kind == "black":
        return (torch.zeros(3, H, W, device=device) - mean) / std
    if kind == "white":
        return (torch.ones(3, H, W, device=device) - mean) / std
    if kind == "noise" or kind == "gaussian":
        g = torch.Generator(device=device); g.manual_seed(seed + 7)
        return torch.randn(3, H, W, generator=g, device=device)
    if kind == "blur":
        return _blur(x)
    raise ValueError(kind)


def _blur(x, k=31):
    import torch.nn.functional as F
    C, H, W = x.shape
    coords = torch.arange(k, device=x.device).float() - k // 2
    g1d = torch.exp(-(coords ** 2) / (2 * (k / 3) ** 2)); g1d = g1d / g1d.sum()
    xb = x[None]
    xb = F.conv2d(xb, g1d.view(1, 1, 1, k).repeat(C, 1, 1, 1), padding=(0, k // 2), groups=C)
    xb = F.conv2d(xb, g1d.view(1, 1, k, 1).repeat(C, 1, 1, 1), padding=(k // 2, 0), groups=C)
    return xb[0]


def make_pool(x, device, seed):
    names = ["black", "white", "noise", "blur"]
    return torch.stack([make_single_baseline(x, n, device, seed) for n in names])


def compute_attr(method, x, args, grad_fn, device, seed):
    """Tra ve attribution (3,H,W) cho method da chon."""
    gen = torch.Generator(device=device); gen.manual_seed(seed)
    if method in ("IG",):
        b = make_single_baseline(x, args.baseline if args.baseline != "pool" else "blur", device, seed)
        return ig_single(x, b, grad_fn, T=args.N)
    if method in ("EG", "SBA", "SBA-D"):
        baselines = make_pool(x, device, seed)
        # baselines = (make_pool(x, device, seed) if args.baseline == "pool"
        #              else make_single_baseline(x, args.baseline, device, seed)[None])
        if method == "EG":
            return eg(x, baselines, grad_fn, N=args.N)
        if method == "SBA":
            return sba(x, baselines, grad_fn, N=args.N, sigma=args.sba_sigma, P=args.sba_P, gen=gen)
        if method == "SBA-D":
            return sba_d(x, baselines, grad_fn, N=args.N, gen=gen)
    if method in ("PEA", "Tube-EG"):
        C, H, W = x.shape
        baselines = make_pool(x, device, seed)
        # baselines = (make_pool(x, device, seed) if args.baseline == "pool"
        #              else make_single_baseline(x, args.baseline, device, seed)[None])
        # dam bao du path = so baseline; lap lai neu can
        P = max(baselines.shape[0], 25)
        if baselines.shape[0] < P:
            reps = (P + baselines.shape[0] - 1) // baselines.shape[0]
            baselines = baselines.repeat(reps, 1, 1, 1)[:P]
        gidx = make_patch_groups(C, H, W, grid=args.grid).to(device)
        T = max(2, args.N // P)
        phi_pea, phi_tube, _ = path_ensemble_attribution(
            x, baselines, gidx, grad_fn, n_groups=args.grid * args.grid,
            L=args.L, rho=args.rho, T=T, generator=gen, log_geometry=False,
        )
        return phi_pea if method == "PEA" else phi_tube
    raise ValueError(method)


def attr_to_map(attr, percentile):
    """(3,H,W) -> (H,W) heatmap chuan hoa [0,1], gom |attr| tren kenh."""
    m = attr.abs().sum(dim=0).cpu().numpy()   # (H,W)
    hi = np.percentile(m, percentile)
    if hi <= 0:
        hi = m.max() if m.max() > 0 else 1.0
    m = np.clip(m / hi, 0, 1)
    return m


def render(img_np, maps: dict, cmap, out_path):
    """img_np (H,W,3); maps: {name: (H,W)}. Moi method 1 hang: goc | heatmap | overlay."""
    n = len(maps)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes[None, :]
    for r, (name, m) in enumerate(maps.items()):
        axes[r, 0].imshow(img_np); axes[r, 0].set_title(f"input" if r == 0 else "")
        axes[r, 0].set_ylabel(name, fontsize=12, rotation=0, ha="right", va="center")
        axes[r, 1].imshow(m, cmap=cmap); axes[r, 1].set_title("saliency" if r == 0 else "")
        axes[r, 2].imshow(img_np); axes[r, 2].imshow(m, cmap=cmap, alpha=0.5)
        axes[r, 2].set_title("overlay" if r == 0 else "")
        for c in range(3):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[i] da luu -> {out_path}")


def main():
    args = parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[!] cuda khong san sang -> cpu"); device = "cpu"
    torch.manual_seed(args.seed)

    from PIL import Image
    model = load_resnet50(device)
    x = preprocess(Image.open(args.image), size=224, device=device)

    if args.target is None:
        with torch.no_grad():
            target = model(x[None]).argmax(1).item()
        print(f"[i] target top-1 = {target}")
    else:
        target = args.target
        print(f"[i] target = {target}")

    grad_fn = make_resnet50_gradfn(model, target, device, chunk=args.chunk, score=args.score)
    img_np = denorm(x)

    methods = ALL_METHODS if args.method == "all" else [args.method]
    maps = {}
    for mth in methods:
        print(f"[i] tinh {mth} (baseline={args.baseline}, N={args.N})...")
        attr = compute_attr(mth, x, args, grad_fn, device, args.seed)
        maps[mth] = attr_to_map(attr, args.percentile)

    out = args.out or f"saliency_{args.method}_{args.baseline}.png"
    render(img_np, maps, args.cmap, out)


if __name__ == "__main__":
    main()
