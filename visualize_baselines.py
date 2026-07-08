"""
Visualize IG voi NHIEU BASELINE khac nhau — moi baseline mot HANG, 5 cot:

    [ input ] [ baseline reference ] [ IG heatmap ] [ overlay ] [ Measure (ins/del) ]

Cot 1 (input)     : anh goc (lap lai moi hang cho de doi chieu).
Cot 2 (baseline)  : anh baseline reference dung cho IG (black/white/noise/blur/mean/
                    shrinkage-FFT@sigma...). Hien thi dang anh de thay "diem xuat phat".
Cot 3 (IG heatmap): saliency |IG| chuan hoa, to mau cmap.
Cot 4 (overlay)   : heatmap chong len anh goc.
Cot 5 (Measure)   : duong insertion/deletion (insdel.py) + so AUC + I-D gap.

Baseline ho tro: black, white, noise, blur, mean, shrink@<sigma> (Wiener low-pass FFT).
Mac dinh chay mot bo tieu chuan. Co the chi dinh --baselines.

Chay (torch GPU mac dinh, tu chay lay):
    python visualize_baselines.py img/church.JPEG
    python visualize_baselines.py img/church.JPEG --baselines black blur mean shrink@4 shrink@16
    python visualize_baselines.py img/church.JPEG --N 500 --substrate black --out grid.png

KHONG train, KHONG smoketest.
"""

import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pea.resnet50_gradfn import (
    load_resnet50, make_resnet50_gradfn, preprocess,
    IMAGENET_MEAN, IMAGENET_STD,
)
from pea.methods import ig_single
from pea.insdel import insertion_deletion
from pea.spectral_reference import spectral_reference_fft

# tai dung tien ich tu visualize.py
from visualize import denorm, attr_to_map, _blur


DEFAULT_BASELINES = ["black", "white", "noise", "blur", "mean", "shrink@4", "shrink@16"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--baselines", type=str, nargs="+", default=DEFAULT_BASELINES,
                    help="danh sach baseline: black white noise blur mean shrink@<sigma>")
    ap.add_argument("--target", type=int, default=None)
    ap.add_argument("--N", type=int, default=500, help="ngan sach gradient eval/anh cho IG")
    ap.add_argument("--insdel_steps", type=int, default=224)
    ap.add_argument("--substrate", type=str, default="blur", choices=["blur", "black"],
                    help="nen cho insertion / gia tri xoa deletion (doc lap voi baseline)")
    ap.add_argument("--score", type=str, default="logit", choices=["logit", "softmax"])
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cmap", type=str, default="jet")
    ap.add_argument("--percentile", type=float, default=99.0)
    ap.add_argument("--out", type=str, default=None)
    return ap.parse_args()


def make_baseline(spec, x, device, seed):
    """Tra ve (baseline_tensor (3,H,W), nhan_hien_thi)."""
    C, H, W = x.shape
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(3, 1, 1)
    if spec == "black":
        return (torch.zeros(3, H, W, device=device) - mean) / std, "black"
    if spec == "white":
        return (torch.ones(3, H, W, device=device) - mean) / std, "white"
    if spec == "noise":
        g = torch.Generator(device=device); g.manual_seed(seed + 7)
        return torch.randn(3, H, W, generator=g, device=device), "noise"
    if spec == "blur":
        return _blur(x), "blur"
    if spec == "mean":
        return torch.zeros(3, H, W, device=device), "mean(0)"   # 0 trong khong gian chuan hoa
    if spec.startswith("shrink@"):
        sigma = float(spec.split("@")[1])
        return spectral_reference_fft(x, sigma=sigma), f"shrink σ={sigma:g}"
    raise ValueError(f"baseline khong ro: {spec}")


def measure_panel(ax, insdel_res):
    """Ve duong insertion/deletion + chu thich AUC/I-D len 1 axes."""
    ins_curve = insdel_res.get("insertion_curve")
    del_curve = insdel_res.get("deletion_curve")
    ins_auc = insdel_res["insertion_auc"]
    del_auc = insdel_res["deletion_auc"]
    idg = insdel_res["id_gap"]
    if ins_curve is not None and del_curve is not None:
        xs_i = np.linspace(0, 1, len(ins_curve))
        xs_d = np.linspace(0, 1, len(del_curve))
        ax.plot(xs_i, ins_curve, color="tab:green", lw=1.5, label="insertion")
        ax.plot(xs_d, del_curve, color="tab:red", lw=1.5, label="deletion")
        ax.fill_between(xs_i, ins_curve, alpha=0.12, color="tab:green")
        ax.set_ylim(-0.02, 1.02); ax.set_xlim(0, 1)
        ax.legend(fontsize=6, loc="center right", framealpha=0.5)
    # so lieu
    txt = f"ins={ins_auc:.3f}\ndel={del_auc:.3f}\nI-D={idg:+.3f}"
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, fontsize=12,
            va="top", ha="left",
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.tick_params(labelsize=6)


def render(img_np, rows, cmap, out_path):
    """
    rows: list dict {name, baseline_np, heat, insdel}. 5 cot:
      input | baseline | IG heatmap | overlay | Measure.
    """
    n = len(rows)
    fig, axes = plt.subplots(n, 5, figsize=(15, 3 * n))
    if n == 1:
        axes = axes[None, :]
    col_titles = ["input", "baseline reference", "IG heatmap", "overlay", "Measure (ins/del)"]
    for r, row in enumerate(rows):
        name = row["name"]
        axes[r, 0].imshow(img_np)
        axes[r, 0].set_ylabel(name, fontsize=12, rotation=0, ha="right", va="center")
        axes[r, 1].imshow(np.clip(row["baseline_np"], 0, 1))
        axes[r, 2].imshow(row["heat"], cmap=cmap)
        axes[r, 3].imshow(img_np); axes[r, 3].imshow(row["heat"], cmap=cmap, alpha=0.5)
        measure_panel(axes[r, 4], row["insdel"])
        for c in range(4):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        if r == 0:
            for c, t in enumerate(col_titles):
                axes[r, c].set_title(t, fontsize=11)
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

    rows = []
    for spec in args.baselines:
        b, label = make_baseline(spec, x, device, args.seed)
        print(f"[i] IG voi baseline='{label}' (N={args.N})...")
        attr = ig_single(x, b, grad_fn, T=args.N)
        heat = attr_to_map(attr, args.percentile)
        insdel = insertion_deletion(model, x, attr, target, device=device,
                                    steps=args.insdel_steps, substrate=args.substrate,
                                    batch=args.chunk, score=args.score)
        rows.append({
            "name": label,
            "baseline_np": denorm(b),
            "heat": heat,
            "insdel": insdel,
        })
        print(f"    ins={insdel['insertion_auc']:.4f}  del={insdel['deletion_auc']:.4f}  "
              f"I-D={insdel['id_gap']:+.4f}")

    out = args.out or "visualize_baselines.png"
    render(img_np, rows, args.cmap, out)


if __name__ == "__main__":
    main()