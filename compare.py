"""
So sanh cac method duoi cung ngan sach gradient N.

Chay:
    python compare.py img/church.JPEG --N 500 --chunk 16
    python compare.py img/church.JPEG --N 500 --target 497

Methods:
    IG-black / IG-white / IG-noise / IG-blur   (moi cai 1 baseline, T=N buoc)
    EG        (pool 4 baseline, N chia deu)
    SBA       (Brownian bridge + Ito, N chia deu)
    SBA-D     (barycentric path + Ito, Def 4/5)

In insertion / deletion / I-D cho tung method. Khong train, khong smoketest.
"""

import argparse
import torch

from pea.resnet50_gradfn import load_resnet50, make_resnet50_gradfn, preprocess, IMAGENET_MEAN, IMAGENET_STD
from pea.insdel import insertion_deletion
from pea.methods import ig_single, eg, sba, sba_d
from pea.estimator import path_ensemble_attribution
from pea.schedules import make_patch_groups


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--target", type=int, default=None)
    ap.add_argument("--N", type=int, default=500, help="ngan sach: so gradient eval/anh")
    ap.add_argument("--sba_sigma", type=float, default=0.3, help="do rong Brownian bridge cho SBA")
    ap.add_argument("--sba_P", type=int, default=4, help="so trajectory/baseline cho SBA")
    ap.add_argument("--rho", type=float, default=0.2, help="schedule-strength cho PEA")
    ap.add_argument("--grid", type=int, default=14, help="patch grid cho PEA")
    ap.add_argument("--L", type=int, default=6, help="so mode cosine cho PEA")
    ap.add_argument("--pea_P", type=int, default=25, help="so path cho PEA/Tube-EG")
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--insdel_steps", type=int, default=224)
    ap.add_argument("--substrate", type=str, default="blur", choices=["blur", "black"])
    ap.add_argument("--score", type=str, default="logit", choices=["logit", "softmax"],
                    help="dung CHUNG cho attribution backward va metric")
    return ap.parse_args()


def make_baselines(x, device, seed):
    """4 baseline: black, white, noise, blur — deu trong khong gian da chuan hoa."""
    C, H, W = x.shape
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(3, 1, 1)
    black = (torch.zeros(3, H, W, device=device) - mean) / std
    white = (torch.ones(3, H, W, device=device) - mean) / std
    g = torch.Generator(device=device); g.manual_seed(seed + 7)
    noise = torch.randn(3, H, W, generator=g, device=device)
    # blur cua chinh anh
    import torch.nn.functional as F
    k = 31; coords = torch.arange(k, device=device).float() - k // 2
    g1d = torch.exp(-(coords ** 2) / (2 * (k / 3) ** 2)); g1d = g1d / g1d.sum()
    xb = x[None]
    xb = F.conv2d(xb, g1d.view(1, 1, 1, k).repeat(C, 1, 1, 1), padding=(0, k // 2), groups=C)
    xb = F.conv2d(xb, g1d.view(1, 1, k, 1).repeat(C, 1, 1, 1), padding=(k // 2, 0), groups=C)
    blur = xb[0]
    names = ["black", "white", "noise", "blur"]
    return names, torch.stack([black, white, noise, blur])


def main():
    args = parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[!] cuda khong san sang -> cpu"); device = "cpu"
    torch.manual_seed(args.seed)
    gen = torch.Generator(device=device); gen.manual_seed(args.seed)

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
    names, baselines = make_baselines(x, device, args.seed)
    blur_baseline = baselines[-1]
    N = args.N
    print(f"[i] ngan sach N={N} gradient eval/anh, device={device}\n")

    attrs = {}
    # IG cho tung baseline (moi cai N buoc)
    for nm, b in zip(names, baselines):
        attrs[f"IG-{nm}"] = ig_single(x, b, grad_fn, T=N)
    # EG pool 4 baseline
    attrs["EG"] = eg(x, baselines, grad_fn, N=N)
    # SBA Brownian bridge + Ito
    attrs["SBA"] = sba(x, blur_baseline, grad_fn, N=N, sigma=args.sba_sigma, P=args.sba_P, gen=gen)
    # SBA-D barycentric + Ito
    attrs["SBA-D"] = sba_d(x, blur_baseline, grad_fn, N=N, gen=gen)

    # PEA + Tube-EG (cung pool baseline, cung ngan sach N)
    # P path = pea_P; lap pool baseline cho du P; T = N // P de tong grad ~ N
    C, H, W = x.shape
    P = args.pea_P
    reps = (P + baselines.shape[0] - 1) // baselines.shape[0]
    pea_baselines = blur_baseline.repeat(reps, 1, 1, 1)[:P]
    gidx = make_patch_groups(C, H, W, grid=args.grid).to(device)
    T_pea = max(2, N // P)
    phi_pea, phi_tube, _ = path_ensemble_attribution(
        x, pea_baselines, gidx, grad_fn, n_groups=args.grid * args.grid,
        L=args.L, rho=args.rho, T=T_pea, generator=gen, log_geometry=False,
    )
    attrs["PEA"] = phi_pea
    attrs["Tube-EG"] = phi_tube

    # insertion / deletion cho tung attribution
    print(f"{'method':<12}{'insertion↑':>12}{'deletion↓':>12}{'I-D↑':>10}")
    print("-" * 44)
    rows = {}
    for nm, a in attrs.items():
        r = insertion_deletion(model, x, a, target, device=device,
                               steps=args.insdel_steps, substrate=args.substrate,
                               batch=args.chunk, score=args.score)
        rows[nm] = r
        print(f"{nm:<12}{r['insertion_auc']:>12.4f}{r['deletion_auc']:>12.4f}{r['id_gap']:>10.4f}")

    # danh dau best I-D
    best = max(rows, key=lambda k: rows[k]['id_gap'])
    print("-" * 44)
    print(f"[i] best I-D: {best} = {rows[best]['id_gap']:.4f}")

    torch.save({"target": target, "N": N,
                "rows": {k: {kk: vv for kk, vv in v.items() if not kk.endswith('curve')}
                         for k, v in rows.items()}},
               "compare.pt")
    print("[i] da luu -> compare.pt")


if __name__ == "__main__":
    main()
