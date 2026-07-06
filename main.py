"""
Path-Ensemble Attribution (monotone) tren ResNet-50. Mac dinh cuda.

Chay:
    python main.py                      # dung anh mau tu torchvision (grace hopper)
    python main.py /duong/dan/anh.jpg   # anh cua may
    python main.py anh.jpg --target 285 # ep target class
    python main.py anh.jpg --P 50 --T 50 --rho 0.5

Chi tinh attribution + in geometry log. KHONG train, KHONG benchmark.
"""

import sys
import argparse
import torch

from pea import path_ensemble_attribution, make_patch_groups
from pea.resnet50_gradfn import (
    load_resnet50,
    make_resnet50_gradfn,
    preprocess,
    black_baseline,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", default=None, help="duong dan anh; bo trong -> anh mau")
    ap.add_argument("--target", type=int, default=None, help="lop ImageNet; mac dinh = top-1 du doan")
    ap.add_argument("--P", type=int, default=25, help="so path (= so baseline)")
    ap.add_argument("--T", type=int, default=25, help="so buoc midpoint")
    ap.add_argument("--L", type=int, default=6, help="so mode cosine")
    ap.add_argument("--rho", type=float, default=0.5, help="schedule-strength (0..1)")
    ap.add_argument("--grid", type=int, default=14, help="grid patch cho schedule group")
    ap.add_argument("--chunk", type=int, default=32, help="so anh moi mini-batch qua model (giam neu OOM)")
    ap.add_argument("--baseline", type=str, default="black",
                    choices=["black", "gaussian", "blur"],
                    help="loai baseline; black hay bao hoa manh -> completeness gap lon")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--antithetic", action="store_true")
    ap.add_argument("--out", type=str, default="attribution.pt", help="luu phi_pea/phi_tube")
    return ap.parse_args()


def load_image(path, device):
    from PIL import Image
    if path is None:
        # anh mau san co trong torchvision (khong can tai)
        import torchvision
        try:
            fp = torchvision.utils._download_url  # noqa: F841
        except Exception:
            pass
        # dung anh test dong goi cua torchvision neu co, khong thi bao loi ro
        import os, torchvision
        cand = os.path.join(os.path.dirname(torchvision.__file__),
                            "assets", "encode_jpeg", "grace_hopper_517x606.jpg")
        if os.path.exists(cand):
            return Image.open(cand)
        raise FileNotFoundError(
            "Khong tim thay anh mau. Truyen duong dan anh: python main.py anh.jpg"
        )
    return Image.open(path)


def main():
    args = parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[!] cuda khong san sang, doi sang cpu")
        device = "cpu"

    torch.manual_seed(args.seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)

    # model + anh
    model = load_resnet50(device)
    pil = load_image(args.image, device)
    x = preprocess(pil, size=224, device=device)  # (3, 224, 224)
    C, H, W = x.shape

    # target: mac dinh top-1
    if args.target is None:
        with torch.no_grad():
            pred = model(x[None]).argmax(dim=1).item()
        target = pred
        print(f"[i] target (top-1 du doan) = {target}")
    else:
        target = args.target
        print(f"[i] target (ep) = {target}")

    grad_fn = make_resnet50_gradfn(model, target, device, chunk=args.chunk)

    # baseline pool
    if args.baseline == "black":
        base = black_baseline((C, H, W), device)          # (3, H, W)
        baselines = base[None].repeat(args.P, 1, 1, 1)    # (P, 3, H, W)
    elif args.baseline == "gaussian":
        # nhieu gaussian quanh 0 trong khong gian da chuan hoa -> tranh diem bao hoa cua anh den
        g2 = torch.Generator(device=device); g2.manual_seed(args.seed + 1)
        baselines = torch.randn(args.P, C, H, W, generator=g2, device=device)
    elif args.baseline == "blur":
        # anh input lam mo manh lam baseline (cung pool, lap lai)
        import torch.nn.functional as F
        k = 31
        coords = torch.arange(k, device=device) - k // 2
        g1d = torch.exp(-(coords.float() ** 2) / (2 * (k / 3) ** 2))
        g1d = (g1d / g1d.sum()).view(1, 1, -1)
        xb = x[None]
        xb = F.conv2d(xb, g1d.view(1, 1, 1, k).repeat(C, 1, 1, 1), padding=(0, k // 2), groups=C)
        xb = F.conv2d(xb, g1d.view(1, 1, k, 1).repeat(C, 1, 1, 1), padding=(k // 2, 0), groups=C)
        baselines = xb.repeat(args.P, 1, 1, 1)
    else:
        raise ValueError(args.baseline)

    # group map: grid x grid patch, share RGB
    gidx = make_patch_groups(C, H, W, grid=args.grid).to(device)
    n_groups = args.grid * args.grid

    print(f"[i] chay PEA: P={args.P} T={args.T} L={args.L} rho={args.rho} "
          f"grid={args.grid} device={device}")

    phi_pea, phi_tube, geom = path_ensemble_attribution(
        x, baselines, gidx, grad_fn,
        n_groups=n_groups, L=args.L, rho=args.rho, T=args.T,
        generator=gen, antithetic=args.antithetic, log_geometry=True,
    )

    # completeness check (muc lien tuc): sum phi ~ f(x) - E[f(x0)]
    with torch.no_grad():
        fx = model(x[None])[0, target].item()
        fx0 = model(baselines)[:, target].mean().item()
    pea_sum = phi_pea.sum().item()

    # --- IG tham chieu doc lap (trapezoid, gom alpha=0 va alpha=1) de chan scale ---
    # Dung dung 1 baseline dau tien; neu code chinh dung, so nay phai xap xi fx - f(x0_0).
    x0_ref = baselines[0]
    with torch.no_grad():
        fx0_ref = model(x0_ref[None])[0, target].item()
    alphas = torch.linspace(0, 1, args.T + 1, device=device)  # gom ca 2 dau
    states_ref = x0_ref.unsqueeze(0) + alphas.view(-1, 1, 1, 1) * (x - x0_ref).unsqueeze(0)
    grads_ref = grad_fn(states_ref)  # (T+1, 3, H, W)
    # trapezoid theo alpha, roi nhan (x - x0)
    avg_grad = 0.5 * (grads_ref[:-1] + grads_ref[1:]).mean(dim=0)  # trung binh trapezoid
    ig_ref = (avg_grad * (x - x0_ref)).sum().item()
    print("\n--- IG tham chieu (trapezoid, 1 baseline) ---")
    print(f"IG_ref sum        = {ig_ref:.4f}")
    print(f"f(x) - f(x0_0)    = {fx - fx0_ref:.4f}")
    print(f"IG_ref resid      = {abs(ig_ref - (fx - fx0_ref)):.4f}  (neu nho -> loi o midpoint/PEA; neu lon -> loi o grad_fn)")

    print("\n=== KET QUA ===")
    print(f"sum(phi_PEA)      = {pea_sum:.4f}")
    print(f"f(x) - E[f(x0)]   = {fx - fx0:.4f}")
    print(f"completeness resid= {abs(pea_sum - (fx - fx0)):.4f}  (giam khi tang T)")
    print(f"|phi_PEA - phi_TubeEG| L1 = {(phi_pea - phi_tube).abs().sum().item():.4f}  "
          f"(khac 0 => increment term co tac dung)")
    print("\n--- geometry log ---")
    print(f"RMS deviation   = {geom.rms_deviation:.6f}")
    print(f"path energy     = {geom.path_energy:.6f}")
    print(f"excess length   = {geom.excess_length:.6f}")

    torch.save(
        {"phi_pea": phi_pea.cpu(), "phi_tube": phi_tube.cpu(),
         "target": target, "geom": vars(geom)},
        args.out,
    )
    print(f"\n[i] da luu -> {args.out}")


if __name__ == "__main__":
    main()
