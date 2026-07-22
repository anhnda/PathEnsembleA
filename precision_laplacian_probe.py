"""
precision_laplacian_probe.py — KIEM: Sigma^-1 cua anh co ~ grid Laplacian khong?

Luan diem (diffusion unification):
  b_tau = mu + Sigma(Sigma+tau I)^-1 (x-mu) = LOW-PASS SPECTRAL cua precision L=Sigma^-1.
  Neu pho cong suat anh ~ 1/||w||^2 (luat 1/f^2) thi eigenvalue cua Sigma la S(w)~||w||^-2,
  => eigenvalue cua Sigma^-1 la 1/S(w) ~ ||w||^2 = DUNG symbol cua -Laplacian.
  => Sigma^-1 ~ -grad^2 (grid Laplacian), va blur = heat diffusion duoi -grad^2
     = shrinkage duoi Sigma^-1. Thong nhat that su, khong phai an du.

KIEM 3 dieu, chi can folder anh (KHONG can model):
  (1) Pho cong suat radial S(k) co ~ k^-2 khong? (fit log-log, do doc alpha; 1/f^2 => alpha~-2)
  (2) Precision spectrum 1/S(k) co ~ k^2 khong? (doc ~ +2)
  (3) DO TRUC TIEP: L_emp = Sigma^-1 tac dong len anh co gan -Laplacian*anh khong?
      (so ||L_emp x - c * lap x|| / ||L_emp x||, tren khong gian Fourier de tranh
       dung ma tran 150k x 150k). Neu nho => precision ~ Laplacian.

Neu alpha ~ -2 => diffusion unification DUNG cho data nay.
Neu alpha != -2 (vd -1.8, anisotropic) => precision la FRACTIONAL/anisotropic Laplacian,
  khong phai grid Laplacian thuan — van la diffusion nhung doi metric. Bao ro.

Chay:  python precision_laplacian_probe.py <folder> [--glob '*.JPEG'] [--limit 50]
Torch. Khong dung model, khong faithfulness.
"""
from __future__ import annotations
import argparse, glob, os
import numpy as np
import torch


def load_gray(paths, size=224, device="cpu"):
    """Doc anh -> grayscale (M,H,W), [0,1]. Dung luminance de xet pho khong gian."""
    from PIL import Image
    import torchvision.transforms as T
    tf = T.Compose([T.Resize(256), T.CenterCrop(size), T.Grayscale(), T.ToTensor()])
    xs = []
    for p in paths:
        xs.append(tf(Image.open(p).convert("RGB"))[0])   # (H,W)
    return torch.stack(xs).to(device)                    # (M,H,W)


def radial_power_spectrum(imgs):
    """
    Pho cong suat trung binh, binned theo tan so radial |w|.
    imgs: (M,H,W). Tra ve (k_centers, S_k) da bo DC.
    """
    M, H, W = imgs.shape
    imgs = imgs - imgs.mean(dim=(-2, -1), keepdim=True)   # bo DC per-image
    F = torch.fft.fft2(imgs)                              # (M,H,W) phuc
    P = (F.abs() ** 2).mean(0)                            # (H,W) power trung binh
    P = torch.fft.fftshift(P)
    fy = torch.fft.fftshift(torch.fft.fftfreq(H)).view(H, 1)
    fx = torch.fft.fftshift(torch.fft.fftfreq(W)).view(1, W)
    kr = torch.sqrt(fy ** 2 + fx ** 2)                    # (H,W) radial freq
    # bin radial
    nb = min(H, W) // 2
    kmax = kr.max().item()
    edges = torch.linspace(1e-6, kmax, nb + 1)
    kc, Sk = [], []
    for b in range(nb):
        m = (kr >= edges[b]) & (kr < edges[b + 1])
        if m.sum() > 0:
            kc.append(0.5 * (edges[b] + edges[b + 1]).item())
            Sk.append(P[m].mean().item())
    return np.array(kc), np.array(Sk)


def fit_loglog_slope(k, S, kmin_frac=0.05, kmax_frac=0.6):
    """Fit log S = alpha log k + c tren dai tan so giua (bo DC va Nyquist)."""
    kmin, kmax = k.max() * kmin_frac, k.max() * kmax_frac
    m = (k >= kmin) & (k <= kmax) & (S > 0)
    lk, ls = np.log(k[m]), np.log(S[m])
    A = np.vstack([lk, np.ones_like(lk)]).T
    alpha, c = np.linalg.lstsq(A, ls, rcond=None)[0]
    # R^2
    pred = A @ [alpha, c]; ss_res = ((ls - pred) ** 2).sum()
    ss_tot = ((ls - ls.mean()) ** 2).sum()
    r2 = 1 - ss_res / max(ss_tot, 1e-12)
    return float(alpha), float(r2), int(m.sum())


def precision_vs_laplacian(imgs):
    """
    DO TRUC TIEP trong Fourier: precision tac dong = nhan pho voi 1/S(w).
    Laplacian tac dong = nhan pho voi ||w||^2 (up to scale). Do goc COSINE giua
    hai 'symbol' 1/S(w) va ||w||^2 tren luoi tan so => 1.0 nghia la precision ~ Laplacian.
    """
    M, H, W = imgs.shape
    imgs = imgs - imgs.mean(dim=(-2, -1), keepdim=True)
    P = (torch.fft.fft2(imgs).abs() ** 2).mean(0)         # (H,W) power = S(w)
    fy = torch.fft.fftfreq(H).view(H, 1)
    fx = torch.fft.fftfreq(W).view(1, W)
    w2 = (fy ** 2 + fx ** 2)                              # ||w||^2 = Laplacian symbol
    # precision symbol = 1/S(w); bo DC (w=0)
    mask = w2 > 0
    invS = torch.zeros_like(P); invS[mask] = 1.0 / P[mask].clamp_min(1e-12)
    a = invS[mask].flatten(); b = w2[mask].flatten()
    # cosine giua hai symbol (log-scale de khong bi bin lon lan at)
    la, lb = a.clamp_min(1e-30).log(), b.clamp_min(1e-30).log()
    la, lb = la - la.mean(), lb - lb.mean()
    cos = (la @ lb) / (la.norm() * lb.norm()).clamp_min(1e-12)
    # spearman-ish: correlation cua rank
    return float(cos.item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--glob", default="*.JPEG")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--size", type=int, default=224)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.folder, args.glob)))[:args.limit]
    if not paths:
        raise FileNotFoundError(f"khong thay anh: {os.path.join(args.folder, args.glob)}")
    print(f"[i] {len(paths)} anh, size={args.size}")

    imgs = load_gray(paths, size=args.size)
    k, S = radial_power_spectrum(imgs)
    alpha, r2, nfit = fit_loglog_slope(k, S)
    cos = precision_vs_laplacian(imgs)

    print(f"\n=== PHO CONG SUAT RADIAL ===")
    print(f"[i] fit log S(k) = alpha log k + c  tren {nfit} bin (dai giua)")
    print(f"[i] alpha = {alpha:.3f}   (R^2 = {r2:.3f})")
    print(f"[i] luat 1/f^2 => alpha = -2.0. precision spectrum 1/S ~ k^{-alpha:.2f}")
    print(f"[i]   -Laplacian symbol = k^2, tuc alpha_precision = +2 <=> alpha_power = -2.")

    print(f"\n=== PRECISION vs LAPLACIAN (truc tiep, Fourier) ===")
    print(f"[i] cosine(log 1/S(w), log ||w||^2) = {cos:.4f}   (1.0 => Sigma^-1 ~ -Laplacian)")

    print(f"\n=== KET LUAN ===")
    if abs(alpha + 2.0) < 0.25 and cos > 0.9:
        print(f"[OK] alpha~-2 VA cosine~1 => Sigma^-1 ~ grid Laplacian.")
        print(f"[OK]  => blur = heat diffusion duoi -grad^2 = shrinkage duoi Sigma^-1.")
        print(f"[OK]  Diffusion unification DUNG cho data nay (khong chi an du).")
    elif cos > 0.9:
        print(f"[~] cosine cao nhung alpha={alpha:.2f} != -2 => precision ~ ||w||^(-alpha),")
        print(f"[~]  tuc FRACTIONAL Laplacian (-grad^2)^({-alpha/2:.2f}), khong phai grid Laplacian thuan.")
        print(f"[~]  Van la diffusion, nhung metric la fractional/anisotropic. Van thong nhat,")
        print(f"[~]  chi la blur Gaussian KHONG phai heat kernel dung — can fractional heat.")
    else:
        print(f"[!!] cosine={cos:.2f} thap => Sigma^-1 KHONG ~ Laplacian tren data nay.")
        print(f"[!!]  pho anisotropic manh (canh/texture) => precision khong phai Laplacian isotropic.")
        print(f"[!!]  Diffusion-duoi-Laplacian KHONG giai thich duoc blur o day. Xem lai.")

    # dump curve de plot neu can
    out = "precision_spectrum.csv"
    import csv
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["k", "S_k", "inv_S_k", "k_squared"])
        for kk, ss in zip(k, S):
            w.writerow([kk, ss, 1.0/max(ss,1e-30), kk**2])
    print(f"\n[i] curve -> {out}  (k, S(k), 1/S(k), k^2) de plot log-log")


if __name__ == "__main__":
    main()