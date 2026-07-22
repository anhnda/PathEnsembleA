"""
Spectral reference — KHUNG THONG NHAT sinh reference "vang thong tin" cho attribution.

Y tuong (da chung minh): reference_sigma(x) = sum_k c_sigma(lambda_k) <x,phi_k> phi_k
  {phi_k}   : eigenbasis cua PRIOR (Fourier / PCA / graph-Laplacian)
  lambda_k  : "tan so" tong quat = nang luong/phuong sai cua mode k
  c_sigma   : ham CO, giu mode phuong sai-cao, dap mode phuong sai-thap

BLUR LA TRUONG HOP DAC BIET: phi_k = song Fourier, c_sigma = Gaussian e^{-sigma^2|w|^2/2}.
  => mode="fft" tai lap DUNG Gaussian blur (kiem chung trong test cua may).

Ba mode:
  "fft"   : eigenbasis = Fourier (anh). Per-sample, KHONG can tap. c = Gaussian.
            => tai lap Gaussian blur. Prior = quy luat pho 1/f^2 cua anh tu nhien.
  "pca"   : eigenbasis = PCA cua covariance TOAN TAP X (tabular / embedding).
            Giu PC dau (phuong sai cao, "chung"), co PC duoi (phuong sai thap, "rieng").
            Prior = phan bo data. Reference = chieu x len subspace phuong sai cao.
  "graph" : eigenbasis = graph-Laplacian tren k-NN cua X (diffusion maps).
            Scale-space tong quat len data-manifold khong luoi (Coifman-Lafon).
            Prior = cau truc manifold cua data. c = heat kernel e^{-sigma*lambda}.

"sigma" trong ca ba: muc TRUNCATION pho. sigma nho = cat it = gan x.
  sigma -> lon = cat het mode phuong sai-thap => ve mean/centroid (da biet: TE, xoa qua tay).
  => co mot dai sigma "vua dung" giong het tren truc blur.

Khong train. Device theo input. Torch.
"""

from __future__ import annotations
import torch


# ===========================================================================
# (A) FFT mode — tai lap Gaussian blur. Per-sample, khong can tap.
#   x: (C,H,W). Gaussian blur = nhan e^{-sigma^2|w|^2/2} tren Fourier.
#   sigma o day tinh theo PIXEL (nhat quan voi blur_baseline cu: sigma ~ k/3).
# ===========================================================================
@torch.no_grad()
def spectral_reference_fft(x, sigma):
    """
    Gaussian blur qua Fourier = spectral reference voi eigenbasis Fourier.
    x     : (C,H,W) da chuan hoa.
    sigma : do lech chuan Gaussian theo PIXEL.
    Tra ve (C,H,W) reference (= blur(x, sigma)).
    """
    C, H, W = x.shape
    X = torch.fft.rfft2(x, dim=(-2, -1))                      # (C,H,W//2+1) phuc
    # luoi tan so (cycles/pixel), rfft: truc W cat mot nua
    fy = torch.fft.fftfreq(H, device=x.device).view(H, 1)      # (H,1)
    fx = torch.fft.rfftfreq(W, device=x.device).view(1, -1)    # (1,W//2+1)
    w2 = (2 * torch.pi) ** 2 * (fy ** 2 + fx ** 2)             # |omega|^2, rad/pixel
    c = torch.exp(-0.5 * (sigma ** 2) * w2)                    # ham co Gaussian
    Xf = X * c[None]                                           # nhan tren pho
    xb = torch.fft.irfft2(Xf, s=(H, W), dim=(-2, -1))
    return xb


# ===========================================================================
# (B) PCA mode — "blur" cho tabular / embedding. Eigenbasis = covariance toan tap.
#   X_data: (N,D) tap tham chieu (data hoac embedding). x: (D,) hoac (L,D).
#   Giu PC phuong sai cao, co PC phuong sai thap bang c_sigma(lambda).
#   lambda_k = phuong sai PC k. "Tan so cao" = PC duoi (lambda nho).
# ===========================================================================
@torch.no_grad()
def _pca_basis(X_data, energy_floor=1e-8):
    """Tra ve mean (D,), eigenvecs V (D,K) cot = PC, eigenvals (K,) giam dan."""
    mu = X_data.mean(dim=0)                                    # (D,)
    Xc = X_data - mu[None]
    # covariance qua SVD cho on dinh: Xc = U S Vt ; eigval_cov = S^2/(N-1)
    # V cot la PC. Dung full_matrices=False.
    U, S, Vt = torch.linalg.svd(Xc, full_matrices=False)
    evals = (S ** 2) / max(1, X_data.shape[0] - 1)             # (K,) giam dan
    V = Vt.transpose(-1, -2)                                   # (D,K)
    keep = evals > energy_floor * evals.max()
    return mu, V[:, keep], evals[keep]


@torch.no_grad()
def spectral_reference_pca(x, X_data=None, sigma=1.0, basis=None,
                           shrink="gauss"):
    """
    Spectral reference qua PCA cua covariance TOAN TAP.
    x       : (D,) mot mau, hoac (L,D) chuoi embedding (moi hang xu ly doc lap).
    X_data  : (N,D) tap tham chieu de dung covariance. Bo qua neu truyen `basis`.
    sigma   : muc co. Lon => co manh mode phuong sai thap => ve mean.
    basis   : tuple (mu,V,evals) da tinh san (tai dung cho nhieu x, khoi SVD lai).
    shrink  : "gauss" c=exp(-sigma^2 * rho^2) voi rho = rank chuan hoa;
              "heat"  c=exp(-sigma * lambda_norm) (lambda = phuong sai chuan hoa nghich).
    Tra ve reference cung shape voi x.

    Tru mean, chieu len PC, nhan he so co, dung lai, cong mean:
       ref = mu + sum_k c_k <x-mu, v_k> v_k
    c_k gan 1 cho PC dau (giu "chung"), gan 0 cho PC duoi (bo "rieng").
    """
    if basis is None:
        assert X_data is not None, "can X_data hoac basis"
        basis = _pca_basis(X_data)
    mu, V, evals = basis                                       # (D,), (D,K), (K,)
    K = V.shape[1]
    single = (x.dim() == 1)
    Xq = x[None] if single else x                             # (M,D)

    coeff = (Xq - mu[None]) @ V                               # (M,K) toa do PC
    # he so co theo thu hang PC (0=chung nhat .. K-1=rieng nhat)
    rank = torch.arange(K, device=x.device).float() / max(1, K - 1)   # 0..1
    if shrink == "gauss":
        c = torch.exp(-(sigma ** 2) * (rank ** 2))           # giu dau, dap duoi
    else:  # "heat": dung truc tiep phuong sai (nghich dao lam "tan so")
        lam = evals / evals.max()                            # 1..~0
        freq = 1.0 - lam                                     # 0 (PC dau) .. ~1 (PC duoi)
        c = torch.exp(-sigma * freq)
    ref = mu[None] + (coeff * c[None]) @ V.transpose(-1, -2)  # (M,D)
    return ref[0] if single else ref


# ===========================================================================
# (C) GRAPH mode — diffusion maps. Scale-space tren data-manifold khong luoi.
#   Eigenbasis = graph-Laplacian tren k-NN cua X_data. Heat kernel e^{-sigma*lambda}.
#   Tong quat hoa TRUC TIEP heat equation (blur) len tap diem bat ky.
#   Luu y: reference song trong khong gian ham-tren-data; ta dung no de LAM MEM
#   toa do cua x theo eigenvector Laplacian (tuong duong blur tren manifold).
# ===========================================================================
@torch.no_grad()
def _graph_laplacian_basis(X_data, k=10, n_evec=64, eps_scale=1.0):
    """
    Eigen cua normalized graph-Laplacian tren k-NN(X_data).
    Tra ve evecs Phi (N,m), evals (m,) tang dan (lambda_0=0).
    """
    N = X_data.shape[0]
    d2 = torch.cdist(X_data, X_data) ** 2                     # (N,N)
    # bandwidth eps = trung binh khoang cach k-NN (heuristic Coifman)
    knn = torch.topk(d2, k + 1, largest=False).values[:, 1:]  # bo chinh no
    eps = eps_scale * knn.mean()
    Wm = torch.exp(-d2 / (eps + 1e-12))                       # affinity Gaussian
    # sparsify ve k-NN (giu doi xung)
    idx = torch.topk(Wm, k + 1, dim=1).indices
    mask = torch.zeros_like(Wm); mask.scatter_(1, idx, 1.0)
    mask = ((mask + mask.t()) > 0).float()
    Wm = Wm * mask
    deg = Wm.sum(1)                                           # (N,)
    dinv = torch.diag(deg.clamp_min(1e-12).rsqrt())
    Lsym = torch.eye(N, device=X_data.device) - dinv @ Wm @ dinv
    evals, evecs = torch.linalg.eigh(Lsym)                    # tang dan
    m = min(n_evec, N)
    return evecs[:, :m], evals[:m]


@torch.no_grad()
def spectral_reference_graph(x_idx, signal, basis, sigma=1.0):
    """
    Heat-smooth mot TIN HIEU tren data-manifold (diffusion maps).
    x_idx  : (khong dung truc tiep) — giu API dong nhat.
    signal : (N,) hoac (N,F) gia tri tren N diem data (vd feature can lam muot).
    basis  : (Phi (N,m), evals (m,)) tu _graph_laplacian_basis.
    sigma  : thoi gian khuech tan. Lon => muot manh => ve trung binh manifold.
    Tra ve signal da heat-smooth: sum_k e^{-sigma*lambda_k} <signal,phi_k> phi_k.

    Day la blur DUNG NGHIA tren manifold: c = e^{-sigma*lambda} = heat kernel,
    lambda = eigenval Laplacian = "tan so" tren graph. lambda_0=0 (hang so) luon giu.
    """
    Phi, evals = basis                                       # (N,m),(m,)
    c = torch.exp(-sigma * evals)                            # (m,) heat kernel
    S = signal if signal.dim() == 2 else signal[:, None]     # (N,F)
    coeff = Phi.transpose(0, 1) @ S                          # (m,F)
    out = Phi @ (c[:, None] * coeff)                         # (N,F)
    return out if signal.dim() == 2 else out[:, 0]


# ===========================================================================
# Dispatcher tien dung.
# ===========================================================================
@torch.no_grad()
def spectral_reference(x, mode="fft", sigma=1.0, X_data=None, basis=None, **kw):
    """
    Khung thong nhat. mode in {"fft","pca","graph"}.
      fft  : x=(C,H,W)   -> blur(x,sigma). Khong can X_data.
      pca  : x=(D,)/(L,D)-> co PC phuong sai thap. Can X_data hoac basis.
      graph: dung spectral_reference_graph truc tiep (signal-tren-manifold).
    """
    if mode == "fft":
        return spectral_reference_fft(x, sigma)
    if mode == "pca":
        return spectral_reference_pca(x, X_data=X_data, sigma=sigma, basis=basis,
                                      shrink=kw.get("shrink", "gauss"))
    if mode == "graph":
        raise ValueError("graph mode: goi spectral_reference_graph(...) truc tiep")
    raise ValueError(f"mode la: {mode}")

def spectral_reference_fracheat(x, sigma, beta=1.0):
    """
    FRACTIONAL-heat reference: low-pass e^{-1/2 sigma^2 |omega|^{2 beta}}.
      beta = 1   -> Gaussian blur (heat duoi -grad^2), = spectral_reference_fft.
      beta = 1.45 -> fractional Laplacian order 1.45 (do do tu pho anh alpha~-2.9).
    Kernel khong-Gaussian (Levy-type) khi beta != 1. Test: anh muon beta = -alpha/2.
    x: (C,H,W). Tra ve (C,H,W).
    """
    C, H, W = x.shape
    X = torch.fft.rfft2(x, dim=(-2, -1))
    fy = torch.fft.fftfreq(H, device=x.device).view(H, 1)
    fx = torch.fft.rfftfreq(W, device=x.device).view(1, -1)
    w2 = (2 * torch.pi) ** 2 * (fy ** 2 + fx ** 2)            # |omega|^2
    w2b = w2.clamp_min(0) ** beta                             # |omega|^{2 beta}
    c = torch.exp(-0.5 * (sigma ** 2) * w2b)
    Xf = X * c[None]
    xb = torch.fft.irfft2(Xf, s=(H, W), dim=(-2, -1))
    return xb
