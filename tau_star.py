"""
tau_star.py — tau* CLOSED FORM. Bo tau khoi danh sach sieu tham so.

=============================================================================
CO SO

Gain cua Shrinkage:  g_k(tau) = tau / (s_k + tau)
Viet theo u = log tau:

    g_k(u) = 1 / (1 + exp(-(u - log s_k)))  =  sigmoid(u - log s_k)

*** Gain la SIGMOID theo log tau, TAM tai log s_k. ***
Day la cau truc quan trong nhat va draft KHONG he khai thac.

=> Δf(u) = f(x) - f(b_tau(x))  la mot TONG SIGMOID:

    Δf(u) ≈ sum_k c_k * sigmoid(u - log s_k)

moi eigen-direction k dong gop MOT BAC THANG, bac thang bat len khi log tau
vuot log s_k. Do cao bac thang = c_k = evidence model dat vao huong k.

Dao ham:
    dΔf/du = sum_k c_k * sigmoid'(u - log s_k)
           = tong cac BELL CURVE, moi cai dinh tai log s_k, do cao c_k.

KNEE (Δf'' = 0)  <=>  DINH cua Δf'  <=>  log tau trung TAM cua khoi eigenvalue
mang nhieu evidence nhat.

    *** tau* ≈ trung binh (hinh hoc) co trong so cua s_k, trong so |c_k| ***

Knee KHONG phai thu thuat do thi. No la ESTIMATOR cua phuong sai trung binh
cac huong ma model THUC SU DOC.

=============================================================================
CONG THUC — MOT BACKWARD PASS, KHONG SWEEP

    d   = x - mu                          (D,)
    d~  = V^T d                           toa do eigen
    g~  = V^T grad_x F(x)                 gradient trong toa do eigen
    c_k = g~_k * d~_k                     dong gop TUYEN TINH cua huong k vao f(x)

    log tau* = sum_k |c_k| * log s_k  /  sum_k |c_k|

    tau* = exp( log tau* )

Per-input TU DONG (c_k phu thuoc x) => va duoc win% 33-38% (mot tau toan cuc
khong the dung cho moi input).

=============================================================================
Y NGHIA CHO PAPER

tau khong con TU DO. No la dai luong DAN XUAT tu (F, x, Sigma):

    "Gia cua viec o trong phan phoi duoc dat tai phuong sai ma model dang doc
     evidence."

Baseline co DUNG nhung huong model dung, giu nguyen nhung huong no khong dung.
Khop voi Eq.(3) cua draft: residual tau/(s_k+tau) * (x-mu)_k — voi tau = s_ev,
ranh gioi xoa/giu dat DUNG tai khoi evidence.

Giai thich luon:
  - vision sigma*~4, NLP tau*~1, tabular tau*>=100 KHAC NHAU vi PHO EVIDENCE
    khac nhau, khong phai tuy tien.
  - vi sao mean (tau->inf) do: no co MOI huong, ke ca s_k >> s_ev ma model khong dung.
  - vi sao tau nho do: chua co toi khoi evidence.

=============================================================================
DIEU KIEN DE CONG THUC CO NGHIA

Trung binh co trong so chi co nghia neu |c_k| TAP TRUNG thanh mot khoi.
Neu c_k TRAI DEU tren moi log s_k thi:
  - trung binh co trong so = mot con so vo nghia (tam cua mot phan bo phang)
  - va knee cung MO (khop du doan PR lon o vision, pho anh 1/f^2)
=> LUON kiem tra spread cua |c_k| theo log s_k TRUOC khi tin tau*.
   Ham evidence_spectrum() lam viec do.

KHONG chay gi.
"""

from __future__ import annotations
import math
import torch


# ---------------------------------------------------------------------------
# 1. Pho evidence: c_k = <grad F, v_k> * <x - mu, v_k>
# ---------------------------------------------------------------------------
def evidence_coeffs(x, grad_x, ref):
    """
    x      : (D,) hoac (M,D)
    grad_x : (D,) hoac (M,D)   grad_x F(x) — MOT backward pass tai x
    ref    : GaussRef (mu, V, s)

    Tra ve c : (D,) hoac (M,D)
        c_k = (V^T grad)_k * (V^T (x-mu))_k
            = dong gop TUYEN TINH cua eigen-direction k vao f(x) - f(mu)
    Vi sum_k c_k = grad^T (x-mu) = xap xi tuyen tinh cua f(x)-f(mu) => c_k la
    phan ra evidence theo huong. Dung |c_k| lam trong so (dau khong quan trong:
    huong nao model DUNG, khong phai dung theo chieu nao).
    """
    single = (x.dim() == 1)
    X = x[None] if single else x
    G = grad_x[None] if single else grad_x
    d = (X - ref.mu[None]) @ ref.V                    # (M,D) toa do eigen
    g = G @ ref.V                                     # (M,D)
    c = g * d
    return c[0] if single else c


# ---------------------------------------------------------------------------
# 2. tau* CLOSED FORM
# ---------------------------------------------------------------------------
def tau_star(x, grad_x, ref, floor_s=1e-12, weight="abs"):
    """
    log tau* = sum_k w_k log s_k / sum_k w_k,   w_k = |c_k| (mac dinh)

    weight:
      "abs"  : w_k = |c_k|                    <- MAC DINH, theo dao ham
      "sq"   : w_k = c_k^2                    (nhan manh huong manh)
      "grad" : w_k = |g~_k|                   (chi gradient, bo qua d — de doi chung)

    Tra ve (tau*, dict chan doan).
    """
    single = (x.dim() == 1)
    c = evidence_coeffs(x, grad_x, ref)               # (M,D) hoac (D,)
    C = c[None] if single else c
    s = ref.s.clamp_min(floor_s)
    ls = s.log()[None]                                # (1,D)

    if weight == "abs":
        w = C.abs()
    elif weight == "sq":
        w = C.pow(2)
    elif weight == "grad":
        G = (x[None] if single else x)
        w = ((grad_x[None] if single else grad_x) @ ref.V).abs()
    else:
        raise ValueError(weight)

    Z = w.sum(1, keepdim=True).clamp_min(1e-30)
    log_tau = (w * ls).sum(1, keepdim=True) / Z       # (M,1)
    tau = log_tau.exp().squeeze(1)                    # (M,)

    # --- chan doan: trong so co TAP TRUNG khong? ---
    p = w / Z                                          # phan bo xac suat tren k
    # do lech chuan cua log s duoi phan bo p  -> spread cua khoi evidence
    m1 = (p * ls).sum(1)
    m2 = (p * ls.pow(2)).sum(1)
    sd_log_s = (m2 - m1.pow(2)).clamp_min(0).sqrt()   # (M,)
    # entropy hieu dung: bao nhieu huong THUC SU dong gop
    ent = -(p.clamp_min(1e-30) * p.clamp_min(1e-30).log()).sum(1)
    k_eff = ent.exp()                                  # (M,) so huong hieu dung

    diag = {"c": c, "w": w, "sd_log_s": sd_log_s if not single else sd_log_s[0],
            "k_eff": k_eff if not single else k_eff[0],
            "log_tau": log_tau.squeeze(1) if not single else log_tau[0, 0]}
    return (tau[0] if single else tau), diag


# ---------------------------------------------------------------------------
# 3. Pho evidence — BAT BUOC xem truoc khi tin tau*
# ---------------------------------------------------------------------------
def evidence_spectrum(x, grad_x, ref, n_bin=12, floor_s=1e-12):
    """
    Histogram cua |c_k| theo log s_k. Tra loi cau hoi SONG CON:

        |c_k| co TAP TRUNG thanh mot khoi khong?

    Neu CO   -> trung binh co trong so co nghia -> tau* dang tin.
    Neu KHONG (trai deu) -> tau* la tam cua mot phan bo phang => VO NGHIA,
       va knee cung se MO. Khop du doan: vision co pho 1/f^2 (PR lon) => trai deu.

    Tra ve (edges, mass, sd_log_s, k_eff, frac_top) — mass da chuan hoa.
    """
    c = evidence_coeffs(x, grad_x, ref)
    C = c[None] if c.dim() == 1 else c
    w = C.abs().mean(0)                                # (D,) trung binh tren input
    s = ref.s.clamp_min(floor_s)
    ls = s.log()

    lo, hi = ls.min().item(), ls.max().item()
    if hi - lo < 1e-9:
        hi = lo + 1.0
    edges = torch.linspace(lo, hi, n_bin + 1, device=w.device)
    idx = torch.bucketize(ls, edges[1:-1])             # (D,) -> bin 0..n_bin-1
    mass = torch.zeros(n_bin, device=w.device)
    mass.scatter_add_(0, idx, w)
    mass = mass / mass.sum().clamp_min(1e-30)

    p = w / w.sum().clamp_min(1e-30)
    m1 = (p * ls).sum()
    sd = ((p * ls.pow(2)).sum() - m1.pow(2)).clamp_min(0).sqrt()
    ent = -(p.clamp_min(1e-30) * p.clamp_min(1e-30).log()).sum()
    k_eff = ent.exp()
    # bao nhieu % evidence nam trong 10% huong manh nhat?
    top = max(1, int(0.1 * w.numel()))
    frac_top = w.sort(descending=True).values[:top].sum() / w.sum().clamp_min(1e-30)

    return {"edges": edges, "mass": mass, "sd_log_s": float(sd),
            "k_eff": float(k_eff), "frac_top10": float(frac_top),
            "D": int(w.numel()), "log_tau_star": float(m1)}


def print_evidence_spectrum(sp, tag=""):
    D = sp["D"]
    print(f"\n=== PHO EVIDENCE {tag} ===")
    print(f"[i] D = {D}")
    print(f"[i] k_eff = {sp['k_eff']:.1f}  ({sp['k_eff']/D*100:.1f}% cua D)"
          f"   <- so huong THUC SU dong gop evidence")
    print(f"[i] top-10% huong manh nhat giu {sp['frac_top10']*100:.1f}% tong evidence")
    print(f"[i] sd(log s) duoi trong so |c| = {sp['sd_log_s']:.3f}"
          f"   <- do TRAI cua khoi evidence (nats)")
    print(f"[i] log tau* = {sp['log_tau_star']:.4f}  =>  tau* = {math.exp(sp['log_tau_star']):.6g}")
    print()
    print(f"{'bin (log s)':>22}{'|c| mass':>12}")
    print("-" * 36)
    e, m = sp["edges"].tolist(), sp["mass"].tolist()
    mx = max(m) if m else 1.0
    for i, v in enumerate(m):
        bar = "#" * int(40 * v / mx) if mx > 0 else ""
        print(f"[{e[i]:>8.2f},{e[i+1]:>8.2f}]{v:>12.4f}  {bar}")
    print("-" * 36)
    if sp["sd_log_s"] > 2.0:
        print("[!!] sd(log s) LON => evidence TRAI DEU tren nhieu bac phuong sai.")
        print("[!!]  Trung binh co trong so = tam cua mot phan bo PHANG => tau* VO NGHIA.")
        print("[!!]  Va knee cung se MO. (Du doan: vision, pho anh 1/f^2.)")
        print("[!!]  KHONG dung tau* o modality nay ma khong noi ro han che.")
    else:
        print("[i] sd(log s) nho => evidence TAP TRUNG => tau* dang tin.")


# ---------------------------------------------------------------------------
# 3b. VISION: sigma* — Fourier/blur KHONG dung cong thuc tren
# ---------------------------------------------------------------------------
# CANH BAO: vision (e1_batch_image.py) dung spectral_reference_fft, tuc GAUSSIAN
# BLUR tren Fourier:
#
#       gain(omega) = exp(-0.5 * sigma^2 * |omega|^2)
#
# KHONG phai gain shrinkage  g_k = tau/(s_k+tau) = sigmoid(log tau - log s_k).
#
# => Cong thuc "trung binh hinh hoc co trong so cua s_k" KHONG ap thang duoc.
#    Ban chat khac: blur GIU tan so thap, XOA tan so cao (low-pass).
#    Shrinkage GIU huong low-variance, XOA huong high-variance.
#    Chung trung nhau CHI KHI Sigma stationary va pho giam theo tan so (Cor. 2
#    cua draft) — va do la mot GIA DINH, khong phai dinh nghia.
#
# Logic tuong duong cho blur: residual  (1 - gain) = 1 - exp(-0.5 sigma^2 w^2).
# Evidence tai tan so omega:  c(w) = <grad F, e_w> * <x, e_w>   (Fourier coeff)
# Muc xoa cua tan so w dat 1/2 khi  0.5*sigma^2*w^2 = ln 2  =>  w_cut = sqrt(2 ln2)/sigma
#
# => sigma* dat sao cho w_cut trung TAM cua khoi evidence theo tan so:
#
#       sigma* = sqrt(2 ln 2) / w_ev,   log w_ev = sum_w |c(w)| log|w| / sum_w |c(w)|
#
# Cung mot y: dat ranh gioi xoa/giu DUNG tai cho model doc evidence. Nhung day la
# TAN SO, khong phai phuong sai. Phai noi ro trong paper la hai dai luong khac nhau.


def sigma_star_fourier(x, grad_x, eps=1e-12, frac=0.5, n_bisect=40):
    """
    x, grad_x : (C,H,W) hoac (M,C,H,W)   — anh chuan hoa va grad_x F(x)
    Tra ve (sigma*, dict). sigma theo PIXEL, khop spectral_reference_fft.

    =====================================================================
    BAN CU SAI 3 CHO — sigma* = 1.74 trong khi best = @4 (lech 2.3x):

    (1) TRONG SO SAI DAU.  Dung c(w) = |X(w)| * |G(w)| (do lon), bo PHA.
        Dong gop THAT cua tan so w vao f(x) la phan THUC cua tich vo huong:
            c(w) = Re[ conj(G(w)) * X(w) ]
        Do moi la <grad F, e_w><x, e_w>, khop cong thuc c_k = g~_k * d~_k o
        phan tabular. |X||G| la CAN TREN cua no, va can tren thi LUON DUONG o
        moi tan so => trai deu GIA TAO (k_eff = 5863/25k = 23% toan pho).

    (2) BIAS DEM-VANH.  So tan so tren vanh ban kinh w TI LE VOI w (dien tich
        vanh ~ 2*pi*w*dw). Nen sum_w bi chi phoi boi tan so CAO thuan tuy do
        DEM, khong phai do evidence. Trung binh hinh hoc bi keo len
        => w_ev qua lon => sigma* qua NHO. Dung huong lech quan sat duoc.

    (3) DINH NGHIA SAI.  "Cat tai TAM pho evidence" khong phai dieu ta muon.
        Cai quyet dinh la baseline XOA duoc bao nhieu evidence, tuc residual
            r(w) = 1 - exp(-0.5 * sigma^2 * w^2)
        Nen dat sigma* sao cho TONG EVIDENCE BI XOA = frac * tong evidence:

            sum_w c(w) * (1 - exp(-0.5 sigma^2 w^2))  =  frac * sum_w c(w)

        Don dieu tang theo sigma => nghiem DUY NHAT, bisection.
        Khong can trung binh hinh hoc, khong dinh bias dem-vanh (vi c(w) xuat
        hien o CA hai ve, ti le bi triet).
    =====================================================================

    frac = 0.5 : xoa mot NUA evidence. La lua chon co the tranh cai — nhung no
    la MOT tham so duy nhat, va co y nghia (diem giua), khac han viec sweep sigma.
    """
    single = (x.dim() == 3)
    X = x[None] if single else x
    G = grad_x[None] if single else grad_x
    M, C, H, W = X.shape

    Xf = torch.fft.rfft2(X, dim=(-2, -1))               # (M,C,H,Wr) phuc
    Gf = torch.fft.rfft2(G, dim=(-2, -1))

    # (1) evidence CO DAU: Re[ conj(G) * X ], gop kenh mau
    c = (Gf.conj() * Xf).real.sum(1)                    # (M,H,Wr)

    fy = torch.fft.fftfreq(H, device=X.device).view(H, 1)
    fx = torch.fft.rfftfreq(W, device=X.device).view(1, -1)
    w = (2 * math.pi) * (fy ** 2 + fx ** 2).sqrt()      # |omega| rad/px, (H,Wr)

    mask = w > eps                                      # bo DC
    cw = c[:, mask]                                     # (M,K) CO DAU
    ww = w[mask][None]                                  # (1,K)

    # dung |c| lam trong so (huong nao model DUNG, khong quan tam chieu)
    a = cw.abs()
    Z = a.sum(1, keepdim=True).clamp_min(1e-30)         # (M,1) tong evidence

    # (3) bisection tren log sigma: tim sigma sao cho evidence bi xoa = frac * Z
    lo = torch.full((M, 1), math.log(1e-3), device=X.device)
    hi = torch.full((M, 1), math.log(1e3), device=X.device)
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        sg = mid.exp()                                  # (M,1)
        removed = (a * (1.0 - torch.exp(-0.5 * sg.pow(2) * ww.pow(2)))).sum(1, keepdim=True)
        too_much = removed > frac * Z                   # xoa qua nhieu -> giam sigma
        hi = torch.where(too_much, mid, hi)
        lo = torch.where(too_much, lo, mid)
    sig = (0.5 * (lo + hi)).exp().squeeze(1)            # (M,)

    # --- chan doan ---
    p = a / Z
    lw = ww.clamp_min(eps).log()
    m1 = (p * lw).sum(1)
    sd = ((p * lw.pow(2)).sum(1) - m1.pow(2)).clamp_min(0).sqrt()
    ent = -(p.clamp_min(1e-30) * p.clamp_min(1e-30).log()).sum(1)
    # bao nhieu % evidence nam trong 10% tan so manh nhat?
    K = a.shape[1]
    top = max(1, int(0.1 * K))
    frac_top = a.sort(1, descending=True).values[:, :top].sum(1) / Z.squeeze(1)
    # ti le evidence DUONG (neu ~0.5 thi c(w) doi dau lung tung -> tin hieu yeu)
    pos = (cw > 0).float().mean(1)

    diag = {"w_ev": m1.exp() if not single else m1.exp()[0],
            "sd_log_w": sd if not single else sd[0],
            "k_eff": ent.exp() if not single else ent.exp()[0],
            "frac_top10": frac_top if not single else frac_top[0],
            "frac_pos": pos if not single else pos[0]}
    return (sig[0] if single else sig), diag


def print_sigma_star(sig, diag, sigma_sweep=None, tag="", frac=0.5):
    M = sig.numel()
    q = torch.tensor([0.25, 0.5, 0.75]).double()
    qq = torch.quantile(sig.double().cpu(), q)
    print(f"\n=== sigma* CLOSED FORM (Fourier) {tag}  n={M} ===")
    print(f"[i] median {qq[1]:.4f}  mean {sig.mean():.4f}  IQR [{qq[0]:.4f}, {qq[2]:.4f}]  (pixel)")
    print(f"[i] dinh nghia: sigma sao cho XOA {frac*100:.0f}% tong evidence")
    print(f"[i]   sum_w |c(w)| (1 - exp(-0.5 σ²ω²)) = {frac:.2f} * sum_w |c(w)|")
    print(f"[i]   c(w) = Re[conj(G(w)) X(w)]  <- CO DAU (ban cu dung |X||G|, SAI)")
    print(f"[i] w_ev {diag['w_ev'].median():.5f} rad/px   k_eff {diag['k_eff'].median():.0f}"
          f"   sd(log w) {diag['sd_log_w'].median():.3f}")
    print(f"[i] top-10% tan so giu {diag['frac_top10'].median()*100:.1f}% evidence")
    print(f"[i] ti le c(w) > 0: {diag['frac_pos'].median()*100:.1f}%"
          f"   (~50% => c doi dau lung tung => tin hieu YEU)")
    if sigma_sweep:
        print(f"[i] sigma_sweep = {sigma_sweep}")
    if diag["frac_top10"].median() < 0.5:
        print("[!!] top-10% tan so giu < 50% evidence => evidence TRAI DEU.")
        print("[!!]  Moi rule dua tren 'tam khoi evidence' se kem tin cay o vision.")


# ---------------------------------------------------------------------------
# 4. Doi chieu tau* voi sweep (knee / max Δf|b-x| / oracle)
# ---------------------------------------------------------------------------
def knee_on_ell(dist, delta_f):
    """
    Kneedle tren (ell, Δf) — TRUC HOANH LA QUANG DUONG, khong phai index/tau/log tau.

    Ly do: index KHONG phai dai luong vat ly (doi grid thi doi ket qua). ell thi co.
    Va tren truc ell, cac baseline NGOAI TRUC (black/zero/IG2 — co ell va Δf nhung
    khong co tau) cung xep duoc len cung do thi.

    dist, delta_f: (M,T). Tra ve index (M,) cua knee.
    LUU Y: Kneedle gia dinh duong cong TANG-LOM. Δf co the KHONG don dieu (NLP:
    0.325 -> 0.448 -> 0.528 -> 0.512). Ta cummax truoc de ep don dieu — nhung do
    la XAP XI, va phai noi ro.
    """
    M, T = delta_f.shape
    out = []
    for i in range(M):
        xx = dist[i] - dist[i].min()
        xx = xx / xx.max().clamp_min(1e-12)
        yy = torch.cummax(delta_f[i].clamp_min(0), dim=0).values
        yy = yy - yy.min()
        yy = yy / yy.max().clamp_min(1e-12)
        out.append(int((yy - xx).argmax().item()))
    return torch.tensor(out, device=delta_f.device)


def compare_rules(taus, dist, delta_f, tau_hat, id_gap=None):
    """
    So sanh tau* CLOSED FORM voi cac rule tu SWEEP.

    taus     : (T,)
    dist     : (M,T)   ||b_tau - x||_2
    delta_f  : (M,T)   f(x) - f(b_tau)
    tau_hat  : (M,)    tau* closed form (tu tau_star())
    id_gap   : (M,T)   I-D / Soft-gap that tren test (ORACLE — CHI doi chung)

    In bang. KHONG tu ket luan.
    """
    T = len(taus)
    M = delta_f.shape[0]
    lt = taus.log()

    # tau* -> index gan nhat tren grid (de so sanh cong bang)
    i_star = (tau_hat[:, None].clamp_min(1e-30).log() - lt[None]).abs().argmin(1)

    i_knee = knee_on_ell(dist, delta_f)
    ratio = delta_f / dist.clamp_min(1e-12)
    i_ratio = ratio.argmax(1)

    rows = [("tau* (closed form)", i_star),
            ("knee (ell, Δf)", i_knee),
            ("max Δf/|b-x|", i_ratio)]
    if id_gap is not None:
        rows.append(("ORACLE (test metric)", id_gap.argmax(1)))

    print(f"\n=== DOI CHIEU RULE (n={M}, grid {T} diem) ===")
    print(f"{'rule':<22}{'median tau':>12}{'mean tau':>12}{'IQR':>24}")
    print("-" * 72)
    q = torch.tensor([0.25, 0.5, 0.75]).double()
    for name, idx in rows:
        t = taus[idx].double().cpu()
        qq = torch.quantile(t, q)
        print(f"{name:<22}{qq[1]:>12.4g}{t.mean():>12.4g}   [{qq[0]:.4g}, {qq[2]:.4g}]")
    print("-" * 72)
    if id_gap is not None:
        orc = id_gap.argmax(1)
        for name, idx in rows[:-1]:
            ag = (idx == orc).float().mean().item() * 100
            near = ((idx - orc).abs() <= 1).float().mean().item() * 100
            print(f"[i] {name:<22} == ORACLE: {ag:>5.1f}%   |diff|<=1 buoc: {near:>5.1f}%")
    print("[i] tau* la CLOSED FORM: 1 backward pass, KHONG sweep, KHONG grid.")
    print("[i]   knee/max-ratio can SWEEP. Neu tau* ~ chung => bo sweep khoi paper.")
    print("[i] tau* raw (khong snap ve grid): median "
          f"{torch.quantile(tau_hat.double().cpu(), torch.tensor(0.5).double()):.6g}")