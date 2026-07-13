"""
tau_diag.py — chon tau/sigma tu dong. CHI dung forward pass (khong cham ins/del).

=============================================================================
CAC RULE DA THU VA DA HONG — ghi lai de khong ai lap lai:

  (1) rho* = f(b)/f(x) ~ 0.3        -> magic number. Bo.

  (2) SNR = Δf / ||b-x||_2^2        -> HONG HAI LAN:
      (a) tau->0: Δf ~ c*tau (bac 1), ||b-x||^2 ~ tau^2 (bac 2) => SNR ~ 1/tau
          -> +vo cuc. KHONG co cuc dai noi. argmax luon tra ve day grid.
          (smoke test xac nhan: tau_snr median = min(grid))
      (b) D lon (vision 3x224x224): ||b-x||_2 ~ 100-1000 nhung Δf bi chan trong
          [0,1] => SNR ~ 1e-5, in ra toan 0.0000. Va vi MOI baseline manh deu
          cham tran Δf ~ 0.838, tu so thanh HANG SO => SNR thoai hoa thanh
          1/||b-x||^2, tuc chi xep hang theo "ai gan x nhat". Vo nghia.

  (3) |b-x|/|x-mu| va amp = (f(b)-1/K)/(f(x)-1/K)
      -> BO. Moi modality mot measure => bang cross-modality vo nghia, va luan diem
      "portable" cua paper sup theo.
      Cu the: o VISION, mu = anh xam trung binh ImageNet = TENSOR 0 trong khong gian
      chuan hoa => ||x-mu|| = ||x||_2, chi la do lon cua chinh buc anh, KHONG mang
      thong tin gi ve prior. Cot do vo nghia o day.
      amp thi can K (so lop) — cung la mot tham so modality-specific.

  (4) tau tai (hoac ngay truoc) DIEM GAY cua Δf
      -> dung o vision (@4/@8) va NLP (@1), nhung SAI o TABULAR.
      Ly do: rule gia dinh "sau gay la quang duong thua". O vision/NLP dung, vi
      ||b-x|| VAN TIEP TUC TANG sau gay. O tabular ||b-x|| BAO HOA CUNG LUC voi
      Δf (ca hai cung hoi tu khi b -> mu) => KHONG co quang duong thua => khong
      co ly do dung lai o diem gay. Best that su la @100 (~ mean), khong phai @1.

=============================================================================
RULE DUNG: TI GIA BIEN

     r(tau) = d(Δf) / d||b_tau(x) - x||_2

"Di them mot don vi quang duong thi mua duoc bao nhieu tin hieu."
Dung khi ti gia tut. KHONG can biet K, khong can biet san o dau, khong can biet
diem gay o dau. Bat bien voi so chieu (khac han SNR).

Doi chieu voi LOG THAT (sai phan tren grid decade hien co):

  VISION   @4->@8   : (0.838-0.773)/(164-128)   =  0.0018
           @8->@16  : (0.8385-0.8384)/(208-164) =  0.000002   <- sap 1000x
           => dung o @8.       best that = @4/@8               OK

  NLP      @0.1->@1 : (0.528-0.448)/(4.74-4.24) =  0.160
           @1->@10  : (0.512-0.528)/(4.93-4.74) = -0.084       <- AM
           => dung o @1.       best that = @1                  OK

  TABULAR  @1->@10  : (0.629-0.160)/(0.640-0.560) = 5.86
           @10->@100: (0.731-0.629)/(0.651-0.640) = 9.27       <- VAN TANG
           => di tiep.         best that = @100                OK

Ket qua kiem tren so THAT (grid decade san co), eps = 0.05:

  modality      best that   tau_rate
  ---------------------------------------------------------------
  VISION n=3    @8          @8        OK
  VISION n=5    @4          @8        trong vung HOA (xem duoi)
  NLP           @1          @1        OK
  TABULAR       @100        @100      OK  + tu bao "GRID CUT DAU"

VISION n=5: best la @4 (I-D 4.67) nhung @8 = 4.48, win% 40/40 HOA, SE +-0.58.
Chenh 4%, khong tach duoc thong ke. Va o n=3 thi @8 LAI thang. Best that su o
vision DAO DONG giua @4 va @8 theo n; rule on dinh chon @8 — nam trong vung hoa.
KHONG giau: rule chua tach duoc @4 vs @8 o vision. Grid 4 diem qua tho, gay nhay
giua cac lan chay -> DUNG dense sweep (--tau_diag --diag_n 30) de xac dinh.

eps: 0.05 va 0.1 cho CUNG ket qua. eps=0.01 lam vision n=5 troi ra @16 (I-D 3.25,
te nhat) vi ti gia chi tut 85 lan (< 100). Chon eps=0.05: on dinh.

TI GIA BIEN la MOT METRIC DUY NHAT cho ca ba modality:
  - KHONG can mu       (khac |b-x|/|x-mu|)
  - KHONG can K        (khac amp)
  - KHONG can biet D   (khac SNR)
  - KHONG can biet san o dau, gay o dau
Chi can HAI cot ma ca ba script deu da co: Δf va ||b-x||_2.

BANG CHUAN (giong het nhau o ca ba modality):

    method        f(x)     f(b)      Δf    |b-x|₂   d(Δf)/d|b-x|

Voi hang CO DINH (black/mean/blur/IG2/EG) khong co ti gia bien (khong nam tren
truc tau), nhung van doc duoc tren CUNG HAI COT dau: cung Δf, ai di xa hon.
Khong can measure rieng — chi can doc bang.

Ghi chu ve tabular: ti gia VAN TANG o cuoi grid => GRID CUT DAU. Rule tu phat
hien duoc dieu do (bao [!!] khi tau_rate roi vao diem cuoi grid).

=============================================================================
KHONG chay gi. Chi cung cap ham.
"""

from __future__ import annotations
import csv
import math
import torch


# ---------------------------------------------------------------------------
# Pho cua Sigma
# ---------------------------------------------------------------------------
def participation_ratio(s: torch.Tensor) -> float:
    """
    PR = (sum s_k)^2 / sum s_k^2.
    PR nho (<< D) => pho TAP TRUNG => duong cong doc, gay SAC, rule on dinh.
    PR lon (~ D)  => pho TRAI DAI (anh tu nhien 1/f^2) => gay MO, rule bat dinh.
    Kiem tra TRUOC khi chay, de biet co nen ky vong rule sac net khong.
    """
    s = s.clamp_min(0.0).double()
    return float(((s.sum() ** 2) / (s ** 2).sum().clamp_min(1e-30)).item())


def effective_tau_scale(s: torch.Tensor, mode: str = "mean") -> float:
    """
    tau chi co nghia SO VOI eigenvalue (g_k = s_k/(s_k+tau)).
    => reparam tau = gamma * s_bar de SO SANH DUOC cross-modality.
    Hien "Shrinkage-IG@100" o tabular va "@100" o NLP la hai con so KHONG lien
    quan gi den nhau — reviewer se hoi ngay.
    """
    s = s.clamp_min(0.0)
    return float({"mean": s.mean(), "median": s.median(),
                  "trace_over_d": s.sum() / s.numel()}[mode].item())


def log_tau_grid(lo: float, hi: float, n: int = 30):
    """Grid log. 4-5 diem (grid decade hien tai) KHONG DU de tinh ti gia bien."""
    return [float(v) for v in torch.logspace(math.log10(lo), math.log10(hi), n)]


def gamma_grid(ref_s: torch.Tensor, lo=1e-2, hi=1e2, n=30, mode="mean"):
    """Grid SCALE-FREE: tau = gamma * s_bar. gamma=1 <=> gain trung binh ~ 1/2."""
    sb = effective_tau_scale(ref_s, mode)
    return [float(g * sb) for g in torch.logspace(math.log10(lo), math.log10(hi), n)], sb


# ---------------------------------------------------------------------------
# TI GIA BIEN — dai luong trung tam
# ---------------------------------------------------------------------------
def _marginal_rate(delta_f: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
    """
    r[i,t] = d(Δf) / d(dist), sai phan TIEN tren grid: gan ti gia cua khoang
    [t, t+1] cho diem t. Diem cuoi lay ti gia cua khoang cuoi.

    KHONG chia cho dist^2 -> do la SNR, da hong o D lon.
    KHONG lay dao ham theo tau -> ti gia phai theo QUANG DUONG, vi chinh quang
    duong (khong phai tau) moi la thu sinh ra meo mo O(L * ||b-x||^2) cua IG.

    Bat bien voi so chieu: tu so va mau so deu la "mot buoc doc theo duong cong".
    """
    M, T = delta_f.shape
    if T < 2:
        return torch.zeros_like(delta_f)
    dd = delta_f[:, 1:] - delta_f[:, :-1]
    ds = (dist[:, 1:] - dist[:, :-1]).clamp_min(1e-12)   # quang duong tang don dieu theo tau
    fwd = dd / ds                                        # (M, T-1): ti gia cua khoang [t, t+1]
    r = torch.full_like(delta_f, float("nan"))
    r[:, :-1] = fwd
    # DIEM CUOI: KHONG co khoang [T-1, T] => KHONG co ti gia. Truoc day copy ti gia cua
    # khoang truoc vao day -> ti gia GIA. Voi vision n=5 no khien r[@16] = r[@8] = 4.3e-05
    # vua du vuot nguong eps*rmax = 4.27e-05 => rule chon @16 (te nhat, I-D 3.25) thay vi @4.
    # De NaN: khong co du lieu thi khong duoc doan.
    return r


# ---------------------------------------------------------------------------
# Quet dense — batch duoc (tabular, vision)
# ---------------------------------------------------------------------------
@torch.no_grad()
def sweep_curve(X_eval, score_fn, baseline_fn, taus,
                mu=None, ref_s=None, ref_V=None, ref_mu=None):
    """
    X_eval    : (M, ...)
    score_fn  : (B,...) -> (B,)  xac suat lop target
    baseline_fn: (x_single, tau) -> baseline
    taus      : list[float], NEN dense (>= 25 diem)
    mu        : chi dung de canh bao f(mu) > f(x). KHONG dung de chuan hoa quang duong.
    ref_*     : chi dung cho Mahalanobis (kiem tra P2). Khong bat buoc.

    Tra ve dict: (M,T) rho, delta_f, dist, rate, maha_b
                 (M,)  f_x, f_mu, maha_x, valid
    """
    M, T = X_eval.shape[0], len(taus)
    dev = X_eval.device
    f_x = score_fn(X_eval).clamp_min(1e-12)                      # (M,) PER-INPUT

    f_mu = (score_fn(mu[None].expand(M, *mu.shape)) if mu is not None
            else torch.full((M,), float("nan"), device=dev))

    have_ref = (ref_s is not None and ref_V is not None and ref_mu is not None)

    def _maha(Z):
        c = (Z.reshape(Z.shape[0], -1) - ref_mu.reshape(1, -1)) @ ref_V
        return ((c ** 2) / ref_s.clamp_min(1e-12)[None]).sum(1).sqrt()

    maha_x = _maha(X_eval) if have_ref else torch.full((M,), float("nan"), device=dev)

    rho = torch.zeros(M, T, device=dev)
    dist = torch.zeros(M, T, device=dev)
    maha_b = torch.full((M, T), float("nan"), device=dev)

    for t_i, tau in enumerate(taus):
        B = torch.stack([baseline_fn(X_eval[i], tau) for i in range(M)], 0)
        rho[:, t_i] = score_fn(B) / f_x
        dist[:, t_i] = (X_eval - B).reshape(M, -1).norm(dim=1)
        if have_ref:
            maha_b[:, t_i] = _maha(B)

    delta_f = f_x[:, None] * (1.0 - rho)                 # = f(x) - f(b): ngan sach Completeness
    rate = _marginal_rate(delta_f, dist)                 # <-- DAI LUONG DUY NHAT CAN

    valid = _validate(f_x, f_mu, M, mu is not None)
    return {"taus": torch.tensor(taus, device=dev),
            "rho": rho, "delta_f": delta_f, "dist": dist, "rate": rate,
            "maha_b": maha_b, "f_x": f_x, "f_mu": f_mu,
            "maha_x": maha_x, "valid": valid}


def _validate(f_x, f_mu, M, have_mu):
    """
    Loai input thoai hoa.
      (a) f(x) < 0.05 -> rho = f(b)/f(x) no.
      (b) f(mu) > f(x) -> co ve mu lam model TU TIN HON => Δf < 0 tai MOI tau
          => KHONG co tau toi uu. Xay ra voi input model kem tu tin hon trung binh.
          Tabular f(x)=0.99 nen an toan; NLP co cau f(x) thap SE dinh.
    """
    valid = f_x >= 0.05
    nb = int((~valid).sum())
    if nb:
        print(f"[!] {nb}/{M} input co f(x) < 0.05 -> rho no, LOAI khoi rule.")
    if have_mu:
        bad = (f_mu > f_x) & valid
        if int(bad.sum()):
            print(f"[!!] {int(bad.sum())}/{M} input co f(mu) > f(x): co ve mu lam model TU TIN HON")
            print(f"[!!]  => Δf < 0 tai MOI tau => KHONG co tau toi uu. Loai, bao cao rieng.")
            valid = valid & (~bad)
    return valid


# ---------------------------------------------------------------------------
# Quet dense — KHONG batch duoc (NLP: moi cau mot seq_len)
# ---------------------------------------------------------------------------
@torch.no_grad()
def sweep_curve_varlen(examples, score_one, embed_of, baseline_one, taus,
                       mu_baseline=None, maha_one=None, mask_one=None):
    """
    Giong sweep_curve nhung loop tung example.
    dist = ||.||_2 tren cac coordinate DUOC GIU (mask_one), KHONG phai .abs().mean()
    nhu code cu (do la L1/D, khong phai quang duong Euclid).
    """
    M, T = len(examples), len(taus)
    rho = torch.zeros(M, T); dist = torch.zeros(M, T)
    maha_b = torch.full((M, T), float("nan"))
    f_x = torch.zeros(M); f_mu = torch.full((M,), float("nan"))
    maha_x = torch.full((M,), float("nan"))

    def _sub(t, m):
        return (t[0][m] if (m is not None and t.dim() == 3) else t).reshape(-1)

    for i, it in enumerate(examples):
        x = embed_of(it)
        f_x[i] = max(float(score_one(it, x)), 1e-12)
        m = mask_one(it) if mask_one is not None else None
        if maha_one is not None:
            maha_x[i] = float(maha_one(it, x))
        if mu_baseline is not None:
            f_mu[i] = float(score_one(it, mu_baseline(it)))
        for t_i, tau in enumerate(taus):
            b = baseline_one(it, x, tau)
            rho[i, t_i] = float(score_one(it, b)) / f_x[i]
            dist[i, t_i] = _sub(x - b, m).norm().item()
            if maha_one is not None:
                maha_b[i, t_i] = float(maha_one(it, b))

    delta_f = f_x[:, None] * (1.0 - rho)
    rate = _marginal_rate(delta_f, dist)

    valid = _validate(f_x, f_mu, M, mu_baseline is not None)
    return {"taus": torch.tensor([float(t) for t in taus]),
            "rho": rho, "delta_f": delta_f, "dist": dist, "rate": rate,
            "maha_b": maha_b, "f_x": f_x, "f_mu": f_mu,
            "maha_x": maha_x, "valid": valid}


# ---------------------------------------------------------------------------
# Selection rules
# ---------------------------------------------------------------------------
def selection_rules(curve: dict, eps: float = 0.05):
    """
    Tra ve (rules, valid).

    tau_rate  : *** RULE CHINH ***
                tau LON NHAT ma ti gia bien r(tau) >= eps * max_t r(t).
                "Di tiep chung nao con mua duoc du tin hieu tren moi don vi duong."
                eps la nguong TUONG DOI: "ti gia da tut 1/eps lan thi dung".

                eps = 0.05 (tut 20 lan). KHONG phai 0.01:
                  vision n=5: ti gia @4->@8 = 0.00367, @8->@16 = 4.3e-05 => tut 85 lan.
                  Voi eps=0.01 (doi tut 100 lan) thi 85 < 100 => KHONG dung => chon @16,
                  la baseline TE NHAT (I-D 3.25) thay vi @4 (4.67). eps=0.01 qua LONG.
                  Voi eps=0.05: nguong = 0.05*0.00427 = 2.1e-04 > 4.3e-05 => dung o @8. Dung.
                  vision : ti gia sap 1000x sau @8   -> dung @8
                  NLP    : ti gia AM sau @1          -> dung @1
                  tabular: ti gia VAN TANG tai @100  -> di tiep (grid cut dau)
                Neu rule roi vao DIEM CUOI grid => grid cut dau, phai noi rong.

    tau_knee  : Kneedle PER-INPUT tren [i, f_i].
                truc x = INDEX cua sweep; truc y = f(b_tau(x)) cua CHINH input do.
                Moi input mot knee (vi b_tau(x) phu thuoc x). In phan bo, khong
                in mot con so chung.
                LUU Y: index KHONG phai dai luong vat ly => phu thuoc GRID.
                Them/bot diem sweep o vung bao hoa se KEO GIAN index va DICH knee.
                Bang chung: tabular, grid 5 diem cho knee @1; them 200/300/400/1000
                (deu la MEAN baseline, f(b) da dung yen) thi knee nhay sang @100.

    (Da bo tau_amp: no can K => modality-specific => bang cross-modality vo nghia.)
    """
    taus, dev = curve["taus"], curve["taus"].device
    M, T = curve["rho"].shape
    valid = curve.get("valid", torch.ones(M, dtype=torch.bool, device=dev))
    rate, df = curve["rate"], curve["delta_f"]
    out = {}

    # --- tau_rate (CHINH) ---
    # rate co T diem nhung chi T-1 KHOANG co nghia: rate[:, j] = ti gia cua khoang
    # [j, j+1], j = 0..T-2. rate[:, T-1] la NaN (khong co khoang sau no).
    #
    # Rule: tim KHOANG TOT CUOI CUNG j*, roi chon tau tai DIEM CUOI cua khoang do,
    # tuc tau[j*+1]. Neu KHONG khoang nao tot (ti gia am ngay tu dau) -> tau[0].
    #
    # LICH SU HAI LOI DA MAC O DAY:
    #  (a) Gan ti gia cua khoang cuoi cho ca DIEM cuoi (rate[T-1] = rate[T-2]).
    #      -> ti gia GIA. Vision n=5: rate[@16] = rate[@8] = 4.3e-05, vua du vuot
    #      nguong => chon @16 (te nhat, I-D 3.25) thay vi @4 (4.67).
    #  (b) Chon tau[j*] thay vi tau[j*+1] (off-by-one).
    #      -> NLP: chon @0.1 trong khi best la @1. Ly do: rate[@1] = -0.089 mo ta
    #      doan @1->@10 (xau), nhung DEN duoc @1 thi van tot. Loai @1 la sai — no
    #      bi loai vi doan SAU no xau, chu khong phai vi den no la xau.
    r_int = rate[:, :T - 1]                                       # (M, T-1) chi cac khoang that
    rmax = r_int.clamp_min(0).max(dim=1, keepdim=True).values.clamp_min(1e-12)
    ok = r_int >= eps * rmax                                      # khoang [j,j+1] con dang di
    ar = torch.arange(T - 1, device=dev, dtype=torch.float)[None]
    j_star = (ok.float() * ar).argmax(dim=1)                      # khoang tot cuoi cung
    has = ok.any(1)
    idx_star = torch.where(has, j_star + 1, torch.zeros_like(j_star))   # DIEM CUOI cua khoang do
    out["tau_rate"] = taus[idx_star]
    at_edge = float((idx_star[valid] == T - 1).float().mean().item()) if int(valid.sum()) else 0.0

    # --- tau_knee: Kneedle PER-INPUT tren [i, f_i] ---
    #
    # HAI LOI DA MAC:
    #
    # (1) TRUNG BINH ROI TIM KNEE, thay vi TIM KNEE ROI XEM PHAN BO.
    #     b_tau(x) PHU THUOC x => moi input co duong cong f(b_tau(x)) RIENG, knee RIENG,
    #     tau toi uu RIENG. Bang chung: ORACLE per-input IQR [0.076, 0.961] — trai hon
    #     mot bac. Va win% o vision chi 33-38% => mot sigma toan cuc chi dung cho 1/3 anh.
    #     Trung binh cac duong cong lam MO knee: input A gay o tau=0.1, input B gay o
    #     tau=2 => duong trung binh thoai, khong co gay ro.
    #
    # (2) SAI TRUC. Da code Kneedle tren (log tau, Δf) + cummax. Phai la [i, f_i]:
    #       truc x = INDEX i cua sweep (0..T-1), chuan hoa [0,1]
    #       truc y = f(b_tau_i(x)) cua CHINH input do, chuan hoa [0,1] theo min/max
    #                cua CHINH no (khong phai min/max toan cuc)
    #     Duong cong GIAM => knee = max (1 - x̂) - ŷ.
    #     Do la ly do tau_knee = day grid o MOI input (0.00047, IQR bang 0): sai ham,
    #     khong phai sai du lieu.
    f_b = curve["rho"] * curve["f_x"][:, None]                    # (M,T) f(b) PER-INPUT
    xh = torch.arange(T, device=dev, dtype=torch.float) / max(T - 1, 1)   # (T,) index chuan hoa
    lo = f_b.min(dim=1, keepdim=True).values
    hi = f_b.max(dim=1, keepdim=True).values
    yh = (f_b - lo) / (hi - lo).clamp_min(1e-12)                  # (M,T) giam 1 -> 0
    dev_knee = (1.0 - xh[None]) - yh                              # (M,T)
    out["tau_knee"] = taus[dev_knee.argmax(dim=1)]

    out["_at_edge"] = at_edge
    return out, valid


def oracle_tau(curve: dict, id_gap_per_tau: torch.Tensor):
    """ORACLE — CHI de doi chung, KHONG duoc dung de chon: argmax I-D that tren test."""
    return curve["taus"][id_gap_per_tau.argmax(dim=1)]


# ---------------------------------------------------------------------------
# In
# ---------------------------------------------------------------------------
def print_curve_table(curve: dict, tag: str = ""):
    taus = curve["taus"].tolist()
    M = curve["rho"].shape[0]
    sq = math.sqrt(M)
    fx = curve["f_x"]

    print(f"\n=== TAU-DIAGNOSTIC {tag}  (n={M}, dense sweep {len(taus)} diem) ===")
    print(f"[i] f(x) = {fx.mean():.4f}  sd {fx.std():.4f}  [min {fx.min():.4f}, max {fx.max():.4f}]")
    print(f"[i]   ^ PER-INPUT. Log cu in f(x) nhu MOT so duy nhat = trung binh da bop phang.")
    print(f"[i]   Ngan sach Completeness = f(x)*(1-rho) TI LE voi f(x); meo mo O(L*||b-x||²) thi KHONG.")
    print(f"[i]   => tau toi uu PHU THUOC f(x) => mot tau toan cuc khong the dung cho moi input.")
    if not torch.isnan(curve["f_mu"]).all():
        print(f"[i] f(mu) = {curve['f_mu'].mean():.4f}   (san ma f(b) tien toi khi tau -> inf)")
    print()
    print(f"{'tau':>10}{'f(b)':>9}{'Δf':>9}{'|b-x|₂':>10}{'d(Δf)/d|b-x|':>16}")
    print("-" * 55)
    for i, t in enumerate(taus):
        fb = (curve["rho"][:, i] * fx).mean().item()
        df_ = curve["delta_f"][:, i].mean().item()
        d_ = curve["dist"][:, i].mean().item()
        r_ = curve["rate"][:, i].mean().item()
        print(f"{t:>10.4g}{fb:>9.4f}{df_:>9.4f}{d_:>10.4f}{r_:>16.5g}")
    print("-" * 55)
    print("[i] Δf = f(x)-f(b) = NGAN SACH COMPLETENESS (= sum_i phi_i).")
    print("[i] d(Δf)/d|b-x| = TI GIA BIEN: di them 1 don vi quang duong -> mua duoc bao nhieu tin hieu.")
    print("[i]   MOT metric duy nhat cho ca 3 modality: khong can mu, khong can K, khong can D.")
    print("[i]   Dung khi ti gia tut => tau_rate. Xem bang rules ben duoi.")


def print_per_input_agreement(rules: dict, oracle, taus, valid=None):
    """
    BANG QUAN TRONG NHAT: rule co bat duoc BIEN THIEN GIUA CAC INPUT khong?

    Mot rule co the co median trung ORACLE median ma van vo dung, neu no tra ve
    CUNG MOT tau cho moi input trong khi ORACLE trai rong.

    In:
      - IQR cua rule vs IQR cua ORACLE  (rule co trai KHONG?)
      - % khop chinh xac / trong 1 buoc grid
      - correlation Spearman giua rule va ORACLE tren log tau
    """
    rules = {k: v for k, v in rules.items() if not k.startswith("_")}
    if valid is not None:
        rules = {k: v[valid] for k, v in rules.items()}
        oracle = oracle[valid]
    lt = taus.log()
    io = (oracle[:, None].clamp_min(1e-30).log() - lt[None]).abs().argmin(1)

    def _rank(v):
        return v.argsort().argsort().float()

    print(f"\n=== RULE CO BAT DUOC BIEN THIEN GIUA CAC INPUT KHONG? ===")
    print(f"{'rule':<14}{'median':>10}{'IQR':>22}{'== orc':>9}{'<=1 buoc':>10}{'spearman':>10}")
    print("-" * 76)
    q = torch.tensor([0.25, 0.5, 0.75]).double()
    for name, t in rules.items():
        it = (t[:, None].clamp_min(1e-30).log() - lt[None]).abs().argmin(1)
        qq = torch.quantile(t.double().cpu(), q)
        ex = (it == io).float().mean().item() * 100
        nr = ((it - io).abs() <= 1).float().mean().item() * 100
        a, b = _rank(t.log()), _rank(oracle.log())
        sp = float(torch.corrcoef(torch.stack([a, b]))[0, 1]) if t.numel() > 2 else float("nan")
        print(f"{name:<14}{qq[1]:>10.4g}   [{qq[0]:.4g}, {qq[2]:.4g}]{'':>3}"
              f"{ex:>8.1f}%{nr:>9.1f}%{sp:>10.3f}")
    qo = torch.quantile(oracle.double().cpu(), q)
    print(f"{'ORACLE':<14}{qo[1]:>10.4g}   [{qo[0]:.4g}, {qo[2]:.4g}]")
    print("-" * 76)
    print("[i] IQR cua rule HEP ma ORACLE RONG => rule tra ve gan nhu CUNG MOT tau cho moi")
    print("[i]   input => KHONG bat duoc bien thien => vo dung du median co trung.")
    print("[i] spearman = tuong quan hang giua rule va ORACLE tren log tau. ~0 => khong lien quan.")


def print_rules_table(rules: dict, oracle=None, valid=None):
    rules = dict(rules)
    at_edge = rules.pop("_at_edge", None)
    if valid is not None:
        rules = {k: v[valid] for k, v in rules.items()}
        if oracle is not None:
            oracle = oracle[valid]
        print(f"\n[i] rules tren {int(valid.sum())} input hop le (loai {int((~valid).sum())}).")
    print(f"{'rule':<12}{'median':>11}{'mean':>11}{'IQR':>24}{'  == oracle':>13}")
    print("-" * 74)
    q3 = torch.tensor([0.25, 0.5, 0.75]).double()
    for name, t in rules.items():
        qq = torch.quantile(t.double().cpu(), q3)
        ag = f"{(t == oracle).float().mean().item()*100:>11.1f}%" if oracle is not None else ""
        print(f"{name:<12}{qq[1]:>11.4g}{t.mean():>11.4g}   [{qq[0]:.4g}, {qq[2]:.4g}]{'':>4}{ag}")
    if oracle is not None:
        qq = torch.quantile(oracle.double().cpu(), q3)
        print(f"{'ORACLE(I-D)':<12}{qq[1]:>11.4g}{oracle.mean():>11.4g}   [{qq[0]:.4g}, {qq[2]:.4g}]")
    print("-" * 74)
    print("[i] tau_rate = ti gia bien.  tau_knee = Kneedle PER-INPUT tren [i, f(b)].")
    print("[i] CA HAI deu la rule per-input: b_tau(x) phu thuoc x => moi input mot tau rieng.")
    print("[i]   Cot IQR cho thay do trai. Neu IQR hep ma ORACLE trai rong => rule khong bat")
    print("[i]   duoc bien thien giua cac input.")
    print("[!] tau_knee dung INDEX lam truc x => PHU THUOC GRID. Them diem sweep o vung")
    print("[!]   bao hoa (b da = mean, f(b) dung yen) se KEO GIAN index va DICH knee.")
    if oracle is not None:
        print("[i] Neu tau_rate ~ ORACLE => chon duoc tau MA KHONG cham test metric. Do la ket qua chinh.")
    if at_edge is not None and at_edge > 0.3:
        print(f"[!!] {at_edge*100:.0f}% input co tau_rate = CUOI GRID  =>  GRID CUT DAU.")
        print(f"[!!]  Ti gia bien VAN CON CAO o tau lon nhat: baseline chua di het duong.")
        print(f"[!!]  Noi rong grid roi chay lai — HOAC ti gia bien khong phai rule dung o day.")


def dump_curve_csv(curve: dict, path: str, extra_cols: dict | None = None):
    """
    Long format: mot dong = (input i, tau t).
    Day la file de FIT rule offline — dung suy dien tiep tu 4-5 diem grid decade nua.
    """
    taus = curve["taus"].tolist()
    M, T = curve["rho"].shape
    cols = ["i", "tau", "f_x", "f_b", "delta_f", "dist", "rate",
            "maha_x", "maha_b", "valid"]
    if extra_cols:
        cols += list(extra_cols.keys())
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for i in range(M):
            fx = curve["f_x"][i].item()
            for t in range(T):
                row = {"i": i, "tau": taus[t], "f_x": fx,
                       "f_b": fx * curve["rho"][i, t].item(),
                       "delta_f": curve["delta_f"][i, t].item(),
                       "dist": curve["dist"][i, t].item(),
                       "rate": curve["rate"][i, t].item(),
                       "maha_x": curve["maha_x"][i].item(),
                       "maha_b": curve["maha_b"][i, t].item(),
                       "valid": int(curve["valid"][i].item())}
                if extra_cols:
                    for k, v in extra_cols.items():
                        row[k] = v[i, t].item() if v.dim() == 2 else v[i].item()
                w.writerow(row)
    print(f"[i] curve -> {path}  ({M}x{T} dong)")


# ---------------------------------------------------------------------------
# Baseline CO DINH (zero/mean/blur/MaxEnt/IG2...) — xep len CUNG TRUC
# ---------------------------------------------------------------------------
@torch.no_grad()
def fixed_baseline_diag(X_eval, score_fn, b_fn, ref_s=None, ref_V=None, ref_mu=None):
    """
    Cung don vi voi sweep_curve, de tra loi: "IG-zero nam o dau tren truc tau?"

    Cot P2_ok: zero co THAT SU vi pham (P2) khong?
    Log NLP cho thay IG-zero THANG moi Shrinkage. Neu P2_ok cao => zero KHONG
    off-distribution trong embedding space => hang "zero ✗" trong Table 1 SAI o
    modality do, va viec zero thang KHONG con la nghich ly.
    """
    M = X_eval.shape[0]
    f_x = score_fn(X_eval).clamp_min(1e-12)
    B = torch.stack([b_fn(X_eval[i]) for i in range(M)], 0)
    f_b = score_fn(B)
    dist = (X_eval - B).reshape(M, -1).norm(dim=1)
    out = {"f_x": f_x, "f_b": f_b, "delta_f": f_x - f_b, "dist": dist}
    if ref_s is not None:
        def mh(Z):
            c = (Z.reshape(Z.shape[0], -1) - ref_mu.reshape(1, -1)) @ ref_V
            return ((c ** 2) / ref_s.clamp_min(1e-12)[None]).sum(1).sqrt()
        out["maha_b"] = mh(B); out["maha_x"] = mh(X_eval)
        out["P2_ok"] = (out["maha_b"] <= out["maha_x"]).float()
    return out


def print_fixed_header():
    """
    Hang CO DINH (black/mean/blur/IG2/EG) khong nam tren truc tau => KHONG co ti gia
    bien. Nhung doc duoc tren CUNG HAI COT Δf va |b-x|₂: cung Δf, ai di xa hon.
    Khong can measure rieng cho chung.
    """
    print(f"\n{'fixed baseline':<20}{'f(b)':>9}{'Δf':>9}{'|b-x|₂':>10}"
          f"{'|b-mu|_S⁻¹':>13}{'P2':>7}")
    print("-" * 66)


def print_fixed_row(name: str, d: dict):
    def g(k):
        if k not in d:
            return float("nan")
        v = d[k]
        return float("nan") if torch.isnan(v).all() else v.mean().item()

    def f(v, w, p=4):
        return f"{v:>{w}.{p}f}" if v == v else f"{'-':>{w}}"

    p2 = f"{d['P2_ok'].mean().item()*100:>6.0f}%" if "P2_ok" in d else f"{'-':>7}"
    print(f"{name:<20}{f(g('f_b'),9)}{f(g('delta_f'),9)}{f(g('dist'),10)}"
          f"{f(g('maha_b'),13)}{p2}")
