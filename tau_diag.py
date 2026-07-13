"""
tau_diag.py — DIAGNOSTIC cho viec chon tau/sigma tu dong (khong dung test metric).

Van de: tau hien dang duoc sweep thu cong (4-5 diem), va bang ket qua chi in
  f(x)  f(xt)  ratio  |b-x|
voi f(x) la MOT SO DUY NHAT (da bi trung binh phang tren ca dataset). Mat sach
variation cua f(x) giua cac input — trong khi CHINH f(x) la ngan sach Completeness:

    sum_i phi_i = f_c(x) - f_c(b_tau(x)) = f_c(x) * (1 - rho(tau))

Meo mo cua IG doc doan thang dai ell = |x-b| co bac O(L * ell^2), KHONG phu thuoc f(x).
=> ti so tin hieu / meo mo, PER-INPUT:

    SNR_i(tau) = [ f_c(x_i) - f_c(b_tau(x_i)) ] / ||x_i - b_tau(x_i)||_2^2
                 \_______ ngan sach Completeness _______/   \___ quang duong^2 ___/

Module nay:
  1. Quet DENSE tau tren log-grid (mac dinh 25 diem), PER-INPUT (khong trung binh som).
  2. Log: rho, delta_f, ||b-x||_2, ||b-x||_2^2, SNR proxy, Mahalanobis ||b-mu||_S^-1.
  3. Tinh cac ung vien selection rule (deu CHI DUNG forward pass, KHONG dung ins/del):
       tau_snr   = argmax_tau SNR_i(tau)                     <- ung vien chinh
       tau_knee  = knee cua log rho vs log tau (Kneedle: max curvature)
       tau_floor = tau nho nhat sao cho rho <= rho_floor + delta   (rho_floor = f(mu)/f(x))
  4. Tinh participation ratio PR cua Sigma => knee co SAC khong (PR nho = sac).
  5. Xuat CSV per-input x per-tau (long format) de fit rule offline.

Dung chung cho ca 3 modality: chi can truyen
    score_fn(x_batch) -> (M,) xac suat lop target      [torch, no_grad]
    baseline_fn(x, tau) -> baseline cung shape voi x   [torch, no_grad]

KHONG chay gi. Chi cung cap ham. Nguoi goi tu chay.
"""

from __future__ import annotations
import csv
import math
import torch


# ---------------------------------------------------------------------------
# 1. Participation ratio cua Sigma — knee co sac khong?
# ---------------------------------------------------------------------------
def participation_ratio(s: torch.Tensor) -> float:
    """
    PR = (sum s_k)^2 / sum s_k^2.  s = eigenvalue cua Sigma (giam dan).

    PR nho  (<< D) => pho TAP TRUNG => rho(tau) co knee SAC => tau_snr xac dinh ro.
    PR lon  (~ D)  => pho TRAI DAI  (vd anh tu nhien 1/f^2) => knee MO => moi rule
                      chon tau se bat dinh, va Shrinkage@4/@8/blur se gan bang nhau.
    Day la DIEU KIEN KIEM TRA TRUOC khi chay: neu PR lon, dung ky vong rule sac net.
    """
    s = s.clamp_min(0.0).double()
    num = s.sum() ** 2
    den = (s ** 2).sum().clamp_min(1e-30)
    return float((num / den).item())


def effective_tau_scale(s: torch.Tensor, mode: str = "mean") -> float:
    """
    tau chi co nghia SO VOI eigenvalue cua Sigma (g_k = s_k/(s_k+tau)).
    => reparam tau = gamma * s_bar de tau SO SANH DUOC giua cac modality.
    Hien tabular tau=100 va NLP tau=100 la hai con so khong lien quan gi nhau.
    """
    s = s.clamp_min(0.0)
    if mode == "mean":
        return float(s.mean().item())
    if mode == "median":
        return float(s.median().item())
    if mode == "trace_over_d":
        return float((s.sum() / s.numel()).item())
    raise ValueError(mode)


# ---------------------------------------------------------------------------
# 2. Quet dense per-input
# ---------------------------------------------------------------------------
@torch.no_grad()
def sweep_curve(
    X_eval: torch.Tensor,          # (M, ...) cac input can giai thich
    score_fn,                      # (B,...) -> (B,) xac suat lop target
    baseline_fn,                   # (x_single, tau) -> baseline cung shape
    taus,                          # list[float] — NEN dense, log-spaced, >= 20 diem
    mu: torch.Tensor | None = None,        # de tinh rho_floor = f(mu)/f(x)
    ref_s: torch.Tensor | None = None,     # eigenvalue Sigma, de tinh Mahalanobis
    ref_V: torch.Tensor | None = None,     # eigenvector Sigma
    ref_mu: torch.Tensor | None = None,
):
    """
    Tra ve dict cac tensor (M, T):
        rho      : f(b_tau(x)) / f(x)
        delta_f  : f(x) - f(b_tau(x))        <- NGAN SACH COMPLETENESS (co f(x)!)
        dist     : ||x - b_tau(x)||_2
        dist2    : ||x - b_tau(x)||_2^2
        snr      : delta_f / dist2           <- proxy chinh
        maha_b   : ||b - mu||_{Sigma^-1}     <- kiem tra (P2) contraction
    va (M,):
        f_x      : f(x)   PER-INPUT (khong trung binh!)
        rho_floor: f(mu)/f(x)  — san ma rho khong the xuong duoi (voi mean baseline)
        maha_x   : ||x - mu||_{Sigma^-1}
    """
    M = X_eval.shape[0]
    T = len(taus)
    dev = X_eval.device

    f_x = score_fn(X_eval).clamp_min(1e-12)                     # (M,)

    if mu is not None:
        f_mu = score_fn(mu[None].expand(M, *mu.shape))          # (M,)
        rho_floor = (f_mu / f_x)
    else:
        f_mu = torch.full((M,), float("nan"), device=dev)
        rho_floor = torch.full((M,), float("nan"), device=dev)

    rho = torch.zeros(M, T, device=dev)
    dist = torch.zeros(M, T, device=dev)
    maha_b = torch.full((M, T), float("nan"), device=dev)

    have_ref = (ref_s is not None and ref_V is not None and ref_mu is not None)

    def _maha(z):
        """||z - mu||_{Sigma^-1}, z: (M,D) flat."""
        d = (z - ref_mu[None]).reshape(z.shape[0], -1)
        c = d @ ref_V                                            # toa do eigen
        return ((c ** 2) / ref_s.clamp_min(1e-12)[None]).sum(1).sqrt()

    if have_ref:
        maha_x = _maha(X_eval.reshape(M, -1))
    else:
        maha_x = torch.full((M,), float("nan"), device=dev)

    for t_i, tau in enumerate(taus):
        B = torch.stack([baseline_fn(X_eval[i], tau) for i in range(M)], 0)   # (M,...)
        f_b = score_fn(B)                                        # (M,)
        rho[:, t_i] = f_b / f_x
        diff = (X_eval - B).reshape(M, -1)
        dist[:, t_i] = diff.norm(dim=1)
        if have_ref:
            maha_b[:, t_i] = _maha(B.reshape(M, -1))

    dist2 = dist.pow(2).clamp_min(1e-20)
    delta_f = f_x[:, None] - rho * f_x[:, None]                  # = f(x)*(1-rho)
    snr = delta_f / dist2

    # -----------------------------------------------------------------
    # HAI CA THOAI HOA — smoke test loi ra, se XAY RA tren du lieu that:
    #
    # (a) f(x) ~ 0  =>  rho = f(b)/f(x) NO (thay 17338 trong smoke test).
    #     Loc: chi giu input co f(x) >= f_min. Input ma model khong tu tin
    #     vao chinh lop no du doan thi ngan sach Completeness ~ 0, khong
    #     dinh nghia duoc tau toi uu cho no.
    #
    # (b) Δf < 0  =>  baseline lam model TU TIN HON. SNR am, argmax vo nghia.
    #     KHONG phai gia dinh: log tabular cua mày co IG-zero ratio = 1.014 > 1
    #     => Δf < 0 that. Voi baseline nay, "quang duong doi lay tin hieu" khong
    #     con la trade-off — no tra ve tin hieu AM. Ta danh dau, khong argmax lien.
    #
    # valid[i] = input i dung duoc; valid_t[i,t] = (i,t) co Δf > 0.
    # -----------------------------------------------------------------
    valid = f_x >= 0.05                                          # (M,) f(x) du lon
    valid_t = delta_f > 0                                        # (M,T) baseline THUC SU xoa
    n_bad_fx = int((~valid).sum().item())
    n_bad_df = int((~valid_t).any(1).sum().item())
    if n_bad_fx:
        print(f"[!] {n_bad_fx}/{M} input co f(x) < 0.05 -> rho no, LOAI khoi rule.")
    if n_bad_df:
        print(f"[!] {n_bad_df}/{M} input co it nhat 1 tau voi Δf<=0 (baseline lam model TU TIN HON).")

    # ---- CA HONG NGHIEM TRONG: f(mu) > f(x) ----
    # Khi do rho_floor > 1: co lai ve mu lam model TU TIN HON, khong phai it hon.
    # Toan bo khung "quang duong doi lay tin hieu" SUP: Δf < 0 tai MOI tau, khong
    # co tau nao toi uu. Xay ra voi bat ky input nao model kem tu tin hon trung binh.
    # Tabular cua mày f(x)=0.9902 nen an toan; NLP co cau f(x) thap SE dinh.
    # KHONG duoc lang lang bo qua: no doi nghia cua ca Shrinkage o nhung input do.
    if mu is not None:
        bad_floor = (rho_floor > 1.0) & valid
        nb = int(bad_floor.sum().item())
        if nb:
            print(f"[!!] {nb}/{M} input co f(mu) > f(x)  =>  rho_floor > 1.")
            print(f"[!!]  Co lai ve mu lam model TU TIN HON. Δf<0 tai MOI tau => KHONG co tau toi uu.")
            print(f"[!!]  Nhung input nay KHONG the chon tau bang SNR. Bao cao rieng, dung tron vao mean.")
            valid = valid & (~bad_floor)

    return {
        "taus": torch.tensor(taus, device=dev),
        "rho": rho, "delta_f": delta_f, "dist": dist, "dist2": dist2,
        "snr": snr, "maha_b": maha_b,
        "f_x": f_x, "f_mu": f_mu, "rho_floor": rho_floor, "maha_x": maha_x,
        "valid": valid, "valid_t": valid_t,
    }


# ---------------------------------------------------------------------------
# 3. Cac ung vien selection rule — TAT CA chi dung forward pass
# ---------------------------------------------------------------------------
def _knee_kneedle_rising(log_tau: torch.Tensor, y: torch.Tensor) -> int:
    """
    Kneedle cho duong cong TANG-roi-BAO-HOA (concave increasing), vd Δf(tau).
    Chuan hoa ca hai truc ve [0,1], knee = diem XA NHAT tren duong cheo y=x.

    (Ban cu dung log rho GIAM don dieu — nhung rho co the > 1 khi baseline lam
    model tu tin hon, khi do log rho KHONG don dieu va Kneedle tra ve toan
    min-grid. Smoke test bat duoc loi nay.)
    """
    x = log_tau - log_tau.min()
    x = x / x.max().clamp_min(1e-12)
    yy = y - y.min()
    yy = yy / yy.max().clamp_min(1e-12)      # tang tu 0 -> 1
    return int((yy - x).argmax().item())


def selection_rules(curve: dict, delta: float = 0.05):
    """
    Tra ve (rules, valid) — rules: dict[name] -> (M,) tau chon per-input;
    valid: (M,) bool, input nao dung duoc (f(x) khong qua nho).
    Chi tinh tren cac (i,t) hop le (Δf>0). CHI dung forward pass.

      tau_snr   : argmax_tau  Δf / ||x-b||_2^2, TREN cac tau co Δf>0.

                  *** RULE NAY HONG. Smoke test XAC NHAN. ***
                  Khi tau -> 0:  Δf ~ c*tau  (bac 1),  h = ||x-b||^2 ~ tau^2  (bac 2)
                  =>  SNR ~ 1/tau  ->  +vo cuc.
                  SNR KHONG co cuc dai noi; no don dieu GIAM. argmax luon tra ve
                  tau = min(grid). Smoke test: tau_snr median = 0.001 = day grid.

                  MAU SO PHAT QUA MANH o tau nho. Giu rule nay lai CHI de doi chung
                  va de khong ai lap lai sai lam nay. KHONG dung de chon tau.

      tau_snr_a : argmax_tau  Δf / ||x-b||_2^(2*alpha),  alpha < 1.
                  Lam yeu hinh phat quang duong. alpha=0.5 => mau so ~ tau, cung bac
                  voi tu so => ti so huu han khi tau->0, cuc dai noi TON TAI.
                  alpha la THAM SO CAN FIT tu du lieu (CSV long-format), khong duoc
                  doan. Day la ly do phai co dense sweep.

      tau_knee  : Kneedle tren (log tau, Δf) — dung Δf CHU KHONG dung log rho.
                  LY DO: rho co the > 1 (baseline lam model tu tin hon) => log rho
                  KHONG don dieu => Kneedle hong (smoke test: tra ve toan min-grid).
                  Δf thi don dieu tang roi bao hoa => knee co nghia.

      tau_floor : tau NHO NHAT sao cho Δf >= (1-delta) * max_t Δf.
                  = "da xoa gan het thong tin model quan tam, quang duong ngan nhat".
                  (Bo rho_floor: no dua tren f(mu)/f(x), cung no khi f(x) nho.)
                  CANH BAO: tren vision rule nay chon @8/@16, deu THUA @4
                  => CHUA phai rule dung. Giu de doi chung.
    """
    taus = curve["taus"]
    M, T = curve["rho"].shape
    dev = taus.device
    valid = curve.get("valid", torch.ones(M, dtype=torch.bool, device=dev))
    valid_t = curve.get("valid_t", torch.ones(M, T, dtype=torch.bool, device=dev))
    NEG = torch.tensor(-1e30, device=dev)
    out = {}

    # --- tau_snr: mask cac tau co Δf<=0 truoc khi argmax ---
    # (HONG — xem docstring. Giu de doi chung.)
    snr_m = torch.where(valid_t, curve["snr"], NEG)
    out["tau_snr"] = taus[snr_m.argmax(dim=1)]

    # --- tau_snr_alpha: hinh phat quang duong lam yeu, mu alpha ---
    for a in (0.5, 0.75):
        snr_a = curve["delta_f"] / curve["dist2"].pow(a).clamp_min(1e-20)
        snr_a = torch.where(valid_t, snr_a, NEG)
        out[f"tau_snr@a={a}"] = taus[snr_a.argmax(dim=1)]

    # --- tau_knee: Kneedle tren (log tau, Δf) ---
    log_tau = taus.log()
    df = curve["delta_f"]
    idx = []
    for i in range(M):
        y = df[i].clone()
        y[~valid_t[i]] = 0.0
        # can y TANG don dieu de Kneedle co nghia -> cummax
        y = torch.cummax(y, dim=0).values
        idx.append(_knee_kneedle_rising(log_tau, y))
    out["tau_knee"] = taus[torch.tensor(idx, device=dev)]

    # --- tau_floor: tau nho nhat dat (1-delta)*max Δf ---
    df_m = torch.where(valid_t, df, torch.zeros_like(df))
    dmax = df_m.max(dim=1, keepdim=True).values.clamp_min(1e-12)
    hit = df_m >= (1.0 - delta) * dmax
    first = torch.where(hit.any(1), hit.float().argmax(1),
                        torch.full((M,), T - 1, device=dev, dtype=torch.long))
    out["tau_floor"] = taus[first]

    return out, valid


def oracle_tau(curve: dict, id_gap_per_tau: torch.Tensor):
    """
    ORACLE (chi de DOI CHUNG, KHONG duoc dung de chon):
    tau tot nhat theo chinh I-D / Soft-gap tren test. Neu tau_snr ~ oracle
    => rule hop le (chon duoc tau ma khong cham test metric).

    id_gap_per_tau: (M, T) — gap cua tung input tai tung tau.
    """
    return curve["taus"][id_gap_per_tau.argmax(dim=1)]


# ---------------------------------------------------------------------------
# 4. In bang + xuat CSV
# ---------------------------------------------------------------------------
def print_curve_table(curve: dict, tag: str = ""):
    """Bang tong hop mean±SE theo tau — THAY cho viec in f(x) nhu MOT so duy nhat."""
    taus = curve["taus"].tolist()
    M = curve["rho"].shape[0]
    sq = math.sqrt(M)

    def ms(t, i):
        v = t[:, i]
        return v.mean().item(), v.std().item() / sq

    print(f"\n=== TAU-DIAGNOSTIC {tag}  (n={M} inputs, dense sweep) ===")
    print(f"[i] f(x)      = {curve['f_x'].mean():.4f} ± {curve['f_x'].std()/sq:.4f}"
          f"   [min {curve['f_x'].min():.4f}, max {curve['f_x'].max():.4f}]  <-- PER-INPUT, khong phang")
    print(f"[i] f(mu)     = {curve['f_mu'].mean():.4f}")
    print(f"[i] rho_floor = {curve['rho_floor'].mean():.4f} ± {curve['rho_floor'].std()/sq:.4f}"
          f"   (= f(mu)/f(x); rho khong xuong duoi day)")
    if not torch.isnan(curve["maha_x"]).all():
        print(f"[i] |x-mu|_S-1 = {curve['maha_x'].mean():.4f}   (Mahalanobis cua chinh input)")
    print()
    print(f"{'tau':>10}{'rho':>10}{'Δf':>10}{'|b-x|':>10}{'|b-x|²':>11}"
          f"{'SNR=Δf/|b-x|²':>16}{'|b-mu|_S-1':>13}")
    print("-" * 80)
    for i, tau in enumerate(taus):
        r, _ = ms(curve["rho"], i)
        df, _ = ms(curve["delta_f"], i)
        d, _ = ms(curve["dist"], i)
        d2, _ = ms(curve["dist2"], i)
        sn, sn_se = ms(curve["snr"], i)
        mb = curve["maha_b"][:, i]
        mb_s = f"{mb.mean().item():>13.4f}" if not torch.isnan(mb).all() else f"{'-':>13}"
        print(f"{tau:>10.4g}{r:>10.4f}{df:>10.4f}{d:>10.4f}{d2:>11.4f}"
              f"{sn:>11.4f}±{sn_se:<4.3f}{mb_s}")
    print("-" * 80)
    i_star = int(curve["snr"].mean(0).argmax().item())
    print(f"[i] argmax SNR (aggregate, CO f(x) ben trong) = tau {taus[i_star]:.4g}")
    print("[i] LUU Y: aggregate argmax != mean cua per-input argmax. Xem bang rule ben duoi.")


def print_rules_table(rules: dict, oracle: torch.Tensor | None = None,
                      valid: torch.Tensor | None = None):
    if valid is not None:
        rules = {k: v[valid] for k, v in rules.items()}
        if oracle is not None:
            oracle = oracle[valid]
        print(f"[i] rules tinh tren {int(valid.sum())} input hop le "
              f"(loai {int((~valid).sum())} input f(x)<0.05)")
    print(f"\n{'rule':<14}{'median tau':>12}{'mean tau':>12}{'IQR':>22}"
          f"{'  agree w/ oracle':>18}")
    print("-" * 80)
    for name, t in rules.items():
        q = torch.quantile(t.double(), torch.tensor([0.25, 0.5, 0.75], device=t.device).double())
        ag = ""
        if oracle is not None:
            # ti le input ma rule chon DUNG tau oracle (hoac trong 1 buoc grid)
            same = (t == oracle).float().mean().item()
            ag = f"{same*100:>16.1f}%"
        print(f"{name:<14}{q[1].item():>12.4g}{t.mean().item():>12.4g}"
              f"   [{q[0].item():.4g}, {q[2].item():.4g}]{'':>6}{ag}")
    if oracle is not None:
        q = torch.quantile(oracle.double(), torch.tensor([0.25, 0.5, 0.75], device=oracle.device).double())
        print(f"{'ORACLE(I-D)':<14}{q[1].item():>12.4g}{oracle.mean().item():>12.4g}"
              f"   [{q[0].item():.4g}, {q[2].item():.4g}]")
    print("-" * 80)


def dump_curve_csv(curve: dict, path: str, extra_cols: dict | None = None):
    """
    Long format: mot dong = (input i, tau t). Day la file de FIT rule offline —
    dung suy dien tiep tu 4 diem sweep nua.
    """
    taus = curve["taus"].tolist()
    M, T = curve["rho"].shape
    cols = ["i", "tau", "f_x", "f_b", "rho", "rho_floor", "delta_f",
            "dist", "dist2", "snr", "maha_x", "maha_b"]
    if extra_cols:
        cols += list(extra_cols.keys())
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for i in range(M):
            fx = curve["f_x"][i].item()
            for t in range(T):
                row = {
                    "i": i, "tau": taus[t], "f_x": fx,
                    "f_b": fx * curve["rho"][i, t].item(),
                    "rho": curve["rho"][i, t].item(),
                    "rho_floor": curve["rho_floor"][i].item(),
                    "delta_f": curve["delta_f"][i, t].item(),
                    "dist": curve["dist"][i, t].item(),
                    "dist2": curve["dist2"][i, t].item(),
                    "snr": curve["snr"][i, t].item(),
                    "maha_x": curve["maha_x"][i].item(),
                    "maha_b": curve["maha_b"][i, t].item(),
                }
                if extra_cols:
                    for k, v in extra_cols.items():
                        row[k] = v[i, t].item() if v.dim() == 2 else v[i].item()
                w.writerow(row)
    print(f"[i] da luu curve -> {path}  ({M}x{T} dong)")


# ---------------------------------------------------------------------------
# 5. Diagnostic cho MOT baseline co dinh (zero/mean/blur/MaxEnt/IG2/...)
#    -> de xep no vao CHINH duong cong tau, thay vi bang rieng.
# ---------------------------------------------------------------------------
@torch.no_grad()
def fixed_baseline_diag(X_eval, score_fn, b_fn, ref_s=None, ref_V=None, ref_mu=None):
    """
    b_fn(x) -> baseline (khong co tau). Tra ve dict (M,) — CUNG DON VI voi sweep_curve,
    de tra loi: "IG-zero nam o dau tren truc tau?" va "no co vi pham (P2) khong?"

    Cu the cho NLP: neu maha_b(zero) < maha_x thi zero KHONG vi pham (P2), va hang
    "zero ✗" trong Table 1 la SAI tren modality do.
    """
    M = X_eval.shape[0]
    f_x = score_fn(X_eval).clamp_min(1e-12)
    B = torch.stack([b_fn(X_eval[i]) for i in range(M)], 0)
    f_b = score_fn(B)
    diff = (X_eval - B).reshape(M, -1)
    dist = diff.norm(dim=1)
    dist2 = dist.pow(2).clamp_min(1e-20)
    rho = f_b / f_x
    delta_f = f_x - f_b
    out = {"f_x": f_x, "f_b": f_b, "rho": rho, "delta_f": delta_f,
           "dist": dist, "dist2": dist2, "snr": delta_f / dist2}
    if ref_s is not None:
        d = (B.reshape(M, -1) - ref_mu[None])
        c = d @ ref_V
        out["maha_b"] = ((c ** 2) / ref_s.clamp_min(1e-12)[None]).sum(1).sqrt()
        dx = (X_eval.reshape(M, -1) - ref_mu[None]) @ ref_V
        out["maha_x"] = ((dx ** 2) / ref_s.clamp_min(1e-12)[None]).sum(1).sqrt()
        out["P2_ok"] = (out["maha_b"] <= out["maha_x"]).float()
    return out


def print_fixed_row(name: str, d: dict):
    M = d["f_x"].shape[0]
    sq = math.sqrt(M)
    p2 = ""
    if "P2_ok" in d:
        frac = d["P2_ok"].mean().item()
        p2 = f"{frac*100:>8.0f}%"
    print(f"{name:<20}{d['rho'].mean():>10.4f}{d['delta_f'].mean():>10.4f}"
          f"{d['dist'].mean():>10.4f}{d['dist2'].mean():>11.4f}"
          f"{d['snr'].mean():>11.4f}±{d['snr'].std().item()/sq:<4.3f}"
          f"{d.get('maha_b', torch.tensor([float('nan')])).mean():>13.4f}{p2}")


def print_fixed_header():
    print(f"\n{'fixed baseline':<20}{'rho':>10}{'Δf':>10}{'|b-x|':>10}{'|b-x|²':>11}"
          f"{'SNR':>16}{'|b-mu|_S-1':>13}{'P2 ok':>9}")
    print("-" * 90)


# ---------------------------------------------------------------------------
# 6. Bien the cho input KHONG batch duoc (NLP: moi cau mot seq_len khac nhau)
# ---------------------------------------------------------------------------
@torch.no_grad()
def sweep_curve_varlen(
    examples,                # list cac item; moi item la gi tuy nguoi goi
    score_one,               # (item, x_like) -> scalar float  f(.) cho lop target
    embed_of,                # (item) -> x  (tensor, shape tuy y — vd (1,seq,d))
    baseline_one,            # (item, x, tau) -> baseline cung shape x
    taus,
    mu_baseline=None,        # (item) -> baseline mean (cho rho_floor). None => bo qua
    maha_one=None,           # (item, z) -> ||z-mu||_{S^-1} (float). None => bo qua
    mask_one=None,           # (item) -> bool mask chon coordinate tinh |b-x| (vd bo special token)
):
    """
    Giong sweep_curve nhung loop tung example. Dung cho NLP (seq_len bien thien).
    Tra ve cung cau truc dict cac tensor (M,T)/(M,).

    LUU Y: |b-x| duoc tinh la ||.||_2 tren cac coordinate DUOC GIU (mask_one),
    KHONG phai .abs().mean() nhu code NLP hien tai (do la L1/D, khong phai quang
    duong Euclid — dung cho SNR la sai).
    """
    M, T = len(examples), len(taus)
    rho = torch.zeros(M, T)
    dist = torch.zeros(M, T)
    maha_b = torch.full((M, T), float("nan"))
    f_x = torch.zeros(M)
    f_mu = torch.full((M,), float("nan"))
    maha_x = torch.full((M,), float("nan"))

    for i, it in enumerate(examples):
        x = embed_of(it)
        fx = float(score_one(it, x))
        f_x[i] = max(fx, 1e-12)
        m = mask_one(it) if mask_one is not None else None
        if maha_one is not None:
            maha_x[i] = float(maha_one(it, x))
        if mu_baseline is not None:
            f_mu[i] = float(score_one(it, mu_baseline(it)))
        for t_i, tau in enumerate(taus):
            b = baseline_one(it, x, tau)
            rho[i, t_i] = float(score_one(it, b)) / f_x[i]
            diff = (x - b)
            if m is not None:
                diff = diff[0][m] if diff.dim() == 3 else diff[m]
            dist[i, t_i] = diff.reshape(-1).norm().item()
            if maha_one is not None:
                maha_b[i, t_i] = float(maha_one(it, b))

    dist2 = dist.pow(2).clamp_min(1e-20)
    delta_f = f_x[:, None] - rho * f_x[:, None]
    return {
        "taus": torch.tensor([float(t) for t in taus]),
        "rho": rho, "delta_f": delta_f, "dist": dist, "dist2": dist2,
        "snr": delta_f / dist2, "maha_b": maha_b,
        "f_x": f_x, "f_mu": f_mu,
        "rho_floor": (f_mu / f_x), "maha_x": maha_x,
    }


def log_tau_grid(lo: float, hi: float, n: int = 25):
    """Log-spaced grid. Mac dinh 25 diem — 4-5 diem nhu hien tai la KHONG DU de tim knee."""
    return [float(v) for v in torch.logspace(math.log10(lo), math.log10(hi), n)]


def gamma_grid(ref_s: torch.Tensor, lo=1e-2, hi=1e2, n=25, mode="mean"):
    """
    Grid SCALE-FREE: tau = gamma * s_bar. Dung cai nay de tau SO SANH DUOC giua
    tabular / vision / NLP. gamma=1 <=> gain trung binh ~ 1/2.
    """
    sb = effective_tau_scale(ref_s, mode)
    return [float(g * sb) for g in torch.logspace(math.log10(lo), math.log10(hi), n)], sb
