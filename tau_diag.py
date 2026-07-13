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

  (3) tau tai (hoac ngay truoc) DIEM GAY cua Δf
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

Ca ba modality khop. Do la ly do bo SNR va giu ti gia bien.

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
    fwd = dd / ds                                        # (M, T-1)
    r = torch.zeros_like(delta_f)
    r[:, :-1] = fwd
    r[:, -1] = fwd[:, -1]
    return r


# ---------------------------------------------------------------------------
# Quet dense — batch duoc (tabular, vision)
# ---------------------------------------------------------------------------
@torch.no_grad()
def sweep_curve(X_eval, score_fn, baseline_fn, taus,
                mu=None, ref_s=None, ref_V=None, ref_mu=None, n_class=None):
    """
    X_eval    : (M, ...)
    score_fn  : (B,...) -> (B,)  xac suat lop target
    baseline_fn: (x_single, tau) -> baseline
    taus      : list[float], NEN dense (>= 25 diem)
    n_class   : de tinh 'amp' = bien do con lai so voi san 1/K (scale-free theo K)

    Tra ve dict: (M,T) rho, delta_f, dist, rate, dist_norm, amp, maha_b
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
    d_mu = ((X_eval.reshape(M, -1) - mu.reshape(1, -1)).norm(dim=1).clamp_min(1e-12)
            if mu is not None else torch.ones(M, device=dev))

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
    rate = _marginal_rate(delta_f, dist)                 # <-- DAI LUONG CHINH
    dist_norm = dist / d_mu[:, None]                     # 1.0 = da toi mu; >1 = di QUA mu

    if n_class:
        fl = 1.0 / n_class
        amp = (((rho * f_x[:, None]) - fl) / (f_x[:, None] - fl).clamp_min(1e-12)).clamp(0, 1)
    else:
        amp = torch.full_like(rho, float("nan"))

    valid = _validate(f_x, f_mu, M, mu is not None)
    return {"taus": torch.tensor(taus, device=dev),
            "rho": rho, "delta_f": delta_f, "dist": dist, "rate": rate,
            "dist_norm": dist_norm, "amp": amp, "maha_b": maha_b,
            "f_x": f_x, "f_mu": f_mu, "maha_x": maha_x, "valid": valid}


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
                       mu_baseline=None, maha_one=None, mask_one=None, n_class=None):
    """
    Giong sweep_curve nhung loop tung example.
    dist = ||.||_2 tren cac coordinate DUOC GIU (mask_one), KHONG phai .abs().mean()
    nhu code cu (do la L1/D, khong phai quang duong Euclid).
    """
    M, T = len(examples), len(taus)
    rho = torch.zeros(M, T); dist = torch.zeros(M, T)
    maha_b = torch.full((M, T), float("nan"))
    f_x = torch.zeros(M); f_mu = torch.full((M,), float("nan"))
    maha_x = torch.full((M,), float("nan")); d_mu = torch.ones(M)

    def _sub(t, m):
        return (t[0][m] if (m is not None and t.dim() == 3) else t).reshape(-1)

    for i, it in enumerate(examples):
        x = embed_of(it)
        f_x[i] = max(float(score_one(it, x)), 1e-12)
        m = mask_one(it) if mask_one is not None else None
        if maha_one is not None:
            maha_x[i] = float(maha_one(it, x))
        if mu_baseline is not None:
            bmu = mu_baseline(it)
            f_mu[i] = float(score_one(it, bmu))
            d_mu[i] = max(_sub(x - bmu, m).norm().item(), 1e-12)
        for t_i, tau in enumerate(taus):
            b = baseline_one(it, x, tau)
            rho[i, t_i] = float(score_one(it, b)) / f_x[i]
            dist[i, t_i] = _sub(x - b, m).norm().item()
            if maha_one is not None:
                maha_b[i, t_i] = float(maha_one(it, b))

    delta_f = f_x[:, None] * (1.0 - rho)
    rate = _marginal_rate(delta_f, dist)
    dist_norm = dist / d_mu[:, None]
    if n_class:
        fl = 1.0 / n_class
        amp = (((rho * f_x[:, None]) - fl) / (f_x[:, None] - fl).clamp_min(1e-12)).clamp(0, 1)
    else:
        amp = torch.full_like(rho, float("nan"))

    valid = _validate(f_x, f_mu, M, mu_baseline is not None)
    return {"taus": torch.tensor([float(t) for t in taus]),
            "rho": rho, "delta_f": delta_f, "dist": dist, "rate": rate,
            "dist_norm": dist_norm, "amp": amp, "maha_b": maha_b,
            "f_x": f_x, "f_mu": f_mu, "maha_x": maha_x, "valid": valid}


# ---------------------------------------------------------------------------
# Selection rules
# ---------------------------------------------------------------------------
def selection_rules(curve: dict, eps: float = 0.01):
    """
    Tra ve (rules, valid).

    tau_rate  : *** RULE CHINH ***
                tau LON NHAT ma ti gia bien r(tau) >= eps * max_t r(t).
                "Di tiep chung nao con mua duoc du tin hieu tren moi don vi duong."
                eps la nguong TUONG DOI (1% ti gia cuc dai) — no chi noi "ti gia
                da tut 100 lan", khong phai mot hang so tuyet doi phai tune.
                  vision : ti gia sap 1000x sau @8   -> dung @8
                  NLP    : ti gia AM sau @1          -> dung @1
                  tabular: ti gia VAN TANG tai @100  -> di tiep (grid cut dau)
                Neu rule roi vao DIEM CUOI grid => grid cut dau, phai noi rong.

    tau_knee  : DOI CHUNG. Kneedle tren (log tau, Δf) — diem gay.
                SAI o tabular (chon @1, best la @100). Giu lai de THAY RO no sai,
                chu khong phai de dung.

    tau_amp   : DOI CHUNG. tau nho nhat con <=10% bien do ban dau so voi san 1/K.
                Scale-free theo K (chay ca K=2 lan K=1000) nhung VAN co tham so.
    """
    taus, dev = curve["taus"], curve["taus"].device
    M, T = curve["rho"].shape
    valid = curve.get("valid", torch.ones(M, dtype=torch.bool, device=dev))
    rate, df = curve["rate"], curve["delta_f"]
    out = {}

    # --- tau_rate (CHINH) ---
    # CAN THAN OFF-BY-ONE: rate[t] la ti gia cua KHOANG [t, t+1], khong phai cua
    # DIEM t. Nen "khoang cuoi cung con tot" la [t*, t*+1], va tau can chon la
    # DIEM CUOI cua khoang do, tuc t*+1 — khong phai t*.
    #
    # Ban dau lay t* -> TRUOT MOT BUOC o NLP: chon @0.1 trong khi best la @1.
    # Ly do: rate[@1] = -0.089 mo ta doan @1->@10 (xau), nhung DEN duoc @1 thi
    # van tot (rate[@0.1] = +0.162). Loai @1 la sai — no bi loai vi doan SAU no
    # xau, chu khong phai vi den no la xau.
    #
    # Sau khi +1:  vision @4->@8 (OK), NLP @0.1->@1 (OK), tabular @100 (clamp, OK).
    rmax = rate.clamp_min(0).max(dim=1, keepdim=True).values.clamp_min(1e-12)
    ok = rate >= eps * rmax                                       # (M,T) khoang [t,t+1] con dang di
    ar = torch.arange(T, device=dev, dtype=torch.float)[None]
    idx_last = (ok.float() * ar).argmax(dim=1)                    # khoang tot cuoi cung
    idx_last = torch.where(ok.any(1), idx_last, torch.zeros_like(idx_last))
    idx_star = (idx_last + 1).clamp_max(T - 1)                    # DIEM CUOI cua khoang do
    out["tau_rate"] = taus[idx_star]
    at_edge = float((idx_star[valid] == T - 1).float().mean().item()) if int(valid.sum()) else 0.0

    # --- tau_knee (doi chung) ---
    log_tau = taus.log()
    xx = (log_tau - log_tau.min()); xx = xx / xx.max().clamp_min(1e-12)
    idx = []
    for i in range(M):
        y = torch.cummax(df[i].clamp_min(0), dim=0).values        # ep tang don dieu
        yy = (y - y.min()); yy = yy / yy.max().clamp_min(1e-12)
        idx.append(int((yy - xx).argmax().item()))                # Kneedle cho duong TANG
    out["tau_knee"] = taus[torch.tensor(idx, device=dev)]

    # --- tau_amp (doi chung) ---
    if not torch.isnan(curve["amp"]).all():
        hit = curve["amp"] <= 0.10
        first = torch.where(hit.any(1), hit.float().argmax(1),
                            torch.full((M,), T - 1, device=dev, dtype=torch.long))
        out["tau_amp"] = taus[first]

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
    print(f"{'tau':>10}{'f(b)':>9}{'Δf':>9}{'|b-x|₂':>10}{'|b-x|/|x-mu|':>14}"
          f"{'amp':>8}{'d(Δf)/d|b-x|':>15}")
    print("-" * 76)
    for i, t in enumerate(taus):
        fb = (curve["rho"][:, i] * fx).mean().item()
        df_ = curve["delta_f"][:, i].mean().item()
        d_ = curve["dist"][:, i].mean().item()
        dn_ = curve["dist_norm"][:, i].mean().item()
        r_ = curve["rate"][:, i].mean().item()
        a = curve["amp"][:, i]
        a_s = f"{a.mean().item():>8.3f}" if not torch.isnan(a).all() else f"{'-':>8}"
        print(f"{t:>10.4g}{fb:>9.4f}{df_:>9.4f}{d_:>10.4f}{dn_:>14.3f}{a_s}{r_:>15.5g}")
    print("-" * 76)
    print("[i] d(Δf)/d|b-x| = TI GIA BIEN: di them 1 don vi quang duong -> mua duoc bao nhieu tin hieu.")
    print("[i]   Dung khi ti gia tut. KHONG chia cho |b-x|² (do la SNR — hong o D lon: mau so ~1e4,")
    print("[i]   tu so <=1, in ra toan 0.0000; va Δf cham tran => SNR thanh 1/|b-x|², vo nghia).")
    print("[i] |b-x|/|x-mu| : 1.0 = da toi mu. >1 = di QUA mu (zero/black o vision ~2.0).")
    print("[i] amp = (f(b)-1/K)/(f(x)-1/K) : bien do con lai so voi san uniform. Scale-free theo K.")


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
    print("[i] tau_rate = RULE CHINH (ti gia bien). tau_knee / tau_amp = DOI CHUNG.")
    print("[i] tau_knee SAI o tabular: no chon diem gay, nhung o tabular ||b-x|| bao hoa CUNG LUC")
    print("[i]   voi Δf (ca hai hoi tu khi b->mu) => khong co quang duong thua => khong co ly do")
    print("[i]   dung o gay. Best that = cuoi grid (@100). Giu tau_knee de THAY RO no sai.")
    if oracle is not None:
        print("[i] Neu tau_rate ~ ORACLE => chon duoc tau MA KHONG cham test metric. Do la ket qua chinh.")
    if at_edge is not None and at_edge > 0.3:
        print(f"[!!] {at_edge*100:.0f}% input co tau_rate = CUOI GRID  =>  GRID CUT DAU.")
        print(f"[!!]  Ti gia bien VAN CON CAO o tau lon nhat: baseline chua di het duong.")
        print(f"[!!]  Noi rong grid roi chay lai. (Dung y het log tabular: @100 van dan dau,")
        print(f"[!!]  Δf van dang tang, ti gia @10->@100 = 9.27 > ti gia @1->@10 = 5.86.)")


def dump_curve_csv(curve: dict, path: str, extra_cols: dict | None = None):
    """
    Long format: mot dong = (input i, tau t).
    Day la file de FIT rule offline — dung suy dien tiep tu 4-5 diem grid decade nua.
    """
    taus = curve["taus"].tolist()
    M, T = curve["rho"].shape
    cols = ["i", "tau", "f_x", "f_b", "rho", "delta_f", "dist", "dist_norm",
            "amp", "rate", "maha_x", "maha_b", "valid"]
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
                       "rho": curve["rho"][i, t].item(),
                       "delta_f": curve["delta_f"][i, t].item(),
                       "dist": curve["dist"][i, t].item(),
                       "dist_norm": curve["dist_norm"][i, t].item(),
                       "amp": curve["amp"][i, t].item(),
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
def fixed_baseline_diag(X_eval, score_fn, b_fn, mu=None,
                        ref_s=None, ref_V=None, ref_mu=None, n_class=None):
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
    out = {"f_x": f_x, "f_b": f_b, "rho": f_b / f_x, "delta_f": f_x - f_b, "dist": dist}
    if mu is not None:
        d_mu = (X_eval.reshape(M, -1) - mu.reshape(1, -1)).norm(dim=1).clamp_min(1e-12)
        out["dist_norm"] = dist / d_mu
    if n_class:
        fl = 1.0 / n_class
        out["amp"] = ((f_b - fl) / (f_x - fl).clamp_min(1e-12)).clamp(0, 1)
    if ref_s is not None:
        def mh(Z):
            c = (Z.reshape(Z.shape[0], -1) - ref_mu.reshape(1, -1)) @ ref_V
            return ((c ** 2) / ref_s.clamp_min(1e-12)[None]).sum(1).sqrt()
        out["maha_b"] = mh(B); out["maha_x"] = mh(X_eval)
        out["P2_ok"] = (out["maha_b"] <= out["maha_x"]).float()
    return out


def print_fixed_header():
    print(f"\n{'fixed baseline':<20}{'f(b)':>9}{'Δf':>9}{'|b-x|₂':>10}"
          f"{'|b-x|/|x-mu|':>14}{'amp':>8}{'|b-mu|_S⁻¹':>13}{'P2':>7}")
    print("-" * 90)


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
          f"{f(g('dist_norm'),14,3)}{f(g('amp'),8,3)}{f(g('maha_b'),13)}{p2}")
