"""
tau_regime.py — NOI TAU-SELECTABILITY VOI PHO CUA SIGMA.

=============================================================================
LUAN DIEM (Layer 2 cua framework)

Layer 1 (da chung minh) cho ra HO mot tham so { b_tau }. Con lai DUY NHAT scalar tau.
Cau hoi: tau co phai hyperparameter that khong, hay no bi PHO cua Sigma quyet dinh?

TRA LOI: participation ratio PR = (sum s_k)^2 / sum s_k^2  DU DOAN che do:

  PR THAP  (pho tap trung)  -> distance BAO HOA truoc Δf -> landscape PHANG theo tau
                            -> tau KHONG quan trong (robust). Bat ky tau qua nguong deu duoc.
  PR CAO   (pho trai dai)   -> Δf bao hoa truoc distance -> co KNEE SAC
                            -> rate rule tim duoc tau interior. tau QUAN TRONG.

Day la CUNG mot truc pho ma Cor. 2 cua paper dung de quyet dinh blur co hop le khong.
=> mot diagnostic, ba he qua: blur-admissibility, tau-selectability, tau*-validity.

=============================================================================
MODULE NAY LAM GI

KHONG assert "match". No DO ca hai ve va BAO CO KHOP KHONG:

  du_doan  = f(PR)                          <- tu pho, TRUOC khi chay metric
  do_duoc  = flatness cua metric-vs-tau     <- tu curve file (ORACLE metric)
  match    = du_doan.regime == do_duoc.regime

Neu KHONG match -> in [MISMATCH], KHONG giau. Do la falsification.

NLP: neu ca HO thua IG-zero (khong co tau nao cho tin hieu duong) thi module bao
     "NO_SIGNAL" — KHONG phai match cung KHONG phai mismatch. Diagnostic ve tau
     vo nghia khi khong co optimum de tim. Phai noi ro trong paper.

=============================================================================
KHONG chay gi. Chi doc file va in.
"""
from __future__ import annotations
import csv
import math
from collections import defaultdict

import torch


# ---------------------------------------------------------------------------
# 1. PHO -> participation ratio (dong bo voi tau_diag.participation_ratio)
# ---------------------------------------------------------------------------
def participation_ratio(s: torch.Tensor) -> float:
    """PR = (sum s)^2 / sum s^2. Trong [1, D]. Effective dimensionality cua prior."""
    s = s.clamp_min(0.0).double()
    return float(((s.sum() ** 2) / (s ** 2).sum().clamp_min(1e-30)).item())


def pr_fraction(s: torch.Tensor) -> float:
    """PR / D. Nguong che do dat o day, KHONG o PR tuyet doi (D khac nhau moi modality)."""
    return participation_ratio(s) / s.numel()


# =============================================================================
# CANH BAO — PR DA BI FALSIFY LAM PREDICTOR (2026-07)
#
# Gia thuyet ban dau: PR/D thap => flat, cao => sharp. KIEM TREN 6 diem:
#
#   dataset   known   PR/D      sd(log s)   ket luan
#   -------------------------------------------------
#   wine      flat    39.3%     0.99        PR dung
#   bcancer   flat    12.9%     1.29        PR dung
#   digits    flat    29.8%     1.06        PR dung
#   iris      sharp*  43.5%     0.78        PR SAI (nhung sharp la NHIEU low-D)
#   image     sharp    0.04%    3.88        PR SAI HAN (concentrated ma sharp!)
#
# => KHONG co nguong PR/D nao tach duoc sharp khoi flat. Anh 1/f^2 co PR/D CUC THAP
#    (power don vao vai bin tan so thap) NHUNG lai sharp — NGUOC HAN wine. Va iris
#    "sharp" thuc ra la curve NHIEU (D=4, t=2.98 nhung khong co knee, chi bounce).
#
# sd(log s) tach image (3.88) ra duoc nhung iris (0.78) van pha. KHONG summary pho
# nao doan dung tau-sensitivity xuyen modality.
#
# KET LUAN CHO PAPER: KHONG claim "PR (hay pho) du doan tau matters". Thay vao do:
#   - DO tau-sensitivity truc tiep bang efficiency curve (forward-only, re).
#   - Bao PR nhu mot predictor DA THU VA THAT BAI (falsification-first).
#
# predict_regime() VAN tinh PR + sd(log s) DE GHI LAI, nhung tra ve
# 'regime_predicted' = None va co (khong con nguong). check_match() vi vay se
# lay regime DO DUOC lam su that, va bao PR-vs-do-duoc nhu MOT PHEP THU, khong
# phai matching diagnostic.
# =============================================================================
PR_FRAC_THRESHOLD = 0.5   # GIU LAI chi de tham chieu; KHONG con dung de phan loai.


def sd_log_spectrum(s: torch.Tensor) -> float:
    """Do trai cua log-pho (nats). Cao => trai nhieu decade. Da thu lam predictor: van pha o iris."""
    s = s.clamp_min(1e-12).double()
    ls = s.log()
    w = s / s.sum()
    m = (w * ls).sum()
    return float(((w * ls.pow(2)).sum() - m.pow(2)).clamp_min(0).sqrt().item())


def predict_regime(s: torch.Tensor) -> dict:
    """
    GHI LAI so lieu pho (PR, sd log s). KHONG con doan regime — da falsify.
    'regime_predicted' = None de check_match bao PARTIAL/FALSIFICATION-TEST thay vi
    gia vo co du doan tin cay.
    """
    return {"PR": participation_ratio(s), "D": int(s.numel()),
            "PR_frac": pr_fraction(s), "sd_log_s": sd_log_spectrum(s),
            "regime_predicted": None,          # <- da bo: PR khong doan duoc
            "tau_matters_predicted": None,
            "PR_naive_guess": "flat" if pr_fraction(s) < PR_FRAC_THRESHOLD else "sharp"}


# ---------------------------------------------------------------------------
# 2. CURVE FILE -> do sensitivity THUC (khong nhin pho)
# ---------------------------------------------------------------------------
def load_curve(path: str):
    """Doc <name>_taucurve.csv / _sigmacurve.csv (long format tu dump_curve_csv)."""
    rows = list(csv.DictReader(open(path)))
    D = defaultdict(dict)
    for r in rows:
        D[int(r["i"])][float(r["tau"])] = r
    taus = sorted({float(r["tau"]) for r in rows})
    cols = set(rows[0].keys())
    return D, taus, cols


def _valid_inputs(D, taus):
    return [i for i in D if int(D[i][taus[0]]["valid"])]


def measure_sensitivity(path: str, metric: str | None = None,
                        seed_noise: float = 0.03) -> dict:
    """
    Do PHANG/SAC cua faithfulness-vs-tau tu curve file.

    metric: ten cot metric ORACLE trong file (vd 'id_gap'). Neu None, tu tim.
            Neu file KHONG co metric (chi delta_f/dist) -> dung SATURATION ORDERING
            lam proxy: Δf bao hoa truoc dist => sharp; nguoc lai => flat.

    seed_noise: nguong de goi "interior gain" la THAT hay trong nhieu. Mac dinh 0.03
                (~ id_se dien hinh o tabular). Neu interior_gain < seed_noise => landscape
                coi nhu PHANG du argmax co nhinh len.

    Tra ve dict co 'regime_measured' in {'flat','sharp'} + so lieu.
    """
    D, taus, cols = load_curve(path)
    valid = _valid_inputs(D, taus)
    T = len(taus)

    # tu dong tim cot metric neu khong chi dinh
    if metric is None:
        for cand in ("id_gap", "soft_gap", "insertion", "id_mean"):
            if cand in cols:
                metric = cand
                break

    out = {"n": len(valid), "n_tau": T, "tau_lo": taus[0], "tau_hi": taus[-1],
           "metric": metric}

    if metric is not None and metric in cols:
        # --- co metric THAT: do interior gain vs endpoints ---
        m = [sum(float(D[i][t][metric]) for i in valid) / len(valid) for t in taus]
        peak = max(m); k_peak = m.index(peak)
        base_end = 0.5 * (m[0] + m[-1])
        interior_gain = peak - base_end
        # paired t giua argmax-tau va endpoint-tau (2 diem cu the)
        a = [float(D[i][taus[k_peak]][metric]) for i in valid]
        b = [float(D[i][taus[-1]][metric]) for i in valid]
        diff = [x - y for x, y in zip(a, b)]
        md = sum(diff) / len(diff)
        var = sum((d - md) ** 2 for d in diff) / max(len(diff) - 1, 1)
        se = (var / len(diff)) ** 0.5
        t_stat = md / se if se > 1e-12 else 0.0
        # SAC neu interior optimum vuot nhieu VA co y nghia thong ke
        regime = "sharp" if (interior_gain >= seed_noise and abs(t_stat) >= 2.0) else "flat"
        out.update({"mode": "metric", "argmax_tau": taus[k_peak],
                    "interior_gain": interior_gain, "base_end": base_end,
                    "peak": peak, "t_stat_vs_endpoint": t_stat,
                    "regime_measured": regime})
    else:
        # --- khong co metric: dung saturation ordering cua Δf vs dist ---
        df = [sum(float(D[i][t]["delta_f"]) for i in valid) / len(valid) for t in taus]
        di = [sum(float(D[i][t]["dist"]) for i in valid) / len(valid) for t in taus]

        def sat90(v):
            lo, hi = v[0], v[-1]
            rng = hi - lo
            if abs(rng) < 1e-12:
                return taus[-1]
            for k, x in enumerate(v):
                if (x - lo) / rng >= 0.9:
                    return taus[k]
            return taus[-1]

        t_df, t_di = sat90(df), sat90(di)
        # Δf bao hoa TRUOC dist => con quang duong thua sau khi tin hieu het => KNEE => sharp
        regime = "sharp" if t_df < t_di else "flat"
        out.update({"mode": "saturation_proxy",
                    "df_sat_tau": t_df, "dist_sat_tau": t_di,
                    "regime_measured": regime})
    return out


def detect_no_signal(summary_path: str, metric: str, zero_name: str,
                     shrink_prefix: str = "Shrinkage-IG") -> dict | None:
    """
    NLP guard: neu MOI Shrinkage-IG@tau <= baseline zero tren metric, thi KHONG co
    tau-optimum de tim => diagnostic ve tau vo nghia. Tra ve dict canh bao, hoac None.

    summary_path: file _summary.csv
    metric: cot faithfulness (vd 'soft_gap_mean'). Cao hon = tot hon.
    zero_name: ten hang baseline zero (vd 'IG-zero').
    """
    rows = {r["method"]: r for r in csv.DictReader(open(summary_path))}
    if zero_name not in rows:
        return None
    zero_val = float(rows[zero_name][metric])
    shrink = {k: float(v[metric]) for k, v in rows.items() if k.startswith(shrink_prefix)}
    if not shrink:
        return None
    best_shrink = max(shrink.values())
    if best_shrink <= zero_val:
        return {"no_signal": True, "zero": zero_val, "best_shrink": best_shrink,
                "best_shrink_method": max(shrink, key=shrink.get),
                "metric": metric}
    return None


# ---------------------------------------------------------------------------
# 3. HOP CHUNG: du doan vs do duoc -> MATCH / MISMATCH / NO_SIGNAL
# ---------------------------------------------------------------------------
def check_match(ref_s: torch.Tensor | None, curve_path: str | None,
                metric: str | None = None,
                summary_path: str | None = None,
                summary_metric: str | None = None,
                zero_name: str = "IG-zero",
                tag: str = "") -> dict:
    """
    Tra ve dict day du. In bang bang print_regime_check().

    ref_s      : eigenvalues cua Sigma (cho du doan tu PR). None -> bo qua ve du doan.
    curve_path : curve CSV (cho do sensitivity). None -> bo qua ve do duoc.
    summary_path/summary_metric: neu cung cap, kiem NO_SIGNAL truoc (NLP).
    """
    res = {"tag": tag}

    # 0) no-signal guard (NLP)
    if summary_path and summary_metric:
        ns = detect_no_signal(summary_path, summary_metric, zero_name)
        if ns:
            res["status"] = "NO_SIGNAL"
            res["no_signal"] = ns
            return res

    # 1) du doan tu pho
    if ref_s is not None:
        res["predict"] = predict_regime(ref_s)

    # 2) do tu curve
    if curve_path:
        res["measure"] = measure_sensitivity(curve_path, metric=metric)

    # 3) DO DUOC la su that. PR chi la phep THU (da falsify).
    if "measure" in res:
        res["status"] = "MEASURED"                    # co regime do duoc
        if "predict" in res:
            # PR-naive-guess co trung voi do duoc khong? (chi de ghi lai, KHONG ket luan)
            guess = res["predict"]["PR_naive_guess"]
            meas = res["measure"]["regime_measured"]
            res["pr_test"] = "AGREE" if guess == meas else "DISAGREE"
    else:
        res["status"] = "PARTIAL"                      # khong co curve
    return res


def print_regime_check(res: dict):
    tag = res.get("tag", "")
    print(f"\n=== TAU-REGIME CHECK {tag} ===")
    st = res["status"]

    if st == "NO_SIGNAL":
        ns = res["no_signal"]
        print(f"[!!] NO_SIGNAL: moi Shrinkage-IG@tau <= {ns['zero']:.4f} (baseline zero) tren {ns['metric']}.")
        print(f"[!!]  best Shrinkage = {ns['best_shrink']:.4f} ({ns['best_shrink_method']}) <= zero.")
        print(f"[!!]  => KHONG co tau-optimum. Diagnostic ve tau VO NGHIA o modality nay.")
        print(f"[!!]  Bao cao rieng: day la thua ve MODEL/METRIC, khong phai ve chon tau.")
        return

    # DO DUOC = su that
    if "measure" in res:
        m = res["measure"]
        if m.get("mode") == "metric":
            print(f"[i] DO DUOC (curve, metric={m['metric']}, n={m['n']}):  "
                  f"regime = '{m['regime_measured'].upper()}'")
            print(f"[i]   argmax tau={m['argmax_tau']:.3g}, interior gain={m['interior_gain']:+.4f}, "
                  f"paired t vs endpoint={m['t_stat_vs_endpoint']:.2f}")
        else:
            print(f"[i] DO DUOC (saturation proxy, n={m['n']}):  "
                  f"regime = '{m['regime_measured'].upper()}'")
            print(f"[i]   Δf bao hoa @tau={m['df_sat_tau']:.3g}, dist @tau={m['dist_sat_tau']:.3g}")

    # PHO = ghi lai + phep THU PR (da falsify)
    if "predict" in res:
        p = res["predict"]
        print(f"[i] PHO (ghi lai): PR={p['PR']:.2f}/D={p['D']} (PR/D={p['PR_frac']*100:.2f}%), "
              f"sd(log s)={p['sd_log_s']:.3f}")
        if "pr_test" in res:
            print(f"[i]   PR-naive-guess = '{p['PR_naive_guess']}'  vs do duoc "
                  f"'{res['measure']['regime_measured']}'  ->  PR test: {res['pr_test']}")
            if res["pr_test"] == "DISAGREE":
                print(f"[!!]  PR SAI o day (da biet: PR khong doan duoc tau-sensitivity xuyen modality).")

    if st == "PARTIAL":
        print(f"[i] PARTIAL: chua co curve => chua do duoc regime. Chay batch script de sinh curve.")


# ---------------------------------------------------------------------------
# 4. BANG CROSS-MODALITY: nhieu diem PR -> co monotonic khong?
# ---------------------------------------------------------------------------
def print_pr_sensitivity_law(entries: list[dict]):
    """
    Bang FALSIFICATION: PR (va sd log s) co doan dung regime DO DUOC khong?

    KET QUA da biet (6 diem): KHONG. Ham nay in bang de nguoi doc TU THAY:
    khong nguong PR/D nao tach sharp/flat, va sd(log s) cung pha o iris.
    Ket luan cho paper: DO tau-sensitivity truc tiep, dung DU DOAN tu pho.
    """
    print(f"\n=== FALSIFICATION: PHO CO DOAN DUOC TAU-SENSITIVITY KHONG? ===")
    print(f"{'modality':<14}{'PR/D':>9}{'sd(logs)':>10}{'PR-guess':>10}"
          f"{'gain':>10}{'MEASURED':>10}{'PR test':>10}")
    print("-" * 73)
    agree = 0; total = 0
    for e in entries:
        tag = e.get("tag", "?")
        if e["status"] == "NO_SIGNAL":
            print(f"{tag:<14}{'-':>9}{'-':>10}{'-':>10}{'NO_SIGNAL':>10}{'-':>10}{'-':>10}")
            continue
        p = e.get("predict", {})
        m = e.get("measure", {})
        prf = p.get("PR_frac"); sdl = p.get("sd_log_s")
        guess = p.get("PR_naive_guess", "-")
        gain = m.get("interior_gain"); rm = m.get("regime_measured", "-")
        test = e.get("pr_test", "-")
        prf_s = f"{prf*100:.1f}%" if prf is not None else "-"
        sdl_s = f"{sdl:.2f}" if sdl is not None else "-"
        gain_s = f"{gain:+.4f}" if gain is not None else "(proxy)"
        print(f"{tag:<14}{prf_s:>9}{sdl_s:>10}{guess:>10}{gain_s:>10}{rm:>10}{test:>10}")
        if test in ("AGREE", "DISAGREE"):
            total += 1
            agree += (test == "AGREE")
    print("-" * 73)
    if total:
        print(f"[i] PR-naive-guess trung do duoc: {agree}/{total}.")
        if agree < total:
            print(f"[!!] PR DISAGREE o {total-agree}/{total} diem => PR KHONG phai predictor tin cay.")
            print(f"[!!]  Cu the: anh 1/f^2 co PR/D CUC THAP nhung SHARP (nguoc wine).")
            print(f"[!!]  => PAPER: DO tau-sensitivity bang efficiency curve, KHONG doan tu pho.")
    else:
        print(f"[i] chua co diem nao co ca pho + curve do duoc. Chay batch scripts truoc.")


if __name__ == "__main__":
    print(__doc__)
    print("Module cung cap ham. Xem vi du chay trong run_regime_check.py")
