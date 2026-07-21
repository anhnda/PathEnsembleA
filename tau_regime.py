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


# nguong: PR/D < 0.5 => tap trung => du doan PHANG.  >= 0.5 => trai dai => du doan SAC.
# 0.5 la ranh gioi CO THE TRANH CAI. Chon vi:
#   wine PR/D = 4.57/13 = 0.352  -> concentrated -> flat  (khop t=0.68)
#   anh 1/f^2 PR/D lon (pho trai) -> spread -> sharp     (khop rate rule work)
# Neu co diem trung gian (PR/D ~ 0.5) ma sensitivity KHONG chuyen tiep muot
# => nguong 0.5 sai HOAC PR khong phai truc dung. Bao ro.
PR_FRAC_THRESHOLD = 0.5


def predict_regime(s: torch.Tensor) -> dict:
    """Du doan che do tau CHI tu pho. Chua nhin metric."""
    prf = pr_fraction(s)
    regime = "sharp" if prf >= PR_FRAC_THRESHOLD else "flat"
    return {"PR": participation_ratio(s), "D": int(s.numel()), "PR_frac": prf,
            "regime_predicted": regime,
            "tau_matters_predicted": (regime == "sharp")}


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

    # 3) so khop
    if "predict" in res and "measure" in res:
        rp = res["predict"]["regime_predicted"]
        rm = res["measure"]["regime_measured"]
        res["status"] = "MATCH" if rp == rm else "MISMATCH"
    else:
        res["status"] = "PARTIAL"  # thieu mot ve
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

    if "predict" in res:
        p = res["predict"]
        print(f"[i] DU DOAN (tu pho): PR = {p['PR']:.2f} / D = {p['D']}  "
              f"(PR/D = {p['PR_frac']*100:.1f}%)")
        print(f"[i]   PR/D {'>=' if p['PR_frac']>=PR_FRAC_THRESHOLD else '<'} "
              f"{PR_FRAC_THRESHOLD:.0%}  =>  regime = '{p['regime_predicted'].upper()}'  "
              f"(tau {'MATTERS' if p['tau_matters_predicted'] else 'is ROBUST'})")

    if "measure" in res:
        m = res["measure"]
        if m.get("mode") == "metric":
            print(f"[i] DO DUOC (tu curve, metric={m['metric']}, n={m['n']}):")
            print(f"[i]   argmax tau = {m['argmax_tau']:.3g}, interior gain over endpoints = "
                  f"{m['interior_gain']:+.4f}")
            print(f"[i]   paired t (argmax-tau vs endpoint) = {m['t_stat_vs_endpoint']:.2f}  "
                  f"=>  regime = '{m['regime_measured'].upper()}'")
        else:
            print(f"[i] DO DUOC (saturation proxy, n={m['n']}):")
            print(f"[i]   Δf bao hoa @tau={m['df_sat_tau']:.3g}, dist bao hoa @tau={m['dist_sat_tau']:.3g}")
            print(f"[i]   {'Δf-first => KNEE' if m['regime_measured']=='sharp' else 'dist-first => FLAT'}"
                  f"  =>  regime = '{m['regime_measured'].upper()}'")

    if st == "MATCH":
        print(f"[OK] MATCH: du doan PR va sensitivity do duoc TRUNG KHOP.")
    elif st == "MISMATCH":
        print(f"[!!] MISMATCH: PR du doan '{res['predict']['regime_predicted']}' nhung do duoc "
              f"'{res['measure']['regime_measured']}'.")
        print(f"[!!]  => PR KHONG phai truc dung o day, HOAC nguong {PR_FRAC_THRESHOLD} sai. FALSIFICATION.")
    else:
        print(f"[i] PARTIAL: thieu mot ve (pho hoac curve). Cung cap ca hai de so khop.")


# ---------------------------------------------------------------------------
# 4. BANG CROSS-MODALITY: nhieu diem PR -> co monotonic khong?
# ---------------------------------------------------------------------------
def print_pr_sensitivity_law(entries: list[dict]):
    """
    entries: list cac dict tu check_match() da co ca predict+measure.
    In bang PR/D vs interior_gain de kiem LUAT: sensitivity co tang theo PR khong?

    2 diem = mot duong thang qua 2 cham. Can >= 4 diem trai PR de goi la LUAT.
    Ham nay in ro con THIEU bao nhieu diem.
    """
    print(f"\n=== LUAT PR -> TAU-SENSITIVITY (can >= 4 diem) ===")
    print(f"{'modality':<16}{'PR/D':>8}{'regime_pred':>14}{'gain_measured':>15}{'regime_meas':>14}{'match':>8}")
    print("-" * 76)
    pts = []
    for e in entries:
        tag = e.get("tag", "?")
        if e["status"] == "NO_SIGNAL":
            print(f"{tag:<16}{'-':>8}{'-':>14}{'NO_SIGNAL':>15}{'-':>14}{'-':>8}")
            continue
        prf = e.get("predict", {}).get("PR_frac")
        rp = e.get("predict", {}).get("regime_predicted", "-")
        meas = e.get("measure", {})
        gain = meas.get("interior_gain")
        rm = meas.get("regime_measured", "-")
        prf_s = f"{prf*100:.1f}%" if prf is not None else "-"
        gain_s = f"{gain:+.4f}" if gain is not None else "(proxy)"
        print(f"{tag:<16}{prf_s:>8}{rp:>14}{gain_s:>15}{rm:>14}{e['status']:>8}")
        if prf is not None and gain is not None:
            pts.append((prf, gain))
    print("-" * 76)
    n = len(pts)
    if n >= 2:
        # spearman tho tren (PR/D, gain)
        xs = sorted(range(n), key=lambda i: pts[i][0])
        ys = sorted(range(n), key=lambda i: pts[i][1])
        rx = [0]*n; ry = [0]*n
        for r, i in enumerate(xs): rx[i] = r
        for r, i in enumerate(ys): ry[i] = r
        dsq = sum((rx[i]-ry[i])**2 for i in range(n))
        rho = 1 - 6*dsq/(n*(n*n-1)) if n > 2 else float("nan")
        print(f"[i] {n} diem co ca PR va gain. Spearman(PR/D, interior_gain) = {rho:.3f}"
              if n > 2 else f"[i] chi {n} diem — chua tinh duoc correlation.")
    if n < 4:
        print(f"[!!] CHI {n} DIEM. 'PR du doan sensitivity' con la duong qua {n} cham, CHUA phai luat.")
        print(f"[!!]  Can them {4-n} dataset trai PR (nhat la vung trung gian PR/D ~ 0.5) truoc khi")
        print(f"[!!]  claim monotonic trong paper. Nếu KHÔNG monotonic khi them diem => PR sai truc.")


if __name__ == "__main__":
    print(__doc__)
    print("Module cung cap ham. Xem vi du chay trong run_regime_check.py")
