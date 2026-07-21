"""
run_regime_check.py — chay tau_regime tren cac file HIEN CO.

Muc dich: kiem "diagnostic PR co khop ket qua khong" cho image / tabular / nlp.

TRUNG THUC:
  - tabular: co curve + id_gap + tai tao duoc Sigma tu wine => check DAY DU (predict+measure).
  - image  : co curve nhung KHONG co Sigma trong file, va khong co id_gap per-tau
             => dung SATURATION PROXY (khong can Sigma). Regime do duoc, KHONG co PR that.
             (PR that phai lay tu e1_batch_image.py khi chay, module san sang nhan.)
  - nlp    : KHONG co curve => chi kiem NO_SIGNAL tu summary. Khong co gi de match.

Chay:  python run_regime_check.py
KHONG train gi, chi doc CSV.
"""
import os
import torch
import tau_regime as tr

UP = "."


def wine_sigma_eigenvalues():
    """Tai tao pho Sigma cua wine giong e1_batch_tabular (standardise + ridge floor)."""
    try:
        from sklearn.datasets import load_wine
    except Exception as e:
        print(f"[!] sklearn khong co ({e}); bo qua PR that cho tabular, dung log da biet.")
        # tu log_wine_wide.txt: PR=4.57, D=13 -> dung placeholder pho khop PR do
        return None
    X = torch.tensor(load_wine().data, dtype=torch.float64)
    X = (X - X.mean(0)) / X.std(0).clamp_min(1e-8)          # standardise
    S = torch.cov(X.T)                                       # (13,13)
    lam = 1e-3                                               # ridge floor giong (5)
    S = 0.5 * (S + S.T) + lam * torch.eye(S.shape[0], dtype=torch.float64)
    s = torch.linalg.eigvalsh(S).clamp_min(0)
    return s


def main():
    entries = []

    # ---------------- TABULAR ----------------
    curve_t = os.path.join(UP, "e1_tabular_wine_taucurve.csv")
    s_wine = wine_sigma_eigenvalues()
    if os.path.exists(curve_t):
        res_t = tr.check_match(ref_s=s_wine, curve_path=curve_t,
                               metric="id_gap", tag="tabular/wine")
        tr.print_regime_check(res_t)
        entries.append(res_t)
    else:
        print(f"[!] khong thay {curve_t}")

    # ---------------- IMAGE ----------------
    curve_i = os.path.join(UP, "e1_image_sigmacurve.csv")
    if os.path.exists(curve_i):
        # KHONG co Sigma trong file => predict=None, measure dung saturation proxy.
        # Khi chay that, truyen ref_s = pho low-rank tu e1_batch_image de co PR.
        res_i = tr.check_match(ref_s=None, curve_path=curve_i, tag="image/benchmark50")
        tr.print_regime_check(res_i)
        print("[i] IMAGE: predict=None (file khong chua Sigma). regime do bang saturation proxy.")
        print("[i]   De co PR that: trong e1_batch_image.py goi")
        print("[i]     tr.check_match(ref_s=ref.s, curve_path=..., tag='image')")
        entries.append(res_i)
    else:
        print(f"[!] khong thay {curve_i}")

    # ---------------- NLP ----------------
    summ_n = os.path.join(UP, "e1_nlp_distilbert_sst2_summary.csv")
    curve_n = os.path.join(UP, "e1_nlp_distilbert_sst2_taucurve.csv")
    if os.path.exists(curve_n):
        res_n = tr.check_match(ref_s=None, curve_path=curve_n,
                               metric="soft_gap",
                               summary_path=summ_n, summary_metric="soft_gap_mean",
                               tag="nlp/sst2")
        tr.print_regime_check(res_n)
        entries.append(res_n)
    elif os.path.exists(summ_n):
        # khong co curve => chi kiem no-signal
        res_n = tr.check_match(ref_s=None, curve_path=None,
                               summary_path=summ_n, summary_metric="soft_gap_mean",
                               tag="nlp/sst2")
        tr.print_regime_check(res_n)
        entries.append(res_n)
        print("[!] NLP: chua co _taucurve.csv. De check DAY DU, chay:")
        print("    python e1_batch_nlp.py --model distilbert --dataset sst2 --limit 50 \\")
        print("      --tau_diag --diag_n 30 --rivals --tau_star")
        print("    (can dam bao script DUMP curve: goi tau_diag.dump_curve_csv)")
    else:
        print(f"[!] khong thay {summ_n}")

    # ---------------- LUAT CROSS-MODALITY ----------------
    tr.print_pr_sensitivity_law(entries)


if __name__ == "__main__":
    main()