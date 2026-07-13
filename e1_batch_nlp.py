"""
E1 (baseline comparison) tren REAL NLP — IG tren EMBEDDING + cac BASELINE.

Modality NLP cua draft (Sec. Instantiation): IG chay tren TOKEN EMBEDDING (khong phai
token id). Shrinkage la dang COVARIANCE-PCA tren embedding — KHONG co FFT/blur vi truc
"chieu embedding" khong co thu tu (Prop. basis existence). mu,Sigma uoc luong tren mot
REFERENCE SET token embedding; baseline = posterior-mean embedding b_tau(x).

IG (embedding, dot-product form nhu yeu cau):
    phi_token = sum_dim  (x - x0) * mean_alpha  d f / d emb  |_{x0 + alpha (x-x0)}
             = < mean_alpha grad ,  (x - x0) >   (dot-product grad · (emb - baseline))
  gom theo dim embedding cua moi token -> 1 diem attribution / token.

Methods (chi IG + BASELINE, path thang — dung E1):
    IG-zero      : baseline = embedding 0
    IG-pad       : baseline = embedding cua [PAD]
    IG-mask      : baseline = embedding cua [MASK] (in-distribution "token bi che" cua MLM)
    IG-mean      : baseline = mu (trung binh embedding tren reference set)
    IG-random    : baseline = 1 embedding sample ngau nhien tu reference set
    EG-K         : trung binh IG tren K embedding sample (ngan sach chia deu)  K in {1,4,16}
    Shrinkage-IG@tau : b_tau = mu + V diag(s/(s+tau)) V^T (x-mu), quet tau (covariance-PCA)
    PM-IG-PPCA   : psi uoc luong (Cor. MMSE), khong ridge floor

Danh gia: SOFT-FAITHFULNESS (Zhao & Aletras 2023, soft_faith.py):
    Soft-NC (xoa mem token quan trong -> prob sap; cao=tot),
    Soft-NS (giu mem token quan trong -> prob dung vung; cao=tot),
    Soft-gap = NC + NS - 1 (gop hai chieu, kieu I-D gap; cao=faithful),
    Soft-log-odds (cao=tot).
    NC va NS la CAP DOI VE HANH VI (xoa-quan-trong vs giu-quan-trong tren CUNG
    attribution) — faithful can CA HAI cao, nen xep hang theo Soft-gap.
    KHONG tai dung baseline lam mask (soft Bernoulli dropout theo attribution).

Model pretrain (BERT/DistilBERT/RoBERTa fine-tuned), dataset mac dinh sst2 test.
Batch: lay mau `--limit` cau (mac dinh 50). Paired test Shrinkage(tau tot) vs baseline.

Chay (torch GPU mac dinh, tu chay lay):
    python e1_batch_nlp.py --model distilbert --dataset sst2 --limit 50
    python e1_batch_nlp.py --model bert --dataset imdb --tau_sweep 0.1 1 10 100
    python e1_batch_nlp.py --model roberta --dataset rotten --steps 32

KHONG train, KHONG smoketest.
"""

import argparse
import csv
import math
import random
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

from synthetic_e0 import fit_reference, shrinkage_baseline, fit_ppca
from soft_faith import (
    calculate_soft_sufficiency,
    calculate_soft_comprehensiveness,
    calculate_soft_log_odds,
)
from stats_utils import paired_t, wilcoxon, mean_se   # module stat doc lap


MODEL_MAP = {
    "distilbert": {
        "sst2": "distilbert-base-uncased-finetuned-sst-2-english",
        "imdb": "textattack/distilbert-base-uncased-imdb",
        "rotten": "textattack/distilbert-base-uncased-rotten-tomatoes",
    },
    "bert": {
        "sst2": "textattack/bert-base-uncased-SST-2",
        "imdb": "textattack/bert-base-uncased-imdb",
        "rotten": "textattack/bert-base-uncased-rotten-tomatoes",
    },
    "roberta": {
        "sst2": "textattack/roberta-base-SST-2",
        "imdb": "textattack/roberta-base-imdb",
        "rotten": "textattack/roberta-base-rotten-tomatoes",
    },
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="distilbert", choices=list(MODEL_MAP))
    ap.add_argument("--dataset", type=str, default="sst2", choices=["sst2", "imdb", "rotten"])
    ap.add_argument("--limit", type=int, default=50, help="so cau lay mau danh gia")
    ap.add_argument("--steps", type=int, default=64, help="so buoc IG (alpha midpoint)")
    ap.add_argument("--tau_sweep", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0, 100.0])
    ap.add_argument("--eg_K", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--ppca_q", type=int, default=32, help="rank q cho PM-IG-PPCA (embedding D lon)")
    ap.add_argument("--floor", type=float, default=1e-6, help="ridge floor lambda cho Sigma")
    ap.add_argument("--ref_size", type=int, default=4000,
                    help="so token gom tu tap ref de uoc luong mu,Sigma embedding")
    ap.add_argument("--ref_sents", type=int, default=500, help="so cau quet de gom token ref")
    ap.add_argument("--n_soft", type=int, default=10, help="so mau Bernoulli cho soft metric")
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--include_special", action="store_true",
                    help="tinh attribution ca [CLS]/[SEP]/[PAD] (mac dinh bo)")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


# ===========================================================================
# Forward-func theo kien truc: nhan input_embed, tu cong position/type embedding,
# chay ENCODER + classifier, tra logits. Chu ky khop soft_faith.nn_forward_func.
# ===========================================================================
def make_forward_func(model_type):
    """
    Tra ve nn_forward_func(model, input_embed, attention_mask, position_embed, type_embed).
    input_embed la WORD embedding (chua cong position/type); ham nay tu cong roi chay.
    Bat chuoc dung nn_forward_func goc (bert/distilbert/roberta helper):
      embeds = word + position (+ type); LayerNorm + dropout; roi
      model(inputs_embeds=embeds, attention_mask=mask) — HF tu lo encoder + head + mask.
    """
    def fwd_bert(model, input_embed, attention_mask=None, position_embed=None, type_embed=None):
        emb = input_embed + position_embed
        if type_embed is not None:
            emb = emb + type_embed
        emb = model.bert.embeddings.dropout(model.bert.embeddings.LayerNorm(emb))
        return model(inputs_embeds=emb, attention_mask=attention_mask)[0]

    def fwd_distilbert(model, input_embed, attention_mask=None, position_embed=None, type_embed=None):
        emb = input_embed + position_embed
        emb = model.distilbert.embeddings.dropout(model.distilbert.embeddings.LayerNorm(emb))
        return model(inputs_embeds=emb, attention_mask=attention_mask)[0]

    def fwd_roberta(model, input_embed, attention_mask=None, position_embed=None, type_embed=None):
        emb = input_embed + position_embed
        if type_embed is not None:
            emb = emb + type_embed
        emb = model.roberta.embeddings.dropout(model.roberta.embeddings.LayerNorm(emb))
        return model(inputs_embeds=emb, attention_mask=attention_mask)[0]

    return {"bert": fwd_bert, "distilbert": fwd_distilbert, "roberta": fwd_roberta}[model_type]


# ===========================================================================
# Lay word/position/type embedding rieng (de forward-func cong lai + IG interpolate
# CHI tren word embedding, giu position/type co dinh — dung chuan IG cho text).
# ===========================================================================
def get_embeddings(model, model_type, input_ids, attention_mask):
    """
    Tra ve (word_embed, position_embed, type_embed) — moi cai (1,seq,d) hoac None.
    IG se noi suy tren word_embed; position/type giu nguyen (cong trong forward-func).
    """
    dev = input_ids.device
    seq = input_ids.shape[1]
    if model_type == "bert":
        E = model.bert.embeddings
        word = E.word_embeddings(input_ids)
        pos_ids = torch.arange(seq, device=dev).unsqueeze(0)
        position = E.position_embeddings(pos_ids)
        type_ids = torch.zeros_like(input_ids)
        type_e = E.token_type_embeddings(type_ids)
        return word, position, type_e
    if model_type == "distilbert":
        E = model.distilbert.embeddings
        word = E.word_embeddings(input_ids)
        pos_ids = torch.arange(seq, device=dev).unsqueeze(0)
        position = E.position_embeddings(pos_ids)
        return word, position, None
    if model_type == "roberta":
        E = model.roberta.embeddings
        word = E.word_embeddings(input_ids)
        # dung buffer position_ids san co cua model (khop helper goc)
        pos_ids = E.position_ids[:, :seq].to(dev)
        position = E.position_embeddings(pos_ids)
        type_ids = torch.zeros_like(input_ids)
        type_e = E.token_type_embeddings(type_ids)
        return word, position, type_e
    raise ValueError(model_type)


# ===========================================================================
# IG tren embedding (dot-product grad · (emb - baseline)), gom theo token.
# ===========================================================================
def ig_embedding(fwd, model, word_embed, base_embed, position_embed, type_embed,
                 attention_mask, pred_class, steps):
    """
    word_embed, base_embed: (1,seq,d). Tra ve (attr_token (seq,), attr_full_embed (1,seq,d)).
    attr_full_embed = (x-x0) * mean_alpha grad  (giu theo dim, cho soft metric).
    attr_token      = sum_dim attr_full_embed   (dot-product, 1 diem/token).
    """
    device = word_embed.device
    diff = word_embed - base_embed                         # (1,seq,d)
    alphas = ((torch.arange(steps, device=device) + 0.5) / steps).view(-1, 1, 1, 1)  # midpoint
    # states: (steps,1,seq,d)
    states = base_embed[None] + alphas * diff[None]
    grad_acc = torch.zeros_like(word_embed)
    for i in range(steps):
        emb = states[i].clone().requires_grad_(True)       # (1,seq,d)
        logits = fwd(model, emb, attention_mask=attention_mask,
                     position_embed=position_embed, type_embed=type_embed)
        score = logits[0, pred_class]
        grad, = torch.autograd.grad(score, emb)
        grad_acc += grad.detach()
    mean_grad = grad_acc / steps                           # (1,seq,d)
    attr_full = mean_grad * diff                           # (1,seq,d) — giu dim cho soft metric
    attr_token = attr_full.sum(dim=-1).squeeze(0)          # (seq,) dot-product
    return attr_token, attr_full


# ===========================================================================
# Reference set embedding: gom token word-embedding tu nhieu cau -> uoc luong mu,Sigma.
# ===========================================================================
@torch.no_grad()
def build_reference(model, model_type, tokenizer, texts, args, device):
    """Gom toi da ref_size token word-embedding (bo pad) -> (M,d) de fit_reference/PPCA."""
    embed = model.get_input_embeddings()
    pad_id = tokenizer.pad_token_id
    collected = []
    total = 0
    for t in texts[:args.ref_sents]:
        enc = tokenizer(t, truncation=True, max_length=args.max_len, return_tensors="pt").to(device)
        ids = enc["input_ids"]
        w = embed(ids)[0]                                  # (seq,d)
        keep = (ids[0] != pad_id)
        w = w[keep]
        collected.append(w)
        total += w.shape[0]
        if total >= args.ref_size:
            break
    X = torch.cat(collected, dim=0)[:args.ref_size]        # (M,d)
    return X


def main():
    args = parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[!] cuda khong san sang -> cpu"); device = "cpu"
    torch.manual_seed(args.seed); random.seed(args.seed)

    model_name = MODEL_MAP[args.model][args.dataset]
    print(f"[i] model={model_name}  dataset={args.dataset}  limit={args.limit}  steps={args.steps}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    fwd = make_forward_func(args.model)
    embed = model.get_input_embeddings()

    # dataset
    if args.dataset == "sst2":
        ds = load_dataset("glue", "sst2")["test"]
        data = list(zip(ds["sentence"], ds["label"]))
    elif args.dataset == "imdb":
        ds = load_dataset("imdb")["test"]
        data = list(zip(ds["text"], ds["label"]))
    else:
        ds = load_dataset("rotten_tomatoes")["test"]
        data = list(zip(ds["text"], ds["label"]))
    texts_all = [t for t, _ in data]
    sample = random.sample(data, min(args.limit, len(data)))
    print(f"[i] {len(sample)} cau danh gia, ref tu <= {args.ref_sents} cau")

    # reference set embedding -> mu, Sigma (covariance-PCA) + PPCA
    X_ref = build_reference(model, args.model, tokenizer, texts_all, args, device)
    print(f"[i] reference embedding: {tuple(X_ref.shape)} token")
    ref = fit_reference(X_ref, floor=args.floor)
    ppca_ref, psi = fit_ppca(X_ref, q=min(args.ppca_q, X_ref.shape[1] - 1))
    mu = X_ref.mean(dim=0)                                  # (d,)
    print(f"[i] PM-IG-PPCA psi = {psi:.4f}  (rank q={min(args.ppca_q, X_ref.shape[1]-1)})")

    # pad + mask embedding (d,) — baseline-IG "missingness" chuan cua NLP.
    pad_emb_single = embed(torch.tensor([[tokenizer.pad_token_id]], device=device))[0, 0]  # (d,)
    # [MASK] embedding: baseline "chuan nhat" cho BERT-family vi model duoc pretrain MLM
    # tren [MASK] -> chuoi toan [MASK] la in-distribution "token bi che". Neu tokenizer
    # khong co mask_token (hiem), fallback ve pad.
    _mask_id = tokenizer.mask_token_id if tokenizer.mask_token_id is not None else tokenizer.pad_token_id
    mask_emb_single = embed(torch.tensor([[_mask_id]], device=device))[0, 0]                # (d,)
    # S(X,y,0) cua soft_faith = "zeroed out sequence" (Eq.1 Zhao-Aletras 2023), KHONG phai
    # PAD, KHONG phai mu. Day chi la HANG SO CHUAN HOA per-cau (mau so 1-S(X,y,0)), giong
    # nhau cho MOI method tren cung 1 cau -> de base_token_emb=None de soft_faith tu tao
    # zeros_like(input_embed). Perturbation X' (Eq.3-5) la Bernoulli mask nhan thang len
    # embedding theo attribution, KHONG chen baseline nao vao.
    base_token_emb = None

    g = torch.Generator(device="cpu"); g.manual_seed(args.seed + 5)
    rand_idx = torch.randperm(X_ref.shape[0], generator=g)[:max(args.eg_K)]
    rand_pool = X_ref[rand_idx]                             # (maxK, d)

    methods = ["IG-zero", "IG-pad", "IG-mask", "IG-mean", "IG-random"]
    methods += [f"EG-{K}" for K in args.eg_K]
    methods += [f"Shrinkage-IG@{t:g}" for t in args.tau_sweep]
    methods += ["PM-IG-PPCA"]

    def baseline_embed_for(name, word_embed):
        """Tra ve base_embed (1,seq,d) cho method. word_embed: (1,seq,d)."""
        seq = word_embed.shape[1]
        if name == "IG-zero":
            return torch.zeros_like(word_embed)
        if name == "IG-pad":
            return pad_emb_single.view(1, 1, -1).expand(1, seq, -1).contiguous()
        if name == "IG-mask":
            return mask_emb_single.view(1, 1, -1).expand(1, seq, -1).contiguous()
        if name == "IG-mean":
            return mu.view(1, 1, -1).expand(1, seq, -1).contiguous()
        if name == "IG-random":
            return rand_pool[0].view(1, 1, -1).expand(1, seq, -1).contiguous()
        if name.startswith("Shrinkage-IG@"):
            tau = float(name.split("@")[1])
            we = word_embed[0]                              # (seq,d)
            return shrinkage_baseline(we, ref, tau=tau).unsqueeze(0)   # per-token posterior mean
        if name == "PM-IG-PPCA":
            we = word_embed[0]
            return shrinkage_baseline(we, ppca_ref, tau=psi).unsqueeze(0)
        raise ValueError(name)

    # ---- accumulators ----
    metric_keys = ["soft_nc", "soft_ns", "soft_logodds"]
    acc = {m: {k: [] for k in metric_keys} for m in methods}
    per_rows = []
    bl_strength = {}          # {method: {"pf":[], "pb":[], "ratio":[], "shift":[]}}  f(x) vs f(baseline)

    special_ids = set(tokenizer.all_special_ids)

    for si, (text, _label) in enumerate(sample):
        enc = tokenizer(text, truncation=True, max_length=args.max_len, return_tensors="pt").to(device)
        input_ids = enc["input_ids"]
        attn = enc["attention_mask"]
        word_embed, position_embed, type_embed = get_embeddings(model, args.model, input_ids, attn)

        # pred class tren input day du
        with torch.no_grad():
            logits = fwd(model, word_embed, attention_mask=attn,
                         position_embed=position_embed, type_embed=type_embed)
            pred_class = int(logits.argmax(-1).item())

        # mask token thuong (bo special) neu can — dung khi tinh attribution-level score
        keep_tok = torch.ones(input_ids.shape[1], dtype=torch.bool, device=device)
        if not args.include_special:
            for j, tid in enumerate(input_ids[0].tolist()):
                if tid in special_ids:
                    keep_tok[j] = False

        for nm in methods:
            if nm.startswith("EG-"):
                K = int(nm.split("-")[1])
                # EG: trung binh attr_full tren K baseline sample, ngan sach chia deu
                steps_k = max(2, args.steps // K)
                attr_full_acc = torch.zeros_like(word_embed)
                for k in range(K):
                    be = rand_pool[k].view(1, 1, -1).expand_as(word_embed).contiguous()
                    _, af = ig_embedding(fwd, model, word_embed, be, position_embed, type_embed,
                                         attn, pred_class, steps_k)
                    attr_full_acc += af
                attr_full = attr_full_acc / K
            else:
                be = baseline_embed_for(nm, word_embed)
                _, attr_full = ig_embedding(fwd, model, word_embed, be, position_embed, type_embed,
                                            attn, pred_class, args.steps)

                # --- DEBUG: baseline strength f(x) vs f(baseline) (softmax pred_class) ---
                with torch.no_grad():
                    pf = F.softmax(logits, dim=-1)[0, pred_class].item()
                    lb = fwd(model, be, attention_mask=attn,
                             position_embed=position_embed, type_embed=type_embed)
                    pb = F.softmax(lb, dim=-1)[0, pred_class].item()
                    # |b-x| chi tren token thuong (bo special) neu co the
                    diff = (be - word_embed).abs()
                    if not args.include_special and keep_tok.any():
                        shift = diff[0][keep_tok].mean().item()
                    else:
                        shift = diff.mean().item()
                d = bl_strength.setdefault(nm, {"pf": [], "pb": [], "ratio": [], "shift": []})
                d["pf"].append(pf); d["pb"].append(pb)
                d["ratio"].append(pb / pf if pf > 1e-9 else float("nan")); d["shift"].append(shift)

            # attr_token cho soft metric: soft_faith nhan attr theo TOKEN (seq,)
            attr_token = attr_full.sum(dim=-1).squeeze(0)  # (seq,)
            attr_for_metric = attr_token.clone()
            if not args.include_special:
                attr_for_metric[~keep_tok] = attr_for_metric[keep_tok].min() if keep_tok.any() else 0.0

            nc = calculate_soft_comprehensiveness(
                fwd, model, word_embed, position_embed, type_embed, attn,
                attr_for_metric, base_token_emb=base_token_emb, n_samples=args.n_soft)
            ns = calculate_soft_sufficiency(
                fwd, model, word_embed, position_embed, type_embed, attn,
                attr_for_metric, base_token_emb=base_token_emb, n_samples=args.n_soft)
            lo = calculate_soft_log_odds(
                fwd, model, word_embed, position_embed, type_embed, attn,
                attr_for_metric, base_token_emb=base_token_emb, n_samples=args.n_soft)

            acc[nm]["soft_nc"].append(nc)
            acc[nm]["soft_ns"].append(ns)
            acc[nm]["soft_logodds"].append(lo)
            per_rows.append({"idx": si, "method": nm, "soft_nc": nc, "soft_ns": ns, "soft_logodds": lo})

        if (si + 1) % 5 == 0 or si + 1 == len(sample):
            print(f"[{si+1}/{len(sample)}] done")

    # ---- bang tong hop (mean±SE). Soft-gap = NC + NS - 1 (kieu I-D gap cua NLP) ----
    # Soft-NC↑ (xoa quan trong -> sap manh), Soft-NS↑ (giu quan trong -> dung vung).
    # Faithful can CA HAI cao. Gop lai: Soft-gap = NC + NS - 1  (cang cao cang faithful).
    n = len(sample)
    for m in methods:
        acc[m]["soft_gap"] = [nc + ns - 1.0
                              for nc, ns in zip(acc[m]["soft_nc"], acc[m]["soft_ns"])]
    print(f"\n{'='*116}\nKET QUA E1-NLP tren {n} cau  ({args.model}/{args.dataset})")
    print(f"{'method':<20}{'Soft-NC↑':>15}{'Soft-NS↑':>15}{'Soft-gap↑':>15}{'Soft-logodds↑':>18}"
          f"{'f(x)':>8}{'f(xt)':>8}{'ratio':>8}{'|b-x|':>9}")
    print("-" * 116)
    # best theo Soft-gap (tinh truoc de danh dau)
    gap_means = {m: mean_se(acc[m]["soft_gap"])[0] for m in methods}
    best_m = max(gap_means, key=gap_means.get)
    best_gap = gap_means[best_m]
    summary_rows = []
    for m in methods:
        nc_m, nc_se = mean_se(acc[m]["soft_nc"])
        ns_m, ns_se = mean_se(acc[m]["soft_ns"])
        gp_m, gp_se = mean_se(acc[m]["soft_gap"])
        lo_m, lo_se = mean_se(acc[m]["soft_logodds"])
        # f(x)/f(xt)/ratio/|b-x| (EG khong co baseline diem -> "-")
        fx, fxt, rtxt, stxt = "   -  ", "   -  ", "   -  ", "    -   "
        if m in bl_strength:
            d = bl_strength[m]
            if d["pf"]:   fx  = f"{sum(d['pf'])/len(d['pf']):>6.4f}"
            if d["pb"]:   fxt = f"{sum(d['pb'])/len(d['pb']):>6.4f}"
            rr = [r for r in d["ratio"] if r == r]
            if rr:        rtxt = f"{sum(rr)/len(rr):>6.4f}"
            if d["shift"]:stxt = f"{sum(d['shift'])/len(d['shift']):>8.4f}"
        mark = "  <-- best" if m == best_m else ""
        print(f"{m:<20}{nc_m:>8.4f}±{nc_se:<5.4f}{ns_m:>8.4f}±{ns_se:<5.4f}"
              f"{gp_m:>8.4f}±{gp_se:<5.4f}{lo_m:>10.4f}±{lo_se:<6.4f}"
              f"{fx:>8}{fxt:>8}{rtxt:>8}{stxt:>9}{mark}")
        summary_rows.append({"method": m, "n": n,
                             "soft_nc_mean": nc_m, "soft_nc_se": nc_se,
                             "soft_ns_mean": ns_m, "soft_ns_se": ns_se,
                             "soft_gap_mean": gp_m, "soft_gap_se": gp_se,
                             "soft_logodds_mean": lo_m, "soft_logodds_se": lo_se})
    print("-" * 116)
    print(f"[i] dan dau Soft-gap (=NC+NS-1): {best_m} = {best_gap:.4f}")
    print("[i] ratio~1 => baseline chua xoa gi; ratio thap => trung tinh/lat lop (vd IG-zero OOD).")

    # ---- Paired test: Shrinkage(tot nhat theo Soft-gap) vs baseline ----
    shr = [m for m in methods if m.startswith("Shrinkage-IG")]
    refm = max(shr, key=lambda m: mean_se(acc[m]["soft_gap"])[0]) if shr else best_m
    print(f"\n=== PAIRED TEST: {refm} vs baseline (n={n} cau, ghep cap per-sentence) ===")
    stat_rows = []
    for metric, key in [("Soft-NC", "soft_nc"), ("Soft-NS", "soft_ns"),
                        ("Soft-gap", "soft_gap"), ("Soft-logodds", "soft_logodds")]:
        print(f"\n-- {metric} (cao hon tot) --")
        print(f"{'vs method':<20}{'mean_diff':>12}{'t':>9}{'p(t)':>11}{'z(W)':>9}{'p(Wilcox)':>12}")
        print("-" * 73)
        a = acc[refm][key]
        for m in methods:
            if m == refm: continue
            b = acc[m][key]
            md, t, pt = paired_t(a, b)
            W, z, pw = wilcoxon(a, b)
            print(f"{m:<20}{md:>12.4f}{t:>9.3f}{pt:>11.4g}{z:>9.3f}{pw:>12.4g}")
            stat_rows.append({"ref": refm, "vs": m, "metric": metric,
                              "mean_diff": md, "t": t, "p_t": pt, "z_wilcoxon": z, "p_wilcoxon": pw})

    tag = f"{args.model}_{args.dataset}"
    with open(f"e1_nlp_{tag}_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)
    with open(f"e1_nlp_{tag}_paired.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stat_rows[0].keys()))
        w.writeheader(); w.writerows(stat_rows)
    with open(f"e1_nlp_{tag}_perrow.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_rows[0].keys()))
        w.writeheader(); w.writerows(per_rows)
    print(f"\n[i] da luu -> e1_nlp_{tag}_summary.csv, _paired.csv, _perrow.csv")


if __name__ == "__main__":
    main()