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
import tau_diag
from soft_faith import (
    calculate_soft_sufficiency,
    calculate_soft_comprehensiveness,
    calculate_soft_log_odds,
)
from stats_utils import paired_t, wilcoxon, mean_se   # module stat doc lap
from pea.baselines_rival import (
    ig2_nlp, prep_ref_embed, max_entropy_embed, fringe_nlp,
)


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
    # --- TAU-DIAGNOSTIC ---
    ap.add_argument("--tau_diag", action="store_true",
                    help="quet DENSE tau, log Δf/|b-x|₂/TI GIA BIEN per-cau + tau_rate (chi forward pass)")
    ap.add_argument("--diag_n", type=int, default=25)
    ap.add_argument("--diag_gamma", action="store_true", help="grid scale-free tau = gamma*s_bar")
    ap.add_argument("--diag_eps", type=float, default=0.01)
    ap.add_argument("--n_soft", type=int, default=10, help="so mau Bernoulli cho soft metric")
    ap.add_argument("--rivals", action="store_true", help="bat IG2 / Max-Entropy / FRInGe")
    ap.add_argument("--ig2_steps", type=int, default=40)
    ap.add_argument("--me_steps", type=int, default=100)
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
    if args.rivals:
        methods += ["IG-MaxEnt", "FRInGe", "IG2"]

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

    # =====================================================================
    # TAU-DIAGNOSTIC (NLP): quet dense, per-CAU. Chi forward pass.
    # LUU Y NLP: log hien tai cho IG-zero THANG, moi Shrinkage deu thua, va
    # ratio(zero)=0.48 ~ ratio(mean)=0.495. Nghi van: zero-embedding KHONG
    # off-distribution nhu gia dinh => (P2) co the KHONG bi vi pham => hang
    # "zero ✗" trong Table 1 SAI o modality nay. Cot maha_b/P2 tra loi cai do.
    # =====================================================================
    if args.tau_diag:
        PR = tau_diag.participation_ratio(ref.s)
        d_emb = X_ref.shape[1]
        s_bar = tau_diag.effective_tau_scale(ref.s, "mean")
        print(f"\n=== PHO CUA SIGMA (embedding) ===")
        print(f"[i] d = {d_emb},  PR = {PR:.2f}  ({PR/d_emb*100:.1f}% cua d)")
        print(f"[i] s_bar = {s_bar:.6g}  s_max = {ref.s.max():.6g}  s_min = {ref.s.min():.6g}")
        if PR >= 0.2 * d_emb:
            print("[!] PR LON => knee MO => rule chon tau se bat dinh o modality nay.")

        if args.diag_gamma:
            taus_d, sb = tau_diag.gamma_grid(ref.s, 1e-2, 1e2, args.diag_n)
            print(f"[i] grid scale-free tau = gamma*s_bar, s_bar={sb:.6g}")
        else:
            taus_d = tau_diag.log_tau_grid(1e-2 * s_bar, 1e2 * s_bar, args.diag_n)

        # chuan bi cac example (encode 1 lan)
        exs = []
        for text, _l in sample:
            enc = tokenizer(text, truncation=True, max_length=args.max_len,
                            return_tensors="pt").to(device)
            we, pe, te = get_embeddings(model, args.model, enc["input_ids"], enc["attention_mask"])
            with torch.no_grad():
                lg = fwd(model, we, attention_mask=enc["attention_mask"],
                         position_embed=pe, type_embed=te)
                pc = int(lg.argmax(-1).item())
            kt = torch.ones(enc["input_ids"].shape[1], dtype=torch.bool, device=device)
            if not args.include_special:
                for j, tid in enumerate(enc["input_ids"][0].tolist()):
                    if tid in set(tokenizer.all_special_ids):
                        kt[j] = False
            exs.append({"we": we, "pe": pe, "te": te, "attn": enc["attention_mask"],
                        "pc": pc, "keep": kt})

        @torch.no_grad()
        def _score_one(it, emb):
            lg = fwd(model, emb, attention_mask=it["attn"],
                     position_embed=it["pe"], type_embed=it["te"])
            return F.softmax(lg, -1)[0, it["pc"]].item()

        @torch.no_grad()
        def _maha_one(it, emb):
            # ||b - mu||_{Sigma^-1} trung binh tren token thuong
            z = emb[0][it["keep"]] if it["keep"].any() else emb[0]
            c = (z - ref.mu[None]) @ ref.V
            return (((c ** 2) / ref.s.clamp_min(1e-12)[None]).sum(1).sqrt()).mean().item()

        curve = tau_diag.sweep_curve_varlen(
            exs,
            score_one=_score_one,
            embed_of=lambda it: it["we"],
            baseline_one=lambda it, x, t: shrinkage_baseline(x[0], ref, tau=t).unsqueeze(0),
            taus=taus_d,
            mu_baseline=lambda it: mu.view(1, 1, -1).expand(1, it["we"].shape[1], -1).contiguous(),
            maha_one=_maha_one,
            mask_one=lambda it: it["keep"],
        )
        tau_diag.print_curve_table(curve, tag=f"[nlp/{args.model}/{args.dataset}]")

        # cac baseline CO DINH, cung don vi -> tra loi cau hoi zero co vi pham (P2) khong
        print(f"\n{'fixed baseline':<20}{'f(x)':>9}{'f(b)':>9}{'Δf':>10}{'|b-x|₂':>11}"
              f"{'|b-mu|_S⁻¹':>13}{'P2':>7}")
        print("-" * 80)
        maha_x_mean = curve["maha_x"].mean().item()
        for nm in ["IG-zero", "IG-pad", "IG-mask", "IG-mean", "IG-random"]:
            fxs, fbs, dds, shs, mms, p2 = [], [], [], [], [], []
            for it in exs:
                x = it["we"]
                b = baseline_embed_for(nm, x)
                fx = _score_one(it, x); fb = _score_one(it, b)
                sh = ((x - b)[0][it["keep"]] if it["keep"].any() else (x - b)).reshape(-1).norm().item()
                mb = _maha_one(it, b); mx = _maha_one(it, x)
                fxs.append(fx); fbs.append(fb); dds.append(fx - fb); shs.append(sh)
                mms.append(mb); p2.append(1.0 if mb <= mx else 0.0)
            n_ = len(fbs)
            print(f"{nm:<20}{sum(fxs)/n_:>9.4f}{sum(fbs)/n_:>9.4f}{sum(dds)/n_:>10.4f}"
                  f"{sum(shs)/n_:>11.4f}{sum(mms)/n_:>13.4f}{sum(p2)/n_*100:>6.0f}%")
        print(f"[i] |x-mu|_S-1 = {maha_x_mean:.4f}  (Mahalanobis cua chinh input)")
        print("[i] Neu IG-zero P2 ok% CAO => zero KHONG vi pham (P2) o embedding space")
        print("[i] => Table 1 hang 'zero ✗' SAI cho NLP, va viec zero THANG khong con la nghich ly.")

        rules, valid_m = tau_diag.selection_rules(curve, eps=args.diag_eps)
        tau_diag.print_rules_table(rules, valid=valid_m)
        tau_diag.dump_curve_csv(curve, f"e1_nlp_{args.dataset}_taucurve.csv")

    # ---- accumulators ----
    metric_keys = ["soft_nc", "soft_ns", "soft_logodds"]
    acc = {m: {k: [] for k in metric_keys} for m in methods}
    per_rows = []
    bl_strength = {}          # {method: {"pf":[], "pb":[], "ratio":[], "shift":[]}}  f(x) vs f(baseline)

    special_ids = set(tokenizer.all_special_ids)

    # --- doi trong: CF reference embedding cho IG2 (1 cau lop khac, dung chung) ---
    n_class_nlp = model.config.num_labels
    ig2_ref_embed = None
    if args.rivals:
        # tim 1 cau trong sample co pred class khac cau dau -> lam CF reference
        with torch.no_grad():
            base_cls = None
            for text, _l in sample:
                enc = tokenizer(text, truncation=True, max_length=args.max_len, return_tensors="pt").to(device)
                we, pe, te = get_embeddings(model, args.model, enc["input_ids"], enc["attention_mask"])
                pc = int(fwd(model, we, attention_mask=enc["attention_mask"],
                             position_embed=pe, type_embed=te).argmax(-1).item())
                if base_cls is None:
                    base_cls = pc
                elif pc != base_cls:
                    ig2_ref_embed = we.detach()             # (1,r,d)
                    break
        if ig2_ref_embed is None:                           # fallback: cau dau
            enc = tokenizer(sample[0][0], truncation=True, max_length=args.max_len, return_tensors="pt").to(device)
            we, _, _ = get_embeddings(model, args.model, enc["input_ids"], enc["attention_mask"])
            ig2_ref_embed = we.detach()
        print(f"[i] RIVALS bat: IG2/MaxEnt/FRInGe (ig2_steps={args.ig2_steps}, me_steps={args.me_steps}), "
              f"CF ref seq={ig2_ref_embed.shape[1]}\n")

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
            if nm == "IG2":
                ref_e = prep_ref_embed(ig2_ref_embed, word_embed.shape[1])
                _, attr_full = ig2_nlp(fwd, model, word_embed, ref_e, pred_class,
                                       attn, position_embed, type_embed, steps=args.ig2_steps)
                with torch.no_grad():
                    ref_e2 = prep_ref_embed(ig2_ref_embed, word_embed.shape[1])
                    pf = F.softmax(logits, -1)[0, pred_class].item()
                    # "baseline" cua IG2 = GradCF; xap xi strength bang ref (lop khac)
                    pb = F.softmax(fwd(model, ref_e2, attention_mask=attn,
                                       position_embed=position_embed, type_embed=type_embed), -1)[0, pred_class].item()
                    shift = (ref_e2 - word_embed)[0][keep_tok].norm().item() if keep_tok.any() else 0.0   # L2, khong phai L1-mean
                d = bl_strength.setdefault(nm, {"pf": [], "pb": [], "ratio": [], "shift": [], "df": []})
                d["pf"].append(pf); d["pb"].append(pb)
                d["ratio"].append(pb/pf if pf > 1e-9 else float("nan")); d["shift"].append(shift)
                d["df"].append(pf - pb)
            elif nm == "FRInGe":
                _, attr_full = fringe_nlp(fwd, model, word_embed, pred_class, n_class_nlp,
                                          attn, position_embed, type_embed,
                                          steps=args.steps, me_steps=args.me_steps)
                with torch.no_grad():
                    b0 = max_entropy_embed(fwd, model, word_embed, n_class_nlp, attn,
                                           position_embed, type_embed, steps=args.me_steps)
                    pf = F.softmax(logits, -1)[0, pred_class].item()
                    pb = F.softmax(fwd(model, b0, attention_mask=attn, position_embed=position_embed,
                                       type_embed=type_embed), -1)[0, pred_class].item()
                    shift = (b0 - word_embed)[0][keep_tok].norm().item() if keep_tok.any() else 0.0   # L2, khong phai L1-mean
                d = bl_strength.setdefault(nm, {"pf": [], "pb": [], "ratio": [], "shift": [], "df": []})
                d["pf"].append(pf); d["pb"].append(pb)
                d["ratio"].append(pb/pf if pf > 1e-9 else float("nan")); d["shift"].append(shift)
                d["df"].append(pf - pb)
            elif nm == "IG-MaxEnt":
                be = max_entropy_embed(fwd, model, word_embed, n_class_nlp, attn,
                                       position_embed, type_embed, steps=args.me_steps)
                _, attr_full = ig_embedding(fwd, model, word_embed, be, position_embed, type_embed,
                                            attn, pred_class, args.steps)
                with torch.no_grad():
                    pf = F.softmax(logits, -1)[0, pred_class].item()
                    pb = F.softmax(fwd(model, be, attention_mask=attn, position_embed=position_embed,
                                       type_embed=type_embed), -1)[0, pred_class].item()
                    shift = (be - word_embed)[0][keep_tok].norm().item() if keep_tok.any() else 0.0   # L2, khong phai L1-mean
                d = bl_strength.setdefault(nm, {"pf": [], "pb": [], "ratio": [], "shift": [], "df": []})
                d["pf"].append(pf); d["pb"].append(pb)
                d["ratio"].append(pb/pf if pf > 1e-9 else float("nan")); d["shift"].append(shift)
                d["df"].append(pf - pb)
            elif nm.startswith("EG-"):
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
                    # |b-x|_2 chi tren token thuong (bo special) neu co the
                    # SUA: truoc day dung .abs().mean() = L1/D, KHONG phai quang duong
                    # Euclid. Meo mo IG ~ O(L*||b-x||_2^2), nen phai la L2 norm.
                    diff = (be - word_embed)
                    if not args.include_special and keep_tok.any():
                        shift = diff[0][keep_tok].norm().item()
                    else:
                        shift = diff.reshape(-1).norm().item()
                d = bl_strength.setdefault(nm, {"pf": [], "pb": [], "ratio": [], "shift": [], "df": []})
                d["pf"].append(pf); d["pb"].append(pb)
                d["ratio"].append(pb / pf if pf > 1e-9 else float("nan")); d["shift"].append(shift)
                d["df"].append(pf - pb)

            # sanitize: rival co the tao attr NaN/Inf/scale lon -> soft metric vo Bernoulli p ngoai [0,1]
            attr_full = torch.nan_to_num(attr_full, nan=0.0, posinf=0.0, neginf=0.0)

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
    # ---- TI GIA BIEN d(Δf)/d|b-x|, giua cac tau LIEN TIEP tren truc sweep ----
    # rate[j] = ti gia cua KHOANG [tau_j, tau_{j+1}], gan cho DIEM DAU khoang.
    # Hang cuoi grid va hang co dinh (zero/pad/mask/EG/IG2/...) khong nam tren truc
    # -> khong co ti gia -> in "-". Van doc duoc tren HAI cot Δf va |b-x|₂.
    rate, tau_rate_by_eps, rmax_r = {}, {}, None
    shr_axis = [f"Shrinkage-IG@{t:g}" for t in sorted(args.tau_sweep)]
    shr_axis = [m for m in shr_axis if m in bl_strength and bl_strength[m].get("df")]
    if len(shr_axis) >= 2:
        pt = {m: (sum(bl_strength[m]["df"]) / len(bl_strength[m]["df"]),
                  sum(bl_strength[m]["shift"]) / len(bl_strength[m]["shift"])) for m in shr_axis}
        r_int = []
        for j in range(len(shr_axis) - 1):
            (df_a, s_a), (df_b, s_b) = pt[shr_axis[j]], pt[shr_axis[j + 1]]
            ds = s_b - s_a
            r_int.append((df_b - df_a) / ds if abs(ds) > 1e-12 else float("nan"))
            rate[shr_axis[j]] = r_int[-1]
        good = [r for r in r_int if r == r and r > 0]
        if good:
            rmax_r = max(good)
            for eps in (0.5, 0.2, 0.1, 0.05, 0.01):
                js = [j for j, r in enumerate(r_int) if r == r and r >= eps * rmax_r]
                tau_rate_by_eps[eps] = shr_axis[max(js) + 1] if js else shr_axis[0]

    print(f"\n{'='*116}\nKET QUA E1-NLP tren {n} cau  ({args.model}/{args.dataset})")
    print(f"{'method':<20}{'Soft-NC↑':>15}{'Soft-NS↑':>15}{'Soft-gap↑':>15}{'Soft-logodds↑':>18}"
          f"{'f(x)':>8}{'f(b)':>8}{'Δf':>9}{'|b-x|₂':>10}{'d(Δf)/d|b-x|':>15}")
    print("-" * 133)
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
        # BON cot, giong het ca ba modality: f(x) f(b) Δf |b-x|₂
        # (bo 'ratio': no bo mat f(x); Δf da co f(x) ben trong)
        fx, fxt, dftxt, stxt = "   -  ", "   -  ", "    -   ", "     -    "
        if m in bl_strength:
            d = bl_strength[m]
            if d["pf"]:   fx  = f"{sum(d['pf'])/len(d['pf']):>6.4f}"
            if d["pb"]:   fxt = f"{sum(d['pb'])/len(d['pb']):>6.4f}"
            if d["df"]:   dftxt = f"{sum(d['df'])/len(d['df']):>8.4f}"
            if d["shift"]:stxt = f"{sum(d['shift'])/len(d['shift']):>9.4f}"
        r = rate.get(m)
        rtxt = f"{r:>15.4g}" if (r is not None and r == r) else f"{'-':>15}"
        mark = "  <-- best" if m == best_m else ""
        print(f"{m:<20}{nc_m:>8.4f}±{nc_se:<5.4f}{ns_m:>8.4f}±{ns_se:<5.4f}"
              f"{gp_m:>8.4f}±{gp_se:<5.4f}{lo_m:>10.4f}±{lo_se:<6.4f}"
              f"{fx:>8}{fxt:>8}{dftxt:>9}{stxt:>10}{rtxt}{mark}")
        summary_rows.append({"method": m, "n": n,
                             "soft_nc_mean": nc_m, "soft_nc_se": nc_se,
                             "soft_ns_mean": ns_m, "soft_ns_se": ns_se,
                             "soft_gap_mean": gp_m, "soft_gap_se": gp_se,
                             "soft_logodds_mean": lo_m, "soft_logodds_se": lo_se})
    print("-" * 133)
    print(f"[i] dan dau Soft-gap (=NC+NS-1): {best_m} = {best_gap:.4f}")
    print("[i] Δf = f(x)-f(b) = ngan sach Completeness. |b-x|₂ = quang duong (L2).")
    print("[i] d(Δf)/d|b-x| = ti gia bien, gan cho DIEM DAU cua khoang [tau_j, tau_j+1].")
    if tau_rate_by_eps:
        print(f"\n[i] tau_rate theo eps (KHONG cham Soft-gap, chi forward pass). ti gia max = {rmax_r:.5g}")
        print(f"{'eps':>8}{'nguong':>14}{'tau_rate':>22}{'   == best?':>12}")
        print("-" * 58)
        for eps, m in tau_rate_by_eps.items():
            ok = "KHOP" if m == best_m else "LECH"
            print(f"{eps:>8g}{eps*rmax_r:>14.5g}{m:>22}{ok:>12}")
        print("-" * 58)
        print("[i] IN CA DAI eps. KHONG tu chon mot eps roi bao la rule — chon eps sau khi")
        print("[i]   nhin dap an chinh la cai ma draft dang chi trich BEE.")

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