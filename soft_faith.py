"""
soft_faith.py
=============
Soft Normalized Sufficiency (Soft-NS) and Soft Normalized Comprehensiveness (Soft-NC)
metrics from:

  Zhao & Aletras, "Incorporating Attribution Importance for Improving
  Faithfulness Metrics", ACL 2023.
  https://github.com/casszhao/SoftFaith

nn_forward_func signature (bert_helper / distilbert_helper / roberta_helper):
    nn_forward_func(model, input_embed, attention_mask=None,
                    position_embed=None, type_embed=None, ...)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _normalize_scores(attr: torch.Tensor) -> torch.Tensor:
    """Map attribution scores to [0, 1] via min-max normalisation."""
    a_min = attr.min()
    a_max = attr.max()
    if (a_max - a_min).abs() < 1e-9:
        return torch.full_like(attr, 0.5)
    return (attr - a_min) / (a_max - a_min)


def _call_forward(nn_forward_func, model, input_embed, position_embed, type_embed, attention_mask):
    return nn_forward_func(
        model, input_embed,
        attention_mask=attention_mask,
        position_embed=position_embed,
        type_embed=type_embed,
    )


def _get_predicted_class_and_prob(nn_forward_func, model, input_embed,
                                  position_embed, type_embed, attention_mask):
    with torch.no_grad():
        logits = _call_forward(nn_forward_func, model, input_embed, position_embed, type_embed, attention_mask)
        probs = F.softmax(logits, dim=-1)
        pred_class = int(probs.argmax(dim=-1).item())
        p_full = float(probs[0, pred_class].item())
    return pred_class, p_full


def _get_prob_from_embed(nn_forward_func, model, perturbed_embed,
                         position_embed, type_embed, attention_mask, pred_class):
    with torch.no_grad():
        logits = _call_forward(nn_forward_func, model, perturbed_embed, position_embed, type_embed, attention_mask)
        probs = F.softmax(logits, dim=-1)
        return float(probs[0, pred_class].item())


def _baseline_prob(nn_forward_func, model, input_embed, position_embed,
                   type_embed, attention_mask, base_token_emb, pred_class):
    """p(pred_class) on a zeroed / baseline sequence — S(X, ŷ, 0)."""
    seq_len = input_embed.shape[1]
    if base_token_emb is not None:
        zero_embed = base_token_emb.unsqueeze(0).expand(1, seq_len, -1).to(input_embed.device)
    else:
        zero_embed = torch.zeros_like(input_embed)
    return _get_prob_from_embed(nn_forward_func, model, zero_embed,
                                position_embed, type_embed, attention_mask, pred_class)


# ---------------------------------------------------------------------------
# Soft perturbation (Eq. 3)
# ---------------------------------------------------------------------------
def soft_input_perturbation(token_embeddings, attr_scores, mode="sufficiency"):
    """Per-token Bernoulli dropout of embeddings. q=a (suff) / 1-a (comp)."""
    assert mode in ("sufficiency", "comprehensiveness")
    scores = _normalize_scores(attr_scores.float())
    q = scores if mode == "sufficiency" else 1.0 - scores
    device = token_embeddings.device
    mask = torch.bernoulli(q.to(device)).unsqueeze(0).unsqueeze(-1)   # (1,seq,1)
    return token_embeddings.detach() * mask


def calculate_soft_sufficiency(nn_forward_func, model, input_embed, position_embed,
                               type_embed, attention_mask, attr_full,
                               base_token_emb=None, n_samples=10):
    """Soft-NS (Eq. 4). Soft-NS = (Soft-S - S0) / (1 - S0)."""
    pred_class, p_full = _get_predicted_class_and_prob(
        nn_forward_func, model, input_embed, position_embed, type_embed, attention_mask)
    p_base = _baseline_prob(nn_forward_func, model, input_embed, position_embed,
                            type_embed, attention_mask, base_token_emb, pred_class)
    s_base = 1.0 - max(0.0, p_full - p_base)
    denom = 1.0 - s_base
    if abs(denom) < 1e-9:
        return 0.0
    soft_s_vals = []
    for _ in range(n_samples):
        x_prime = soft_input_perturbation(input_embed, attr_full, mode="sufficiency")
        p_prime = _get_prob_from_embed(nn_forward_func, model, x_prime,
                                       position_embed, type_embed, attention_mask, pred_class)
        soft_s_vals.append(1.0 - max(0.0, p_full - p_prime))
    return (float(np.mean(soft_s_vals)) - s_base) / denom


def calculate_soft_comprehensiveness(nn_forward_func, model, input_embed, position_embed,
                                     type_embed, attention_mask, attr_full,
                                     base_token_emb=None, n_samples=10):
    """Soft-NC (Eq. 5). Soft-NC = Soft-C / (1 - S0)."""
    pred_class, p_full = _get_predicted_class_and_prob(
        nn_forward_func, model, input_embed, position_embed, type_embed, attention_mask)
    p_base = _baseline_prob(nn_forward_func, model, input_embed, position_embed,
                            type_embed, attention_mask, base_token_emb, pred_class)
    s_base = 1.0 - max(0.0, p_full - p_base)
    denom = 1.0 - s_base
    if abs(denom) < 1e-9:
        return 0.0
    soft_c_vals = []
    for _ in range(n_samples):
        x_prime = soft_input_perturbation(input_embed, attr_full, mode="comprehensiveness")
        p_prime = _get_prob_from_embed(nn_forward_func, model, x_prime,
                                       position_embed, type_embed, attention_mask, pred_class)
        soft_c_vals.append(max(0.0, p_full - p_prime))
    return float(np.mean(soft_c_vals)) / denom


def calculate_soft_log_odds(nn_forward_func, model, input_embed, position_embed,
                            type_embed, attention_mask, attr_full,
                            base_token_emb=None, n_samples=10):
    """Soft log-odds: mean drop in log-odds after soft comprehensiveness erasure."""
    pred_class, p_full = _get_predicted_class_and_prob(
        nn_forward_func, model, input_embed, position_embed, type_embed, attention_mask)
    eps = 1e-9
    log_odds_vals = []
    for _ in range(n_samples):
        x_prime = soft_input_perturbation(input_embed, attr_full, mode="comprehensiveness")
        p_prime = _get_prob_from_embed(nn_forward_func, model, x_prime,
                                       position_embed, type_embed, attention_mask, pred_class)
        lo = (np.log((p_full + eps) / (1 - p_full + eps))
              - np.log((p_prime + eps) / (1 - p_prime + eps)))
        log_odds_vals.append(lo)
    return float(np.mean(log_odds_vals))