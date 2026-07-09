"""
e1_gnnexplainer_ref.py — DOC LAP (PyG). MUTAG graph classification.

TRONG TAM (mot cau): mask cua GNNExplainer luon co dang
      x_masked = m (*) x + (1 - m) (*) r
  m = mask HOC duoc; r = REFERENCE ("phan bi bo thay bang gi").
  PyG mac dinh r = 0 (h = x * m.sigmoid()). Zero la baseline te/OOD.
  Ta THAY r = b_tau(x) (shrinkage) — chi thay REFERENCE, thuat toan GNNExplainer
  giu nguyen. So sanh: GNNExplainer(ref=zero) vs GNNExplainer(ref=shrinkage@tau),
  QUET vai muc tau. Cung 1 permutation metric.

  b_tau(x) = mu + V diag(s/(s+tau)) V^T (x - mu)   (PCA covariance node-feature train)
  tau -> 0 : r -> x (khong bo gi).  tau -> inf : r -> mu (bo het ve trung binh).

Metric — permutation fidelity (chuan GNN-XAI, dang permutation, KHONG zero-out):
  PermFid+ : permute node QUAN TRONG (top-k theo mask) -> E[prob tut].  CAO = tot.
  PermFid- : permute node KHONG quan trong               -> E[prob giu].  THAP = tot.
  perm_gap = PermFid+ - PermFid-.
  "permute node" = BOC feature node that tu POOL train (across-graph), K lan lay ky vong
  -> pha thong tin nhung GIU in-distribution (khong tao node OOD).

torch GPU mac dinh (--device). Ban tu chay. KHONG smoketest.

Chay:
    python e1_gnnexplainer_ref.py --tau_sweep 0.1 1 10 100 --topk_frac 0.25 --K_perm 16
    python e1_gnnexplainer_ref.py --limit 80 --gnnex_epochs 150
"""

from __future__ import annotations
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool


# ===========================================================================
# (0) Model
# ===========================================================================
class GCNGraph(nn.Module):
    def __init__(self, in_feats, hidden, n_classes):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.cls = nn.Linear(hidden, n_classes)

    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = x.new_zeros(x.shape[0], dtype=torch.long)
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        return self.cls(global_mean_pool(h, batch))


# ===========================================================================
# (1) Shrinkage reference b_tau(x)  (PCA covariance node-feature train)
# ===========================================================================
@torch.no_grad()
def fit_node_reference(all_feats, floor=1e-6):
    mu = all_feats.mean(0)
    Xc = all_feats - mu[None]
    S = (Xc.T @ Xc) / max(1, all_feats.shape[0] - 1)
    S = 0.5 * (S + S.T) + floor * torch.eye(all_feats.shape[1], device=all_feats.device)
    s, V = torch.linalg.eigh(S)
    order = torch.argsort(s, descending=True)
    return mu, V[:, order].contiguous(), s[order].contiguous()


@torch.no_grad()
def shrinkage_baseline(feat, ref, tau):
    """b_tau node-wise: (N,F)->(N,F)."""
    mu, V, s = ref
    coeff = (feat - mu[None]) @ V
    g = s / (s + tau)
    return mu[None] + (coeff * g[None]) @ V.T


# ===========================================================================
# (2) GNNExplainer voi REFERENCE CAM DUOC.
#   x_masked = m (*) x + (1-m) (*) baseline.  baseline = 0 (default) hoac b_tau(x).
#   Hoc node mask (per-node scalar) bang GD, loss = -logit_target + L1 + entropy
#   (dung tinh than GNNExplainer). Tra ve node importance (N,) in [0,1].
# ===========================================================================
def gnnexplainer(model, x, edge_index, target, baseline,
                 epochs=150, lr=0.01, l1=0.005, ent=1.0):
    N = x.shape[0]
    m = torch.zeros(N, device=x.device, requires_grad=True)   # logit mask
    opt = torch.optim.Adam([m], lr=lr)
    for _ in range(epochs):
        mask = m.sigmoid()[:, None]                           # (N,1)
        x_masked = mask * x + (1.0 - mask) * baseline         # <- REFERENCE o day
        logit = model(x_masked, edge_index)[0, target]
        s = m.sigmoid()
        entropy = -(s * (s + 1e-9).log() + (1 - s) * (1 - s + 1e-9).log()).mean()
        loss = -logit + l1 * s.mean() + ent * entropy
        opt.zero_grad(); loss.backward(); opt.step()
    return m.sigmoid().detach()                               # (N,)


# ===========================================================================
# (3) Permutation fidelity
# ===========================================================================
@torch.no_grad()
def _permute_fill(x, sel, node_pool, gen):
    out = x.clone()
    idx = torch.where(sel)[0]
    if idx.numel():
        rows = torch.randint(0, node_pool.shape[0], (idx.numel(),), generator=gen)
        out[idx] = node_pool[rows.to(x.device)]
    return out


@torch.no_grad()
def perm_fidelity(model, x, edge_index, node_mask, target, node_pool, gen,
                  topk_frac=0.25, K=16):
    N = x.shape[0]
    k = max(1, int(round(topk_frac * N)))
    order = torch.argsort(node_mask, descending=True)
    p_full = F.softmax(model(x, edge_index), 1)[0, target].item()
    top = torch.zeros(N, dtype=torch.bool, device=x.device); top[order[:k]] = True
    rest = ~top
    dp_p = dp_m = 0.0
    for _ in range(K):
        dp_p += p_full - F.softmax(model(_permute_fill(x, top, node_pool, gen), edge_index), 1)[0, target].item()
        dp_m += p_full - F.softmax(model(_permute_fill(x, rest, node_pool, gen), edge_index), 1)[0, target].item()
    dp_p /= K; dp_m /= K
    return {"permfid+": dp_p, "permfid-": dp_m, "perm_gap": dp_p - dp_m}


# ===========================================================================
# (4) Main
# ===========================================================================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau_sweep", type=float, nargs="+", default=[0.1, 1.0, 10.0, 100.0])
    ap.add_argument("--topk_frac", type=float, default=0.25)
    ap.add_argument("--K_perm", type=int, default=16)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=120, help="epoch train GCN")
    ap.add_argument("--gnnex_epochs", type=int, default=150)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--root", type=str, default="/tmp/TUDataset")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[!] cuda khong san sang -> cpu"); device = "cpu"
    torch.manual_seed(args.seed)

    dataset = TUDataset(root=args.root, name="MUTAG").shuffle()
    n = len(dataset); in_feats = dataset.num_node_features; n_class = dataset.num_classes
    n_tr = int(0.8 * n); tr = dataset[:n_tr]; te = dataset[n_tr:]
    print(f"[i] MUTAG: {n} graphs, in_feats={in_feats}, n_class={n_class}, train={len(tr)} test={len(te)}")

    # train GCN
    model = GCNGraph(in_feats, args.hidden, n_class).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    loader = DataLoader(tr, batch_size=32, shuffle=True)
    model.train()
    for ep in range(args.epochs):
        for b in loader:
            b = b.to(device)
            loss = F.cross_entropy(model(b.x.float(), b.edge_index, b.batch), b.y)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()

    @torch.no_grad()
    def acc(dset):
        c = t = 0
        for b in DataLoader(dset, batch_size=64):
            b = b.to(device)
            c += int((model(b.x.float(), b.edge_index, b.batch).argmax(1) == b.y).sum()); t += b.y.numel()
        return c / t
    print(f"[i] train acc={acc(tr):.4f}  test acc={acc(te):.4f}")

    all_train_feats = torch.cat([g.x.float() for g in tr], 0).to(device)
    node_ref = fit_node_reference(all_train_feats)
    node_pool = all_train_feats
    print(f"[i] permute pool = {node_pool.shape[0]} node-feature vectors\n")

    te_eval = te[:args.limit]
    gen = torch.Generator(device="cpu"); gen.manual_seed(args.seed + 7)

    # cac REFERENCE cua GNNExplainer: zero (default) + shrinkage@tau (sweep)
    refs = ["zero"] + [f"shrink@{t:g}" for t in args.tau_sweep]

    def baseline_for(name, x):
        if name == "zero":
            return torch.zeros_like(x)
        tau = float(name.split("@")[1])
        return shrinkage_baseline(x, node_ref, tau)

    print(f"{'GNNExplainer ref':<22}{'PermFid+↑':>11}{'PermFid-↓':>11}{'perm_gap↑':>12}")
    print("-" * 56)
    results = {}
    for rf in refs:
        pp, pm, pg = [], [], []
        for g in te_eval:
            g = g.to(device)
            x = g.x.float(); ei = g.edge_index
            with torch.no_grad():
                target = model(x, ei).argmax(1).item()
            base = baseline_for(rf, x)
            mask = gnnexplainer(model, x, ei, target, base, epochs=args.gnnex_epochs)
            r = perm_fidelity(model, x, ei, mask, target, node_pool, gen,
                              topk_frac=args.topk_frac, K=args.K_perm)
            pp.append(r["permfid+"]); pm.append(r["permfid-"]); pg.append(r["perm_gap"])
        mp, mm, mg = sum(pp)/len(pp), sum(pm)/len(pm), sum(pg)/len(pg)
        results[rf] = mg
        tag = "GNNExplainer(default)" if rf == "zero" else f"GNNExplainer({rf})"
        print(f"{tag:<22}{mp:>11.4f}{mm:>11.4f}{mg:>12.4f}")
    print("-" * 56)
    best = max(results, key=results.get)
    print(f"[i] best perm_gap: {best} = {results[best]:.4f}")
    if "zero" in results:
        shr = {k: v for k, v in results.items() if k != "zero"}
        if shr:
            bs = max(shr, key=shr.get)
            print(f"[i] default(zero)={results['zero']:.4f}  |  shrink tot nhat ({bs})={shr[bs]:.4f}  "
                  f"|  chenh = {shr[bs]-results['zero']:+.4f}")


if __name__ == "__main__":
    main()