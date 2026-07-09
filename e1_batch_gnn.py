"""
e1_batch_gnn.py — DOC LAP (PyG). Dua nguyen ly baseline "information-removing
shrinkage" sang GRAPH CLASSIFICATION (MUTAG, PyTorch Geometric).
Khong import synthetic_e0/pea.

IG cho GNN chay tren NODE FEATURES x (R^{N x F}), giu edge_index CO DINH.
Node feature lien tuc -> objective baseline ap thang (portability: chi can mean+cov).

Hai reference shrinkage:
  (S1) Shrink-node  : PCA covariance tren TOAN BO node feature train.
        b_tau(x) = mu + V diag(s/(s+tau)) V^T (x-mu).  Ban "tabular".
  (S2) Shrink-graph : heat-kernel tren graph-Laplacian CUA CHINH do thi
        (graph Fourier): x_smooth = sum_k e^{-sigma*lam_k}<x,phi_k>phi_k.
        Ban "Fourier/blur" — THUC SU dung topo. (Blur-IG-analog cho graph.)

Baseline chuan linh vuc (khong phai zero/mean):
  - GNNExplainer   (torch_geometric.explain, node+edge mask)
  - PGExplainer    (torch_geometric.explain, edge mask; can train truoc)
  - IG-zero        (IG node-feature, baseline=0 — doi chieu, thuong OOD)
  - SubgraphX      (khong co trong PyG core; BO QUA co canh bao, KHONG gia lap)

Metric — CHUAN GNN-XAI:
  Fidelity+ (fid+): bo node QUAN TRONG -> prob tut  = tot (CAO tot).
  Fidelity- (fid-): bo node KHONG quan trong -> prob giu = tot (THAP tot).
  fid_gap = fid+ - fid-.
  "Bo node" = MARGINAL SAMPLING (boc node-feature that tu train pool), KHONG zero-out
  (zero = node ma, OOD -> fidelity vo nghia; dung bai hoc tu tabular deletion=1.0).

torch GPU mac dinh (--device). Ban tu chay. KHONG smoketest.

Chay:
    python e1_batch_gnn.py --limit 60 --tau_sweep 0.1 1 10 100 --sigma_sweep 0.5 1 2 4
    python e1_batch_gnn.py --skip_pg --topk_frac 0.2
"""

from __future__ import annotations
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import to_dense_adj


# ===========================================================================
# (0) Model — GCN graph classifier. forward(x, edge_index, batch) chuan PyG.
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
        hg = global_mean_pool(h, batch)
        return self.cls(hg)                       # (B, n_classes)


# ===========================================================================
# (1) Shrinkage references
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
def shrinkage_node(feat, ref, tau):
    mu, V, s = ref
    d = feat - mu[None]
    coeff = d @ V
    g = s / (s + tau)
    return mu[None] + (coeff * g[None]) @ V.T


@torch.no_grad()
def shrinkage_graph(edge_index, feat, sigma, num_nodes):
    """Heat-kernel smoothing tren graph-Laplacian cua chinh do thi. MUTAG nho -> eigh dense."""
    N = num_nodes
    A = to_dense_adj(edge_index, max_num_nodes=N)[0].to(feat.device)
    A = ((A + A.T) > 0).float()
    A.fill_diagonal_(0.0)
    deg = A.sum(1)
    dinv = deg.clamp_min(1e-12).rsqrt()
    L = torch.eye(N, device=feat.device) - (dinv[:, None] * A * dinv[None, :])
    L = 0.5 * (L + L.T)
    lam, Phi = torch.linalg.eigh(L)
    c = torch.exp(-sigma * lam)
    coeff = Phi.T @ feat
    return Phi @ (c[:, None] * coeff)


# ===========================================================================
# (2) IG tren node feature. importance/node = sum_F |phi|.
# ===========================================================================
def ig_node(model, x, edge_index, baseline, target, T=50):
    device = x.device
    alphas = (torch.arange(T, device=device) + 0.5) / T
    grads = torch.zeros_like(x)
    for a in alphas:
        X = (baseline + a * (x - baseline)).clone().requires_grad_(True)
        logit = model(X, edge_index)[0, target]
        g, = torch.autograd.grad(logit, X)
        grads += g.detach()
    grads /= T
    phi = grads * (x - baseline)
    return phi.abs().sum(1)


# ===========================================================================
# (3) Fidelity+/- voi REMOVE = MARGINAL SAMPLING
# ===========================================================================
@torch.no_grad()
def _marginal_fill(x, remove_mask, node_pool, gen):
    out = x.clone()
    idx = torch.where(remove_mask)[0]
    if idx.numel():
        rows = torch.randint(0, node_pool.shape[0], (idx.numel(),), generator=gen)
        out[idx] = node_pool[rows.to(x.device)]
    return out


@torch.no_grad()
def fidelity(model, x, edge_index, node_importance, target, node_pool, gen, topk_frac=0.15):
    N = x.shape[0]
    k = max(1, int(round(topk_frac * N)))
    order = torch.argsort(node_importance, descending=True)
    p_full = F.softmax(model(x, edge_index), 1)[0, target].item()

    rm_imp = torch.zeros(N, dtype=torch.bool, device=x.device); rm_imp[order[:k]] = True
    p_imp = F.softmax(model(_marginal_fill(x, rm_imp, node_pool, gen), edge_index), 1)[0, target].item()

    rm_un = torch.zeros(N, dtype=torch.bool, device=x.device); rm_un[order[k:]] = True
    p_un = F.softmax(model(_marginal_fill(x, rm_un, node_pool, gen), edge_index), 1)[0, target].item()

    return {"fid+": p_full - p_imp, "fid-": p_full - p_un,
            "fid_gap": (p_full - p_imp) - (p_full - p_un)}


# ===========================================================================
# (4) Baseline linh vuc -> node importance
# ===========================================================================
def make_explainer_gnn(model, epochs):
    from torch_geometric.explain import Explainer, GNNExplainer
    return Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(mode="multiclass_classification",
                          task_level="graph", return_type="raw"),
    )


def imp_from_explanation(expl, N, device):
    if getattr(expl, "node_mask", None) is not None:
        nm = expl.node_mask
        return nm.abs().sum(1).to(device) if nm.dim() == 2 else nm.abs().to(device)
    em = expl.edge_mask
    ei = expl.edge_index
    imp = torch.zeros(N, device=device)
    imp.index_add_(0, ei[1].to(device), em.to(device))
    imp.index_add_(0, ei[0].to(device), em.to(device))
    return imp


# ===========================================================================
# (5) Main
# ===========================================================================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N_ig", type=int, default=50)
    ap.add_argument("--tau_sweep", type=float, nargs="+", default=[0.1, 1.0, 10.0, 100.0])
    ap.add_argument("--sigma_sweep", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--topk_frac", type=float, default=0.15)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--gnnex_epochs", type=int, default=100)
    ap.add_argument("--pg_epochs", type=int, default=30)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--root", type=str, default="/tmp/TUDataset")
    ap.add_argument("--skip_pg", action="store_true")
    ap.add_argument("--skip_gnnex", action="store_true")
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
    n = len(dataset)
    in_feats = dataset.num_node_features
    n_class = dataset.num_classes
    n_tr = int(0.8 * n)
    tr = dataset[:n_tr]; te = dataset[n_tr:]
    print(f"[i] MUTAG: {n} graphs, in_feats={in_feats}, n_class={n_class}, "
          f"train={len(tr)} test={len(te)}")

    model = GCNGraph(in_feats, args.hidden, n_class).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    loader = DataLoader(tr, batch_size=32, shuffle=True)
    model.train()
    for ep in range(args.epochs):
        for b in loader:
            b = b.to(device)
            out = model(b.x.float(), b.edge_index, b.batch)
            loss = F.cross_entropy(out, b.y)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()

    @torch.no_grad()
    def acc(dset):
        c = t = 0
        for b in DataLoader(dset, batch_size=64):
            b = b.to(device)
            pred = model(b.x.float(), b.edge_index, b.batch).argmax(1)
            c += int((pred == b.y).sum()); t += b.y.numel()
        return c / t
    print(f"[i] train acc={acc(tr):.4f}  test acc={acc(te):.4f}")

    all_train_feats = torch.cat([g.x.float() for g in tr], 0).to(device)
    node_ref = fit_node_reference(all_train_feats)
    node_pool = all_train_feats
    print(f"[i] node pool (marginal) = {node_pool.shape[0]} vectors\n")

    # ---- PGExplainer (train truoc) ----
    pg_expl = None
    if not args.skip_pg:
        try:
            from torch_geometric.explain import Explainer, PGExplainer
            pg_expl = Explainer(
                model=model,
                algorithm=PGExplainer(epochs=args.pg_epochs, lr=0.003),
                explanation_type="phenomenon",
                edge_mask_type="object",
                model_config=dict(mode="multiclass_classification",
                                  task_level="graph", return_type="raw"),
            )
            for ep in range(args.pg_epochs):
                for g in tr:
                    g = g.to(device)
                    with torch.no_grad():
                        tgt = model(g.x.float(), g.edge_index).argmax(1)
                    pg_expl.algorithm.train(ep, model, g.x.float(), g.edge_index, target=tgt)
            print("[i] PGExplainer trained.")
        except Exception as e:
            print(f"[!] PGExplainer bo qua (loi/API khac): {e}")
            pg_expl = None

    gnn_expl = None
    if not args.skip_gnnex:
        try:
            gnn_expl = make_explainer_gnn(model, args.gnnex_epochs)
        except Exception as e:
            print(f"[!] GNNExplainer bo qua: {e}")

    print("[!] SubgraphX: khong co trong PyG core. BO QUA — khong gia lap.\n")

    te_eval = te[:args.limit]
    gen = torch.Generator(device="cpu"); gen.manual_seed(args.seed + 7)

    methods = ["IG-zero"]
    methods += [f"Shrink-node@{t:g}" for t in args.tau_sweep]
    methods += [f"Shrink-graph@{s:g}" for s in args.sigma_sweep]
    if gnn_expl is not None: methods.append("GNNExplainer")
    if pg_expl is not None:  methods.append("PGExplainer")

    def node_importance(nm, x, edge_index, target, N):
        if nm == "IG-zero":
            return ig_node(model, x, edge_index, torch.zeros_like(x), target, T=args.N_ig)
        if nm.startswith("Shrink-node@"):
            tau = float(nm.split("@")[1])
            return ig_node(model, x, edge_index, shrinkage_node(x, node_ref, tau), target, T=args.N_ig)
        if nm.startswith("Shrink-graph@"):
            sig = float(nm.split("@")[1])
            return ig_node(model, x, edge_index, shrinkage_graph(edge_index, x, sig, N), target, T=args.N_ig)
        if nm == "GNNExplainer":
            e = gnn_expl(x, edge_index, target=torch.tensor([target], device=x.device))
            return imp_from_explanation(e, N, x.device)
        if nm == "PGExplainer":
            e = pg_expl(x, edge_index, target=torch.tensor([target], device=x.device))
            return imp_from_explanation(e, N, x.device)
        raise ValueError(nm)

    print(f"{'method':<20}{'fid+↑':>10}{'fid-↓':>10}{'fid_gap↑':>12}")
    print("-" * 52)
    results = {}
    for nm in methods:
        fp, fm, fg = [], [], []
        for g in te_eval:
            g = g.to(device)
            x = g.x.float(); ei = g.edge_index; N = x.shape[0]
            with torch.no_grad():
                target = model(x, ei).argmax(1).item()
            try:
                imp = node_importance(nm, x, ei, target, N)
            except Exception:
                continue
            r = fidelity(model, x, ei, imp, target, node_pool, gen, topk_frac=args.topk_frac)
            fp.append(r["fid+"]); fm.append(r["fid-"]); fg.append(r["fid_gap"])
        if not fp:
            print(f"{nm:<20}{'-':>10}{'-':>10}{'(bo)':>12}"); continue
        mp, mm, mg = sum(fp)/len(fp), sum(fm)/len(fm), sum(fg)/len(fg)
        results[nm] = mg
        print(f"{nm:<20}{mp:>10.4f}{mm:>10.4f}{mg:>12.4f}")
    print("-" * 52)
    if results:
        best = max(results, key=results.get)
        print(f"[i] best fid_gap: {best} = {results[best]:.4f}")


if __name__ == "__main__":
    main()