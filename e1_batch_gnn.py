"""
e1_batch_gnn.py — DOC LAP (PyG). MUTAG graph classification.
Khong import synthetic_e0/pea.

KHUNG THONG NHAT (theo dung y: reference va mask la CUNG MOT VAT o hai dang):
  - Co MOT co che danh gia duy nhat: PERMUTATION test tren node.
  - Co che do can mot MASK (node importance) chi ra "node nao mang thong tin".
  - MASK do co the den tu NHIEU NGUON — ta cam cac nguon khac nhau vao CUNG co che:

      NGUON MASK tu SHRINKAGE (phuong phap cua ta — spectral reference):
        Shrink-node@tau : mask = | residual |_node, residual = x - b_tau(x),
                          b_tau = mu + V diag(s/(s+tau)) V^T (x-mu)  (PCA covariance).
                          -> "luong thong tin bi xoa moi node" lam mask.
        Shrink-graph@sig: mask = | x - heat_kernel_smooth(x) |_node tren graph-Laplacian
                          (graph Fourier). Ban "blur" that su dung topo.

      NGUON MASK MAC DINH cua linh vuc (de SO SANH):
        GNNExplainer    : node/edge mask no HOC.
        PGExplainer     : edge mask (parameterized).
        IG-zero         : |x-0|*grad gom theo node (reference dang suy bien = mask).

  => TAT CA ra cung 1 vector importance/node -> cung 1 PERMUTATION test -> cung metric.
     So sanh tao-voi-tao: moi thu deu la "mask cho permutation", chi khac NGUON sinh mask.

CO CHE PERMUTATION (thay cho remove-to-zero/marginal cung):
  Mask -> chon top-k node (hoac bo top-k). Voi node duoc chon, PERMUTE feature bang
  cach BOC feature node that tu POOL train (across-graph marginal permutation, kieu
  Breiman). Lam K lan, lay ky vong -> pha thong tin CUA node do nhung GIU in-distribution,
  KHONG tao node "ma" OOD (bai hoc tu tabular deletion=1.0).

METRIC — permutation fidelity (chuan GNN-XAI, dang permutation):
  PermFid+ : permute node QUAN TRONG (top-k) -> E[prob tut].  CAO = tot.
  PermFid- : permute node KHONG quan trong    -> E[prob giu].  THAP = tot.
  perm_gap = PermFid+ - PermFid-.

torch GPU mac dinh (--device). Ban tu chay. KHONG smoketest.

Chay:
    python e1_batch_gnn.py --limit 60 --tau_sweep 0.1 1 10 100 --sigma_sweep 0.5 1 2 4 --K_perm 8
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
        hg = global_mean_pool(h, batch)
        return self.cls(hg)


# ===========================================================================
# (1) Shrinkage references -> MASK (residual magnitude theo node)
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
def shrink_node_baseline(feat, ref, tau):
    mu, V, s = ref
    d = feat - mu[None]
    coeff = d @ V
    g = s / (s + tau)
    return mu[None] + (coeff * g[None]) @ V.T          # b_tau(x)


@torch.no_grad()
def shrink_graph_baseline(edge_index, feat, sigma, num_nodes):
    N = num_nodes
    A = to_dense_adj(edge_index, max_num_nodes=N)[0].to(feat.device)
    A = ((A + A.T) > 0).float(); A.fill_diagonal_(0.0)
    deg = A.sum(1); dinv = deg.clamp_min(1e-12).rsqrt()
    L = torch.eye(N, device=feat.device) - (dinv[:, None] * A * dinv[None, :])
    L = 0.5 * (L + L.T)
    lam, Phi = torch.linalg.eigh(L)
    c = torch.exp(-sigma * lam)
    coeff = Phi.T @ feat
    return Phi @ (c[:, None] * coeff)                  # heat-smoothed baseline


@torch.no_grad()
def mask_from_residual(feat, baseline):
    """MASK = luong thong tin bi xoa moi node = |x - baseline| gom theo feature."""
    return (feat - baseline).abs().sum(1)              # (N,)


# ===========================================================================
# (2) IG-zero -> mask (reference dang suy bien). Van la 1 nguon mask.
# ===========================================================================
def mask_ig_zero(model, x, edge_index, target, T=50):
    device = x.device
    alphas = (torch.arange(T, device=device) + 0.5) / T
    grads = torch.zeros_like(x)
    base = torch.zeros_like(x)
    for a in alphas:
        X = (base + a * (x - base)).clone().requires_grad_(True)
        logit = model(X, edge_index)[0, target]
        g, = torch.autograd.grad(logit, X)
        grads += g.detach()
    grads /= T
    return (grads * (x - base)).abs().sum(1)           # (N,)


# ===========================================================================
# (3) CO CHE PERMUTATION duy nhat — nhan MASK bat ky, tra PermFid+/-
# ===========================================================================
@torch.no_grad()
def _permute_fill(x, perm_mask, node_pool, gen):
    """Voi node duoc chon (perm_mask=True), thay feature = feature node BOC tu pool."""
    out = x.clone()
    idx = torch.where(perm_mask)[0]
    if idx.numel():
        rows = torch.randint(0, node_pool.shape[0], (idx.numel(),), generator=gen)
        out[idx] = node_pool[rows.to(x.device)]
    return out


@torch.no_grad()
def perm_fidelity(model, x, edge_index, node_mask, target, node_pool, gen,
                  topk_frac=0.15, K=8):
    """
    node_mask: (N,) importance BAT KY nguon. Permute K lan, lay ky vong.
    PermFid+ : permute TOP-k node quan trong -> E[p_full - p].
    PermFid- : permute (1-topk) node con lai  -> E[p_full - p].
    """
    N = x.shape[0]
    k = max(1, int(round(topk_frac * N)))
    order = torch.argsort(node_mask, descending=True)
    p_full = F.softmax(model(x, edge_index), 1)[0, target].item()

    top = torch.zeros(N, dtype=torch.bool, device=x.device); top[order[:k]] = True
    rest = ~top

    dp_plus, dp_minus = 0.0, 0.0
    for _ in range(K):
        xp = _permute_fill(x, top, node_pool, gen)
        dp_plus += p_full - F.softmax(model(xp, edge_index), 1)[0, target].item()
        xm = _permute_fill(x, rest, node_pool, gen)
        dp_minus += p_full - F.softmax(model(xm, edge_index), 1)[0, target].item()
    dp_plus /= K; dp_minus /= K
    return {"permfid+": dp_plus, "permfid-": dp_minus, "perm_gap": dp_plus - dp_minus}


# ===========================================================================
# (4) Nguon mask MAC DINH cua linh vuc: GNNExplainer / PGExplainer
# ===========================================================================
def make_gnnex(model, epochs):
    from torch_geometric.explain import Explainer, GNNExplainer
    return Explainer(
        model=model, algorithm=GNNExplainer(epochs=epochs),
        explanation_type="model", node_mask_type="attributes", edge_mask_type="object",
        model_config=dict(mode="multiclass_classification", task_level="graph", return_type="raw"),
    )


# ---------------------------------------------------------------------------
# GNNExplainer THU CONG voi REFERENCE CAM DUOC.
#   PyG mac dinh: x_masked = mask (*) x = mask (*) x + (1-mask) (*) 0  -> reference = 0.
#   Ta thay 0 bang `baseline` (vd b_tau(x) shrinkage):
#       x_masked = mask (*) x + (1-mask) (*) baseline
#   Hoc node mask (per-node scalar) bang GD, loss = -logit_target + L1 + entropy
#   (tinh than GNNExplainer). Tra ve node mask (N,).
#   => CUNG thuat toan, chi khac REFERENCE noi bo:
#      "GNNExplainer(zero) vs GNNExplainer(shrinkage)".
# ---------------------------------------------------------------------------
def gnnex_manual(model, x, edge_index, target, baseline, epochs=100,
                 lr=0.01, l1=0.005, ent=1.0):
    N = x.shape[0]
    m = torch.zeros(N, device=x.device, requires_grad=True)   # logit mask
    opt = torch.optim.Adam([m], lr=lr)
    for _ in range(epochs):
        mask = m.sigmoid()[:, None]                           # (N,1)
        x_masked = mask * x + (1.0 - mask) * baseline         # <- reference cam vao day
        logit = model(x_masked, edge_index)[0, target]
        s = m.sigmoid()
        loss = -logit + l1 * s.mean() + ent * (-(s * (s + 1e-9).log()
                                                  + (1 - s) * (1 - s + 1e-9).log())).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return m.sigmoid().detach()                               # (N,)


def mask_from_explanation(expl, N, device):
    if getattr(expl, "node_mask", None) is not None:
        nm = expl.node_mask
        return nm.abs().sum(1).to(device) if nm.dim() == 2 else nm.abs().to(device)
    em = expl.edge_mask; ei = expl.edge_index
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
    ap.add_argument("--K_perm", type=int, default=8, help="so lan permutation lay ky vong")
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
    n = len(dataset); in_feats = dataset.num_node_features; n_class = dataset.num_classes
    n_tr = int(0.8 * n); tr = dataset[:n_tr]; te = dataset[n_tr:]
    print(f"[i] MUTAG: {n} graphs, in_feats={in_feats}, n_class={n_class}, train={len(tr)} test={len(te)}")

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
    node_pool = all_train_feats                          # pool permute (across-graph)
    print(f"[i] permute pool = {node_pool.shape[0]} node-feature vectors\n")

    # PGExplainer (nguon mask mac dinh, can train)
    pg = None
    if not args.skip_pg:
        try:
            from torch_geometric.explain import Explainer, PGExplainer
            pg = Explainer(
                model=model, algorithm=PGExplainer(epochs=args.pg_epochs, lr=0.003),
                explanation_type="phenomenon", edge_mask_type="object",
                model_config=dict(mode="multiclass_classification", task_level="graph", return_type="raw"),
            )
            for ep in range(args.pg_epochs):
                for g in tr:
                    g = g.to(device)
                    with torch.no_grad():
                        tgt = model(g.x.float(), g.edge_index).argmax(1)
                    pg.algorithm.train(ep, model, g.x.float(), g.edge_index, target=tgt)
            print("[i] PGExplainer trained.")
        except Exception as e:
            print(f"[!] PGExplainer bo qua: {e}"); pg = None

    print("[!] SubgraphX: khong co trong PyG core. BO QUA — khong gia lap.")
    print("[i] GNNExplainer chay ban THU CONG (reference cam duoc): zero vs shrinkage.\n")

    te_eval = te[:args.limit]
    gen = torch.Generator(device="cpu"); gen.manual_seed(args.seed + 7)

    # tau dung cho reference shrinkage cam vao GNNExplainer (lay tau giua cua sweep)
    tau_for_gnnex = args.tau_sweep[len(args.tau_sweep) // 2]

    # DANH SACH NGUON MASK
    mask_sources = ["IG-zero"]
    mask_sources += [f"Shrink-node@{t:g}" for t in args.tau_sweep]
    mask_sources += [f"Shrink-graph@{s:g}" for s in args.sigma_sweep]
    # --- THI NGHIEM CHINH: cung GNNExplainer, khac REFERENCE noi bo ---
    mask_sources += ["GNNExplainer(zero)", f"GNNExplainer(shrink@{tau_for_gnnex:g})"]
    if pg is not None:    mask_sources.append("PGExplainer(default)")

    def get_mask(nm, x, edge_index, target, N):
        # --- nguon SHRINKAGE (phuong phap cua ta): residual lam mask ---
        if nm.startswith("Shrink-node@"):
            tau = float(nm.split("@")[1])
            return mask_from_residual(x, shrink_node_baseline(x, node_ref, tau))
        if nm.startswith("Shrink-graph@"):
            sig = float(nm.split("@")[1])
            return mask_from_residual(x, shrink_graph_baseline(edge_index, x, sig, N))
        # --- IG-zero: reference suy bien -> mask ---
        if nm == "IG-zero":
            return mask_ig_zero(model, x, edge_index, target, T=args.N_ig)
        # --- THI NGHIEM CHINH: GNNExplainer cung thuat toan, khac REFERENCE ---
        if nm == "GNNExplainer(zero)":
            base = torch.zeros_like(x)                        # reference mac dinh = 0
            return gnnex_manual(model, x, edge_index, target, base, epochs=args.gnnex_epochs)
        if nm.startswith("GNNExplainer(shrink@"):
            tau = float(nm.split("@")[1].rstrip(")"))
            base = shrink_node_baseline(x, node_ref, tau)     # reference = b_tau(x)
            return gnnex_manual(model, x, edge_index, target, base, epochs=args.gnnex_epochs)
        # --- PGExplainer mac dinh ---
        if nm == "PGExplainer(default)":
            e = pg(x, edge_index, target=torch.tensor([target], device=x.device))
            return mask_from_explanation(e, N, x.device)
        raise ValueError(nm)

    print(f"{'mask source':<22}{'PermFid+↑':>11}{'PermFid-↓':>11}{'perm_gap↑':>12}")
    print("-" * 56)
    results = {}
    for nm in mask_sources:
        pp, pm, pg_gap = [], [], []
        for g in te_eval:
            g = g.to(device)
            x = g.x.float(); ei = g.edge_index; N = x.shape[0]
            with torch.no_grad():
                target = model(x, ei).argmax(1).item()
            try:
                m = get_mask(nm, x, ei, target, N)
            except Exception:
                continue
            r = perm_fidelity(model, x, ei, m, target, node_pool, gen,
                              topk_frac=args.topk_frac, K=args.K_perm)
            pp.append(r["permfid+"]); pm.append(r["permfid-"]); pg_gap.append(r["perm_gap"])
        if not pp:
            print(f"{nm:<22}{'-':>11}{'-':>11}{'(bo)':>12}"); continue
        mp, mm, mg = sum(pp)/len(pp), sum(pm)/len(pm), sum(pg_gap)/len(pg_gap)
        results[nm] = mg
        print(f"{nm:<22}{mp:>11.4f}{mm:>11.4f}{mg:>12.4f}")
    print("-" * 56)
    if results:
        best = max(results, key=results.get)
        print(f"[i] best perm_gap: {best} = {results[best]:.4f}")
        # so shrinkage tot nhat vs mask mac dinh
        shr = {k: v for k, v in results.items() if k.startswith("Shrink")}
        dfl = {k: v for k, v in results.items() if "default" in k}
        if shr and dfl:
            bs = max(shr, key=shr.get); bd = max(dfl, key=dfl.get)
            print(f"[i] shrinkage tot nhat: {bs}={shr[bs]:.4f}   |   "
                  f"mask mac dinh tot nhat: {bd}={dfl[bd]:.4f}   |   "
                  f"chenh = {shr[bs]-dfl[bd]:+.4f}")


if __name__ == "__main__":
    main()