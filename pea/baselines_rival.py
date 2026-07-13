"""
baselines_rival.py — DOI TRONG cho Shrinkage-IG tren ANH.
Ba phuong phap "derive/learn baseline" de so cong bang trong E1:

  1) IG2  (Zhuo & Ge, TPAMI'24, arXiv 2406.10852)
       Baseline = GradCF: toi thieu hoa khoang cach BIEU DIEN (penultimate layer)
       giua explicand va mot counterfactual reference x^r (anh lop khac), bang
       gradient-descent CO CHUAN HOA tung buoc (Alg.1). Path = GradPath (quy dao
       cua chuoi perturbation). Attribution = tich HAI gradient:
         phi_i = sum_j  d f(gamma_j)/dx_i  *  d ||rep(gamma_j)-rep(x^r)||/dx_i * eta/W_j
       -> day la doi trong MANH nhat: cung "derive baseline", cung chong black.
       Khac shrinkage: can representation layer + reference lop khac + toi uu lap
       moi anh (dat), va KHONG dinh nghia cho tabular/embedding tuy y (can rep-layer).

  2) Max-Entropy baseline (Tan, IJCNN'23, arXiv 2204.05948)
       "No information" = OUTPUT cua model gan UNIFORM nhat. Tim baseline bang GD
       tren input de min KL(softmax(f(b)) || uniform). Roi IG THANG tu baseline do.
       Khac shrinkage: dinh nghia missingness o KHONG GIAN OUTPUT (uniform pred),
       khong phai khong gian INPUT (in-distribution removal).

  3) FRInGe (arXiv 2605.06404) — ban thuc dung.
       Reference max-entropy trong PREDICTIVE space + di theo geodesic Fisher-Rao
       tren simplex xac suat (thay duong thang input). O day ta hien thuc:
         - reference = max-entropy baseline (nhu (2)),
         - PATH noi suy tren simplex: p(t) geodesic tu p_uniform -> p(x) trong khong
           gian can bac hai xac suat (Fisher-Rao tren simplex = goc tren hinh cau),
           roi lay diem input xap xi bang noi suy thang co dieu chinh theo p(t).
       Ghi ro: ban RUT GON (khong lam pullback-metric day du), du de lam doi trong.

Tat ca tra ve attribution (3,H,W) — cung dinh dang ig_single, cam thang vao
attributions_for_image cua e1_batch_image.py.

torch GPU. Ban tu chay. KHONG smoketest.
"""
import torch
import torch.nn.functional as F


# ===========================================================================
# Representation extractor cho resnet50 (penultimate: sau avgpool, truoc fc)
# ===========================================================================
def resnet50_penultimate(model):
    """Tra ve ham rep(x_batch)->(B,2048): activation truoc lop fc cuoi."""
    modules = list(model.children())[:-1]        # bo fc cuoi
    feat = torch.nn.Sequential(*modules)
    def rep(x):
        h = feat(x)                              # (B,2048,1,1)
        return h.flatten(1)                      # (B,2048)
    return rep


# ===========================================================================
# 1) IG2 — GradCF + GradPath  (Alg.1 + Eq.3/Eq.8, backward difference)
# ===========================================================================
def ig2_attribution(model, x, x_ref, target, rep_fn,
                    steps=50, step_size=None, device="cuda", chunk=16):
    """
    x     : explicand (3,H,W)
    x_ref : counterfactual reference (3,H,W) — anh LOP KHAC
    target: lop cua explicand (de tich gradient f)
    rep_fn: ham representation (penultimate)
    Tra ve phi (3,H,W).
    """
    C, H, W = x.shape
    eta = step_size if step_size is not None else (x.abs().mean().item() * 4.0)
    x_ref_rep = rep_fn(x_ref[None]).detach()     # (1,2048)

    # --- Stage 1: build GradPath (Alg.1) tu explicand -> GradCF ---
    delta = torch.zeros_like(x)
    path = [ (x + delta).detach().clone() ]      # gamma(1)=x, se them dan ve gamma(0)
    for _ in range(steps):
        xd = (x + delta).clone().requires_grad_(True)
        rep = rep_fn(xd[None])
        d = (rep - x_ref_rep).pow(2).sum()       # ||rep(x+delta)-rep(x^r)||^2
        g, = torch.autograd.grad(d, xd)
        Wn = g.norm(p=2) + 1e-12                 # l2 normalization (Eq.20)
        delta = (delta - eta * g / Wn).detach()
        path.append((x + delta).detach().clone())
    # path[0]=x (gamma(1)) ... path[-1]=GradCF (gamma(0)); dao lai cho tang theo alpha
    path = path[::-1]                            # gamma(0)..gamma(1) = GradCF..x

    # --- Stage 2: integrate 2 gradients tren GradPath (Eq.8, backward diff) ---
    phi = torch.zeros_like(x)
    for j in range(len(path) - 1):
        gj = path[j].clone().requires_grad_(True)
        # explicand's gradient: d f_target / dx
        logit = model(gj[None])[0, target]
        gf, = torch.autograd.grad(logit, gj, retain_graph=True)
        # counterfactual gradient: d ||rep(gamma)-rep(x^r)|| / dx  (path direction proxy)
        rep = rep_fn(gj[None])
        dcf = (rep - x_ref_rep).pow(2).sum().sqrt()
        gc, = torch.autograd.grad(dcf, gj)
        Wn = gc.norm(p=2) + 1e-12
        phi += (gf.detach() * gc.detach()) * (eta / Wn)
    return phi


def sample_counterfactual_ref(model, x, target, pool, device="cuda"):
    """Chon 1 anh trong pool co lop KHAC target lam reference (uu tien lop du doan cao)."""
    with torch.no_grad():
        best, best_ref = -1e9, None
        for xr in pool:
            logit = model(xr[None])
            pc = logit.argmax(1).item()
            if pc != target:
                s = F.softmax(logit, 1)[0, pc].item()
                if s > best:
                    best, best_ref = s, xr
    return best_ref if best_ref is not None else pool[0]


# ===========================================================================
# 2) Max-Entropy baseline (Tan 2023): min KL(softmax(f(b)) || uniform)
# ===========================================================================
def max_entropy_baseline(model, x, n_class, steps=100, lr=0.05, device="cuda"):
    """Tim baseline b (khoi tu x) sao cho output gan uniform nhat. Tra ve b (3,H,W)."""
    b = x.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([b], lr=lr)
    logu = torch.full((1, n_class), 1.0 / n_class, device=device).log()
    for _ in range(steps):
        logp = F.log_softmax(model(b[None]), 1)
        # KL(p_model || uniform) = sum p (logp - logu); min -> p ~ uniform
        kl = (logp.exp() * (logp - logu)).sum()
        opt.zero_grad(); kl.backward(); opt.step()
    return b.detach()


def ig_from_baseline(x, x0, grad_fn, T=50):
    """IG thang tu baseline x0 -> x (giong ig_single, doc lap)."""
    device = x.device
    a = ((torch.arange(T, device=device) + 0.5) / T).view(-1, 1, 1, 1)
    states = x0[None] + a * (x - x0)[None]
    g = grad_fn(states)
    return g.mean(0) * (x - x0)


# ===========================================================================
# 3) FRInGe (ban rut gon): max-entropy reference + Fisher-Rao geodesic path
# ===========================================================================
def fringe_attribution(model, x, target, n_class, grad_fn,
                       steps=50, me_steps=100, device="cuda"):
    """
    Reference = max-entropy baseline b0 (predictive ~ uniform).
    Path input xap xi: gamma(t) = b0 + s(t)*(x-b0), voi s(t) uon theo geodesic
    Fisher-Rao tren simplex (goc tren hinh cau cua can bac hai xac suat) giua
    p_uniform va p(x). Tich phan gradient nhu IG tren path uon do.
    Ban RUT GON: dung 1 he so uon vo huong s(t) thay cho pullback-metric day du.
    """
    b0 = max_entropy_baseline(model, x, n_class, steps=me_steps, device=device)
    with torch.no_grad():
        p_x = F.softmax(model(x[None]), 1)[0]                 # (n_class,)
        p_u = torch.full_like(p_x, 1.0 / n_class)
        # goc Fisher-Rao tren simplex: theta = arccos(sum sqrt(p_u * p_x))
        bc = (p_u.sqrt() * p_x.sqrt()).sum().clamp(-1, 1)
        theta = torch.arccos(bc).item()

    device = x.device
    ts = (torch.arange(steps, device=device) + 0.5) / steps
    # geodesic tren hinh cau: trong so slerp -> he so uon s(t) in [0,1]
    if theta < 1e-4:
        s_of_t = ts                                          # gan nhu thang
    else:
        import math
        s_of_t = torch.tensor(
            [math.sin(t.item() * theta) / math.sin(theta) for t in ts],
            device=device)
    phi = torch.zeros_like(x)
    states = torch.stack([b0 + s * (x - b0) for s in s_of_t], 0)   # (steps,3,H,W)
    g = grad_fn(states)                                       # (steps,3,H,W)
    # Riemann sum voi buoc khong deu ds
    s_prev = torch.zeros(1, device=device)
    s_all = torch.cat([torch.zeros(1, device=device), s_of_t])
    for j in range(steps):
        ds = (s_all[j + 1] - s_all[j])
        phi += g[j] * (x - b0) * ds
    return phi


# ===========================================================================
# ============  BAN TABULAR (vector D-chieu, MLP)  ==========================
# ===========================================================================
# score_target-style scalar cho MLP tabular (khop synthetic_e0.score_target).
def _tab_score(model, x, target, score="softmax"):
    single = (x.dim() == 1)
    out = model(x[None] if single else x)
    if getattr(model, "n_out", 2) == 1:
        s = out.squeeze(-1)
    elif score == "logit":
        s = out[:, target]
    else:
        s = F.softmax(out, dim=1)[:, target]
    return s[0] if single else s


def mlp_penultimate(model):
    """Representation = output truoc lop Linear cuoi cua model.net (sau GELU thu 2)."""
    layers = list(model.net.children())
    feat = torch.nn.Sequential(*layers[:-1])          # bo Linear cuoi
    def rep(x):                                       # x: (M,D) hoac (D,)
        single = (x.dim() == 1)
        h = feat(x[None] if single else x)
        return h                                      # (M, hidden)
    return rep


# ---- 1) IG2 tabular (GradCF + GradPath) ----
def ig2_tabular(model, x, x_ref, target, rep_fn, steps=40, step_size=None,
                score="softmax"):
    """x,(x_ref): (D,). Tra ve phi (D,)."""
    eta = step_size if step_size is not None else (x.abs().mean().item() * 2.0 + 1e-3)
    x_ref_rep = rep_fn(x_ref).detach()
    delta = torch.zeros_like(x)
    path = [x.detach().clone()]
    for _ in range(steps):
        xd = (x + delta).clone().requires_grad_(True)
        d = (rep_fn(xd) - x_ref_rep).pow(2).sum()
        g, = torch.autograd.grad(d, xd)
        Wn = g.norm(p=2) + 1e-12
        delta = (delta - eta * g / Wn).detach()
        path.append((x + delta).detach().clone())
    path = path[::-1]                                 # GradCF..x
    phi = torch.zeros_like(x)
    for j in range(len(path) - 1):
        gj = path[j].clone().requires_grad_(True)
        logit = _tab_score(model, gj, target, score)
        gf, = torch.autograd.grad(logit, gj, retain_graph=True)
        dcf = (rep_fn(gj) - x_ref_rep).pow(2).sum().sqrt()
        gc, = torch.autograd.grad(dcf, gj)
        Wn = gc.norm(p=2) + 1e-12
        phi += (gf.detach() * gc.detach()) * (eta / Wn)
    return phi


def sample_cf_ref_tabular(model, x, target, pool, score="softmax"):
    """Chon 1 vector trong pool (M,D) co lop du doan KHAC target."""
    with torch.no_grad():
        preds = model(pool).argmax(1)
        mask = preds != target
        if mask.any():
            cand = pool[mask]
            # uu tien confident nhat o lop cua no
            sc = F.softmax(model(cand), 1).max(1).values
            return cand[sc.argmax()]
    return pool[0]


# ---- 2) Max-Entropy baseline tabular ----
def max_entropy_baseline_tab(model, x, n_class, steps=100, lr=0.05):
    b = x.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([b], lr=lr)
    logu = torch.full((1, n_class), 1.0 / n_class, device=x.device).log()
    for _ in range(steps):
        logp = F.log_softmax(model(b[None]), 1)
        kl = (logp.exp() * (logp - logu)).sum()
        opt.zero_grad(); kl.backward(); opt.step()
    return b.detach()


def ig_from_baseline_tab(model, x, x0, target, T=64, score="softmax"):
    """IG thang tu x0 -> x (tabular), dung score_target."""
    a = ((torch.arange(T, device=x.device) + 0.5) / T).view(-1, 1)
    states = x0[None] + a * (x - x0)[None]            # (T,D)
    states = states.clone().requires_grad_(True)
    s = _tab_score(model, states, target, score).sum()
    g, = torch.autograd.grad(s, states)
    return g.mean(0) * (x - x0)


# ---- 3) FRInGe tabular (max-ent ref + Fisher-Rao geodesic path) ----
def fringe_tabular(model, x, target, n_class, steps=50, me_steps=100, score="softmax"):
    import math
    b0 = max_entropy_baseline_tab(model, x, n_class, steps=me_steps)
    with torch.no_grad():
        p_x = F.softmax(model(x[None]), 1)[0]
        p_u = torch.full_like(p_x, 1.0 / n_class)
        bc = (p_u.sqrt() * p_x.sqrt()).sum().clamp(-1, 1)
        theta = torch.arccos(bc).item()
    ts = (torch.arange(steps, device=x.device) + 0.5) / steps
    if theta < 1e-4:
        s_of_t = ts
    else:
        s_of_t = torch.tensor([math.sin(t.item()*theta)/math.sin(theta) for t in ts],
                              device=x.device)
    s_all = torch.cat([torch.zeros(1, device=x.device), s_of_t])
    states = torch.stack([b0 + s * (x - b0) for s in s_of_t], 0)   # (steps,D)
    states = states.clone().requires_grad_(True)
    sc = _tab_score(model, states, target, score).sum()
    g, = torch.autograd.grad(sc, states)
    phi = torch.zeros_like(x)
    for j in range(steps):
        ds = (s_all[j + 1] - s_all[j])
        phi += g[j] * (x - b0) * ds
    return phi