"""
Insertion / Deletion faithfulness (RISE-style), torch GPU.

- Diem attribution gom theo pixel = tong |attr| tren 3 kenh RGB (mac dinh),
  roi xep hang giam dan.
- Insertion: bat dau tu anh mo (blur) / den, chen dan pixel quan trong nhat vao,
  do score lop target sau moi buoc -> AUC cao la tot.
- Deletion:  bat dau tu anh goc, xoa dan pixel quan trong nhat (thay bang blur/den),
  do score -> AUC thap la tot.
- I-D gap = insertion_AUC - deletion_AUC  (cao hon = attribution trung thuc hon).

Score duoc chuan hoa qua softmax cua target class de AUC nam trong [0,1].
"""

from __future__ import annotations
import torch
import torch.nn.functional as F


def _pixel_scores(attr: torch.Tensor) -> torch.Tensor:
    """attr: (3, H, W) -> (H*W,) diem theo pixel = tong tri tuyet doi tren kenh."""
    return attr.abs().sum(dim=0).reshape(-1)


def _gaussian_blur(x: torch.Tensor, k: int = 31, sigma: float | None = None) -> torch.Tensor:
    """x: (3, H, W) -> lam mo, tra (3, H, W). Dung lam 'anh nen' cho insertion/deletion."""
    C, H, W = x.shape
    if sigma is None:
        sigma = k / 3.0
    coords = torch.arange(k, device=x.device).float() - k // 2
    g1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g1d = g1d / g1d.sum()
    xb = x[None]
    xb = F.conv2d(xb, g1d.view(1, 1, 1, k).repeat(C, 1, 1, 1), padding=(0, k // 2), groups=C)
    xb = F.conv2d(xb, g1d.view(1, 1, k, 1).repeat(C, 1, 1, 1), padding=(k // 2, 0), groups=C)
    return xb[0]


# ---------------------------------------------------------------------------
# REFERENCE SET cho metric ky vong (KHONG dung 1 substrate co dinh).
# Ly do: neu substrate = blur thi cac method dung path blur/diffusion duoc cham
# bang CHINH baseline cua no -> tu-nhat-quan, khong phai faithfulness. Trung binh
# tren mot phan phoi reference method-agnostic moi la measure dung.
# ---------------------------------------------------------------------------
@torch.no_grad()
def make_reference_set(
    x: torch.Tensor,
    n_refs: int = 8,
    seed: int = 0,
    real_bank: torch.Tensor | None = None,   # (M,3,H,W) anh THAT khac lop (tuy chon)
    sigmas=(0.5, 1.0, 2.0),                  # nhieu muc blur, khong chi k=31
    noise_stds=(0.5, 1.0),
):
    """
    Tra ve (names, refs) — refs: list cac tensor (3,H,W) cung device/dtype voi x.
    Thanh phan (method-agnostic, khong uu ai ho method nao):
      zeros / gaussian noise (nhieu seed) / blur nhieu sigma / dataset-mean / anh that.
    """
    device, dtype = x.device, x.dtype
    C, H, W = x.shape
    g = torch.Generator(device="cpu").manual_seed(seed)
    names, refs = [], []

    names.append("zero"); refs.append(torch.zeros_like(x))

    for i, sd in enumerate(noise_stds):
        n = torch.randn(x.shape, generator=g).to(device=device, dtype=dtype) * sd
        names.append(f"noise{sd}"); refs.append(n)

    for sg in sigmas:
        k = int(2 * round(3 * sg) + 1)
        names.append(f"blur{sg}"); refs.append(_gaussian_blur(x, k=max(k, 3), sigma=float(sg)))

    names.append("mean"); refs.append(x.mean(dim=(1, 2), keepdim=True).expand_as(x).contiguous())

    if real_bank is not None and real_bank.shape[0] > 0:
        M = real_bank.shape[0]
        idx = torch.randperm(M, generator=g)[: max(1, n_refs - len(refs))]
        for j in idx.tolist():
            names.append(f"real{j}"); refs.append(real_bank[j].to(device=device, dtype=dtype))

    return names[:n_refs], refs[:n_refs]


@torch.no_grad()
def insertion_deletion_fixed(
    model,
    x: torch.Tensor,          # (3, H, W) da chuan hoa, tren cuda
    attr: torch.Tensor,       # (3, H, W) attribution
    target: int,
    device: str = "cuda",
    steps: int = 224,         # so buoc (moi buoc them/bot mot khoi pixel)
    substrate: str = "blur",  # 'blur' / 'black' — CHI dung khi base=None
    base: torch.Tensor | None = None,   # reference TUONG MINH (uu tien hon substrate)
    batch: int = 32,
    score: str = "softmax",   # 'softmax' -> prob target; 'logit' -> raw logit target (DUNG CHUNG voi attribution)
    trapezoid: bool = True,   # AUC theo quy tac hinh thang tren truc ti le (khop RISE)
):
    """
    MOT reference. Tra ve dict: insertion_auc, deletion_auc, id_gap, curves.
    Chi dung truc tiep khi ban CO Y muon 1 reference; measure chinh la
    insertion_deletion_expected (trung binh tren nhieu reference).
    """
    C, H, W = x.shape
    N = H * W
    x = x.to(device)
    attr = attr.to(device)

    if base is not None:
        base = base.to(device=device, dtype=x.dtype)
    elif substrate == "blur":
        base = _gaussian_blur(x)
    elif substrate == "black":
        base = torch.zeros_like(x)  # 0 trong khong gian da chuan hoa
    else:
        raise ValueError(substrate)

    order = torch.argsort(_pixel_scores(attr), descending=True)  # (N,) chi so pixel

    # so pixel them vao sau moi buoc
    pcts = torch.linspace(0, N, steps + 1, device=device).round().long()

    def score_of(imgs: torch.Tensor) -> torch.Tensor:
        """imgs: (B, 3, H, W) -> (B,) score cua target theo co `score`."""
        outs = []
        for i in range(0, imgs.shape[0], batch):
            logit = model(imgs[i:i + batch])
            if score == "logit":
                outs.append(logit[:, target])
            else:
                outs.append(F.softmax(logit, dim=1)[:, target])
        return torch.cat(outs)

    x_flat = x.reshape(C, N)
    base_flat = base.reshape(C, N)

    # ----- INSERTION: bat dau tu base, chen dan pixel quan trong nhat cua x -----
    ins_imgs = []
    for kk in pcts:
        canvas = base_flat.clone()
        idx = order[:kk]
        canvas[:, idx] = x_flat[:, idx]
        ins_imgs.append(canvas.reshape(C, H, W))
    ins_scores = score_of(torch.stack(ins_imgs))

    # ----- DELETION: bat dau tu x, xoa dan pixel quan trong nhat (thay bang base) -----
    del_imgs = []
    for kk in pcts:
        canvas = x_flat.clone()
        idx = order[:kk]
        canvas[:, idx] = base_flat[:, idx]
        del_imgs.append(canvas.reshape(C, H, W))
    del_scores = score_of(torch.stack(del_imgs))

    if trapezoid:
        # truc x = ti le pixel trong [0,1], khoang deu -> trapz = (mean - (f0+fT)/(2T))
        frac = pcts.to(ins_scores.dtype) / float(N)
        ins_auc = torch.trapz(ins_scores, frac).item()
        del_auc = torch.trapz(del_scores, frac).item()
    else:
        ins_auc = ins_scores.mean().item()
        del_auc = del_scores.mean().item()
    return {
        "insertion_auc": ins_auc,
        "deletion_auc": del_auc,
        "id_gap": ins_auc - del_auc,
        "insertion_curve": ins_scores.detach().cpu(),
        "deletion_curve": del_scores.detach().cpu(),
    }


@torch.no_grad()
def insertion_deletion_expected(
    model,
    x: torch.Tensor,
    attr: torch.Tensor,
    target: int,
    device: str = "cuda",
    steps: int = 224,
    batch: int = 32,
    score: str = "softmax",
    refs=None,                  # list (3,H,W); None -> make_reference_set
    ref_names=None,
    n_refs: int = 8,
    seed: int = 0,
    real_bank: torch.Tensor | None = None,
    trapezoid: bool = True,
    return_per_ref: bool = True,
):
    """
    MEASURE CHINH: ky vong I-D tren mot PHAN PHOI reference, khong phai 1 substrate.

        insertion_auc = E_b[ AUC_ins(b) ],  deletion_auc = E_b[ AUC_del(b) ]
        id_gap        = E_b[ AUC_ins(b) - AUC_del(b) ]

    Tra them std tren cac reference: neu std lon => thu hang method PHU THUOC
    reference, do chinh la ket qua can bao cao (substrate co dinh se giau di).
    """
    if refs is None:
        ref_names, refs = make_reference_set(x, n_refs=n_refs, seed=seed, real_bank=real_bank)
    if ref_names is None:
        ref_names = [f"ref{i}" for i in range(len(refs))]

    ins_l, del_l, per_ref = [], [], {}
    for nm, b in zip(ref_names, refs):
        r = insertion_deletion_fixed(model, x, attr, target, device=device, steps=steps,
                                     base=b, batch=batch, score=score, trapezoid=trapezoid)
        ins_l.append(r["insertion_auc"]); del_l.append(r["deletion_auc"])
        if return_per_ref:
            per_ref[nm] = {"insertion_auc": r["insertion_auc"],
                           "deletion_auc": r["deletion_auc"],
                           "id_gap": r["id_gap"]}

    ins = torch.tensor(ins_l); dele = torch.tensor(del_l); gap = ins - dele
    _std = (lambda t: t.std(unbiased=True).item() if t.numel() > 1 else 0.0)
    return {
        "insertion_auc": ins.mean().item(), "insertion_std": _std(ins),
        "deletion_auc": dele.mean().item(), "deletion_std": _std(dele),
        "id_gap": gap.mean().item(), "id_gap_std": _std(gap),
        "n_refs": len(refs), "ref_names": list(ref_names),
        "per_ref": per_ref,
    }


# Ten cu -> tro toi ban FIXED (giu tuong thich nguoc cho script chua sua).
insertion_deletion = insertion_deletion_fixed
