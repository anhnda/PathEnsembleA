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


@torch.no_grad()
def insertion_deletion(
    model,
    x: torch.Tensor,          # (3, H, W) da chuan hoa, tren cuda
    attr: torch.Tensor,       # (3, H, W) attribution
    target: int,
    device: str = "cuda",
    steps: int = 224,         # so buoc (moi buoc them/bot mot khoi pixel)
    substrate: str = "blur",  # 'blur' hoac 'black' cho nen insertion / gia tri xoa deletion
    batch: int = 32,
):
    """
    Tra ve dict: insertion_auc, deletion_auc, id_gap, insertion_curve, deletion_curve.
    AUC tinh bang trung binh score (softmax target) tren cac buoc, thang [0,1].
    """
    C, H, W = x.shape
    N = H * W
    x = x.to(device)
    attr = attr.to(device)

    if substrate == "blur":
        base = _gaussian_blur(x)
    elif substrate == "black":
        base = torch.zeros_like(x)  # 0 trong khong gian da chuan hoa
    else:
        raise ValueError(substrate)

    order = torch.argsort(_pixel_scores(attr), descending=True)  # (N,) chi so pixel

    # so pixel them vao sau moi buoc
    pcts = torch.linspace(0, N, steps + 1, device=device).round().long()

    def score_of(imgs: torch.Tensor) -> torch.Tensor:
        """imgs: (B, 3, H, W) -> (B,) softmax prob cua target."""
        outs = []
        for i in range(0, imgs.shape[0], batch):
            logit = model(imgs[i:i + batch])
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

    ins_auc = ins_scores.mean().item()
    del_auc = del_scores.mean().item()
    return {
        "insertion_auc": ins_auc,
        "deletion_auc": del_auc,
        "id_gap": ins_auc - del_auc,
        "insertion_curve": ins_scores.detach().cpu(),
        "deletion_curve": del_scores.detach().cpu(),
    }
