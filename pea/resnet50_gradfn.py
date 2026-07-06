"""
grad_fn cho ResNet-50 (torchvision). Mặc định chạy trên cuda.

grad_fn(states) nhan batch (T, 3, H, W) da chuan hoa ImageNet, tra ve
grad cua logit lop `target` theo input, shape (T, 3, H, W).

Khong train gi, chi forward + backward de lay gradient.
"""

from __future__ import annotations
import torch
import torchvision
import torchvision.transforms as T


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_resnet50(device: str = "cuda") -> torch.nn.Module:
    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    model = torchvision.models.resnet50(weights=weights)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def make_resnet50_gradfn(model: torch.nn.Module, target: int, device: str = "cuda",
                         chunk: int = 32):
    """
    target: chi so lop ImageNet (0..999) can giai thich.
    chunk : so anh moi mini-batch forward/backward (giam de tranh OOM).
    Tra ve grad_fn dung cho path_ensemble_attribution.
    """
    def grad_fn(states: torch.Tensor) -> torch.Tensor:
        # states: (T, 3, H, W) — cac trang thai noi suy gamma_r
        T = states.shape[0]
        out = torch.empty_like(states)
        for i in range(0, T, chunk):
            sb = states[i:i + chunk].to(device).clone().requires_grad_(True)
            logits = model(sb)                       # (b, 1000)
            score = logits[:, target].sum()          # grad tach roi theo hang
            grad, = torch.autograd.grad(score, sb)   # (b, 3, H, W)
            out[i:i + chunk] = grad.detach()
            del sb, logits, score, grad
        return out
    return grad_fn


def preprocess(pil_img, size: int = 224, device: str = "cuda") -> torch.Tensor:
    """PIL -> tensor (3, H, W) da chuan hoa, tren cuda."""
    tf = T.Compose([
        T.Resize(256),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return tf(pil_img.convert("RGB")).to(device)


def black_baseline(shape, device: str = "cuda") -> torch.Tensor:
    """Baseline anh den sau khi chuan hoa (gia tri 0 truoc chuan hoa)."""
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(3, 1, 1)
    C, H, W = shape
    zero_img = torch.zeros(3, H, W, device=device)
    return (zero_img - mean) / std
