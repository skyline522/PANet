
"""
ldr_aug.py — Line-domain Robustness (LDR) augmentation utilities

提供：random_photometric_aug(x)
x: [B,3,H,W] tensor, ImageNet normalized
返回同尺度增强后的 tensor（仍保持 ImageNet normalized）
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

def _denorm(x: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(x.device, x.dtype)
    std = IMAGENET_STD.to(x.device, x.dtype)
    return x * std + mean

def _norm(x: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(x.device, x.dtype)
    std = IMAGENET_STD.to(x.device, x.dtype)
    return (x - mean) / std

@torch.no_grad()
def random_photometric_aug(x: torch.Tensor,
                           p_gray: float = 0.2,
                           brightness: float = 0.25,
                           contrast: float = 0.25,
                           gamma: float = 0.25,
                           noise_std: float = 0.03,
                           blur: bool = True):
    """
    轻量、纯 torch 的 photometric 域随机化（模拟不同产线照明/成像差异）。
    """
    y = _denorm(x).clamp(0,1)

    B = y.shape[0]
    # brightness & contrast
    b = (torch.rand(B,1,1,1, device=y.device, dtype=y.dtype) * 2 - 1) * brightness
    c = 1.0 + (torch.rand(B,1,1,1, device=y.device, dtype=y.dtype) * 2 - 1) * contrast
    y = (y + b)
    mean = y.mean(dim=(2,3), keepdim=True)
    y = (y - mean) * c + mean

    # gamma
    g = 1.0 + (torch.rand(B,1,1,1, device=y.device, dtype=y.dtype) * 2 - 1) * gamma
    y = (y.clamp(0,1) + 1e-6) ** g

    # grayscale
    if p_gray > 0:
        m = (torch.rand(B,1,1,1, device=y.device) < p_gray).float()
        gray = (0.299*y[:,0:1] + 0.587*y[:,1:2] + 0.114*y[:,2:3])
        y = y*(1-m) + torch.cat([gray,gray,gray], dim=1)*m

    # noise
    if noise_std > 0:
        n = torch.randn_like(y) * noise_std
        y = (y + n).clamp(0,1)

    # blur (3x3)
    if blur:
        # simple box blur with prob 0.3
        prob = (torch.rand(B, device=y.device) < 0.3).float().view(B,1,1,1)
        if prob.sum() > 0:
            k = torch.ones((3,3), device=y.device, dtype=y.dtype) / 9.0
            k = k.view(1,1,3,3)
            y_blur = []
            for ch in range(3):
                y_blur.append(F.conv2d(y[:,ch:ch+1], k, padding=1))
            yb = torch.cat(y_blur, dim=1)
            y = y*(1-prob) + yb*prob

    y = y.clamp(0,1)
    return _norm(y)
