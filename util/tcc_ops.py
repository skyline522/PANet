
"""
tcc_ops.py — utilities for Chapter 4 (TCC-Head)

提供：
  - make_edge_target(): 根据 label 生成边界带 target（binary）
  - make_skeleton_target(): 基于形态学 skeletonization 生成骨架 target（binary）
  - balanced_bce_with_logits(): 稀疏目标（edge/skeleton）更稳定的 BCE

注意：这些 target 建议在“模型 logits 分辨率”上生成（训练脚本中已按 nearest 下采样 label）。
"""

from __future__ import annotations
import torch
import torch.nn.functional as F


def _dilate(x: torch.Tensor, r: int = 1) -> torch.Tensor:
    if r <= 0:
        return x
    k = 2 * r + 1
    return F.max_pool2d(x, kernel_size=k, stride=1, padding=r)


def _erode(x: torch.Tensor, r: int = 1) -> torch.Tensor:
    if r <= 0:
        return x
    k = 2 * r + 1
    # minpool via maxpool on inverted
    return 1.0 - F.max_pool2d(1.0 - x, kernel_size=k, stride=1, padding=r)


def make_edge_target(label: torch.Tensor, ignore_index: int = 255, radius: int = 2) -> torch.Tensor:
    """
    label: [B,H,W] long, multi-class (0=bg, 1..=fg)
    return: edge target [B,1,H,W] float {0,1}, edges around fg regions
    """
    if label.dim() != 3:
        raise ValueError("label must be [B,H,W]")
    B, H, W = label.shape
    valid = (label != ignore_index)

    fg = ((label > 0) & valid).float().unsqueeze(1)  # [B,1,H,W]

    # morphological gradient to get boundary: dilate(fg) - erode(fg)
    edge = (_dilate(fg, r=1) - _erode(fg, r=1)).clamp(min=0.0, max=1.0)
    edge = (edge > 0.0).float()

    if radius > 0:
        edge = _dilate(edge, r=radius)

    # ignore pixels
    edge = edge * valid.float().unsqueeze(1)
    return edge


def make_skeleton_target(label: torch.Tensor, ignore_index: int = 255, iters: int = 25, radius: int = 1) -> torch.Tensor:
    """
    基于形态学 skeletonization（经典公式：S = ⋃(Erosion^k - Open(Erosion^k))）
    label: [B,H,W] long
    return: skeleton [B,1,H,W] float {0,1}
    """
    if label.dim() != 3:
        raise ValueError("label must be [B,H,W]")
    valid = (label != ignore_index)
    img = ((label > 0) & valid).float().unsqueeze(1)  # binary fg

    skel = torch.zeros_like(img)
    cur = img

    for _ in range(max(1, int(iters))):
        er = _erode(cur, r=1)
        op = _dilate(_erode(er, r=1), r=1)  # open(er): erode then dilate
        delta = (er - op).clamp(min=0.0, max=1.0)
        skel = torch.max(skel, delta)
        cur = er
        # stop if empty
        if float(cur.max()) <= 0.0:
            break

    if radius > 0:
        skel = _dilate(skel, r=radius)

    skel = (skel > 0.0).float()
    skel = skel * valid.float().unsqueeze(1)
    return skel


def balanced_bce_with_logits(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor | None = None, eps: float = 1e-6):
    """
    logits: [B,1,H,W]
    target: [B,1,H,W] in {0,1}
    valid : [B,1,H,W] in {0,1} (optional)
    使用 batch 内 pos/neg 比例设置 pos_weight，缓解骨架/边界稀疏导致训练不稳定。
    """
    if valid is None:
        valid = torch.ones_like(target)

    # mask out invalid pixels
    t = target * valid
    v = valid

    pos = t.sum()
    neg = v.sum() - pos
    pos_weight = (neg / (pos + eps)).clamp(min=1.0, max=50.0).detach()

    loss = F.binary_cross_entropy_with_logits(logits, t, reduction="none", pos_weight=pos_weight)
    loss = (loss * v).sum() / v.sum().clamp(min=1.0)
    return loss
