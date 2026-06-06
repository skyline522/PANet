# pml_loss.py — EPML (Edge-aware Pixel-wise Metric Loss)
# 论文一致实现要点：
#   • 在训练阶段，根据 Query GT 提取边界区域，构建边界权重
#   • 在边界附近对像素分类（基于 cosine similarity logits）进行加权强化
#
# 本实现将 EPML 具体化为：边界带权的 CrossEntropy(logits, label)
# logits 期望来自模型的“相似度 logits”（temperature * cosine similarity），shape [B, 1+n_way, h, w]

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def _edge_map_4n(label: torch.Tensor, ignore_index: int = 255) -> torch.Tensor:
    """
    label: [B,H,W] long
    return: [B,1,H,W] float in {0,1}
    4-neighborhood boundary indicator (ignore_index excluded)
    """
    B, H, W = label.shape

    edge = torch.zeros((B, H, W), device=label.device, dtype=torch.bool)

    # up/down
    valid_ud = (label[:, 1:, :] != ignore_index) & (label[:, :-1, :] != ignore_index)
    diff_ud = (label[:, 1:, :] != label[:, :-1, :]) & valid_ud
    edge[:, 1:, :] |= diff_ud
    edge[:, :-1, :] |= diff_ud

    # left/right
    valid_lr = (label[:, :, 1:] != ignore_index) & (label[:, :, :-1] != ignore_index)
    diff_lr = (label[:, :, 1:] != label[:, :, :-1]) & valid_lr
    edge[:, :, 1:] |= diff_lr
    edge[:, :, :-1] |= diff_lr

    return edge.unsqueeze(1).float()


class PixelMetricLoss(nn.Module):
    """
    Edge-aware Pixel-wise Metric Loss (EPML)

    Args:
      ignore_index: label to ignore
      edge_radius : boundary band radius (in pixels at the logits resolution)
      edge_weight : extra weight added on boundary band (final weight = 1 + edge_weight * band)
      normalize   : normalize weights to mean=1 over valid pixels
    """
    def __init__(self, ignore_index: int = 255, edge_radius: int = 3, edge_weight: float = 4.0, normalize: bool = True):
        super().__init__()
        self.ignore_index = int(ignore_index)
        self.edge_radius = int(edge_radius)
        self.edge_weight = float(edge_weight)
        self.normalize = bool(normalize)

    def forward(self, logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        logits: [B,C,h,w]
        label : [B,h,w]  (already downsampled to logits size)
        """
        if logits.dim() != 4:
            raise ValueError("logits must be [B,C,H,W]")
        if label.dim() != 3:
            raise ValueError("label must be [B,H,W]")
        B, C, H, W = logits.shape
        if label.shape[-2:] != (H, W):
            raise ValueError("label spatial size must match logits")

        # boundary band
        edge = _edge_map_4n(label, ignore_index=self.ignore_index)  # [B,1,H,W]
        if self.edge_radius > 0:
            k = 2 * self.edge_radius + 1
            band = F.max_pool2d(edge, kernel_size=k, stride=1, padding=self.edge_radius)
        else:
            band = edge

        weight = 1.0 + self.edge_weight * band  # [B,1,H,W]
        valid = (label != self.ignore_index).float().unsqueeze(1)
        weight = weight * valid

        if self.normalize:
            denom = valid.sum().clamp(min=1.0)
            mean_w = weight.sum() / denom
            weight = weight / mean_w.clamp(min=1e-6)

        ce = F.cross_entropy(logits, label, ignore_index=self.ignore_index, reduction="none")  # [B,H,W]
        ce = ce.unsqueeze(1)  # [B,1,H,W]
        loss = (ce * weight).sum() / valid.sum().clamp(min=1.0)
        return loss
