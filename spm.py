# spm.py — Support-guided Prior Map (SPMM)
# 论文一致实现要点：
#   1) 使用支持高层特征 + 支持掩码得到“前景原型(高层)”
#   2) 与查询高层特征做余弦相似得到 prior
#   3) gate = sigmoid(alpha * prior)，并上采样到中层尺度
#   4) y_Q = q_feat_m * gate，用于后续原型/匹配
#
# 备注：为适配不同 backbone 的通道数，本模块内部用 1×1 conv 自动对齐通道到 C_mid。

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def l2norm(x: torch.Tensor, dim: int = 1, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(p=2, dim=dim, keepdim=True) + eps)


def masked_average_pool(feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    feat: [K,C,H,W]
    mask: [K,1,H,W] in {0,1}
    return: [1,C,1,1]
    """
    mask = mask.float()
    num = (feat * mask).sum(dim=(0, 2, 3), keepdim=True)
    den = mask.sum(dim=(0, 2, 3), keepdim=True).clamp(min=1.0)
    return num / den


class SPM(nn.Module):
    """
    SPMM: Support Prior-guided Matching Module

    Inputs:
      s_feat_h: [K,C_high,Hh,Wh]    support high-level feature (C5)
      s_mask  : [K,Hs,Ws]          support mask (0=bg, >0=fg)
      q_feat_h: [B,C_high,Hh,Wh]   query high-level feature (C5)
      q_feat_m: [B,C_mid,Hm,Wm]    query mid-level feature (after reduce/aspp)

    Outputs:
      y_Q  : [B,C_mid,Hm,Wm]       gated query mid-level feature
      prior: [B,1,Hm,Wm]           prior heatmap (upsampled to mid)
      gate : [B,1,Hm,Wm]           sigmoid(alpha * prior)
    """
    def __init__(self, alpha: float = 1.0, fuse: str = "mean", detach: bool = True):
        super().__init__()
        self.alpha = float(alpha)
        self.fuse = fuse
        self.detach = bool(detach)
        self._proj = None  # lazy 1x1 conv for channel alignment

    def _build_proj(self, c_in: int, c_out: int, device):
        self._proj = nn.Conv2d(c_in, c_out, kernel_size=1, bias=False).to(device)

    def forward(
        self,
        s_feat_h: torch.Tensor,
        s_mask: torch.Tensor,
        q_feat_h: torch.Tensor,
        q_feat_m: torch.Tensor,
    ):
        device = q_feat_m.device
        K, C_high, Hh, Wh = s_feat_h.shape
        B, C_mid, Hm, Wm = q_feat_m.shape

        # 0) mask -> high-level size, binary fg
        if s_mask.dim() == 4:
            s_mask_ = s_mask.squeeze(1)
        else:
            s_mask_ = s_mask
        fg = (s_mask_ > 0).float().unsqueeze(1)  # [K,1,Hs,Ws]
        fg = F.interpolate(fg, size=(Hh, Wh), mode="nearest")  # [K,1,Hh,Wh]

        # 1) channel align high->mid
        if self._proj is None:
            self._build_proj(C_high, C_mid, device)
        s_h = self._proj(s_feat_h)
        q_h = self._proj(q_feat_h)

        # 2) support fg prototype (high)
        s_proto = masked_average_pool(s_h, fg)  # [1,C_mid,1,1]
        s_proto = l2norm(s_proto, dim=1)

        # 3) cosine prior on query high-level
        q_h_n = l2norm(q_h, dim=1)
        prior_h = (q_h_n * s_proto).sum(dim=1, keepdim=True)  # [B,1,Hh,Wh]
        # 4) upsample prior to mid
        prior = F.interpolate(prior_h, size=(Hm, Wm), mode="bilinear", align_corners=False)

        # 5) gate + gated query mid
        gate = torch.sigmoid(self.alpha * prior)
        if self.detach:
            gate = gate.detach()
        y_Q = q_feat_m * gate

        return y_Q, prior, gate
