from __future__ import annotations
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from spm import SPM


def l2norm(x: torch.Tensor, dim: int = 1, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(p=2, dim=dim, keepdim=True) + eps)


def _safe_mean(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return torch.zeros(1, x.shape[-1], device=x.device, dtype=x.dtype)
    return x.mean(dim=0, keepdim=True)


class GroupNorm(nn.Module):
    def __init__(self, num_channels: int, num_groups: int = 32):
        super().__init__()
        g = min(num_groups, num_channels)
        while num_channels % g != 0 and g > 1:
            g -= 1
        self.gn = nn.GroupNorm(g, num_channels)

    def forward(self, x):
        return self.gn(x)


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        a = self.mlp(self.avg_pool(x))
        m = self.mlp(self.max_pool(x))
        return self.sigmoid(a + m)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        attn = self.conv(torch.cat([avg, mx], dim=1))
        return self.sigmoid(attn)


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class ASPP(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = 256, dilations=(1, 6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList()
        for d in dilations:
            if d == 1:
                self.branches.append(nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1, bias=False),
                    GroupNorm(out_ch),
                    nn.ReLU(inplace=True),
                ))
            else:
                self.branches.append(nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, padding=d, dilation=d, bias=False),
                    GroupNorm(out_ch),
                    nn.ReLU(inplace=True),
                ))
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * len(dilations), out_ch, 1, bias=False),
            GroupNorm(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.project(torch.cat([b(x) for b in self.branches], dim=1))


class TCCHead(nn.Module):
    def __init__(self, in_ch: int = 512, mid_ch: int = 256):
        super().__init__()
        self.skel = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            GroupNorm(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, 1, 1, bias=True),
        )
        self.edge = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            GroupNorm(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, 1, 1, bias=True),
        )

    def forward(self, y_q: torch.Tensor):
        return self.skel(y_q), self.edge(y_q)


class SelectionClusteringPrototypeModule(nn.Module):
    def __init__(self, n_parts: int = 6, tau: float = 0.2, min_reliable: int = 20, cluster_step: int = 30, kmeans_iters: int = 10):
        super().__init__()
        self.n_parts = int(n_parts)
        self.tau = float(tau)
        self.min_reliable = int(min_reliable)
        self.cluster_step = int(cluster_step)
        self.kmeans_iters = int(kmeans_iters)

    @staticmethod
    def _flatten_feat(x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        return x.permute(0, 2, 3, 1).reshape(-1, c)

    @torch.no_grad()
    def _mutual_nn_select(self, Sf: torch.Tensor, Qf: torch.Tensor, fg_mask_flat: torch.Tensor):
        Sf = l2norm(Sf.float(), dim=1)
        Qf = l2norm(Qf.float(), dim=1)

        fg_idx = torch.nonzero(fg_mask_flat > 0.5, as_tuple=True)[0]
        if fg_idx.numel() == 0:
            return fg_idx, None

        Sf_fg = Sf[fg_idx]
        sim_sq = Sf_fg @ Qf.t()
        best_sim, best_j = sim_sq.max(dim=1)
        sim_qs = Qf @ Sf.t()
        back_i = sim_qs.argmax(dim=1)
        mutual = (back_i[best_j] == fg_idx)
        keep = mutual & (best_sim >= self.tau)
        reliable_support_idx = fg_idx[keep]
        reliable_sim = best_sim[keep]
        return reliable_support_idx, reliable_sim

    @torch.no_grad()
    def _kmeans(self, X: torch.Tensor, k: int, iters: int):
        n, _ = X.shape
        perm = torch.randperm(n, device=X.device)
        centers = X[perm[:k]].clone()

        for _ in range(iters):
            sim = X @ centers.t()
            assign = sim.argmax(dim=1)
            for ci in range(k):
                m = (assign == ci)
                if m.any():
                    centers[ci] = l2norm(X[m].mean(dim=0), dim=0)

        sim = X @ centers.t()
        assign = sim.argmax(dim=1)
        return centers, assign

    def forward(self, s_feat: torch.Tensor, s_mask: torch.Tensor, q_feat: torch.Tensor, q_prior: Optional[torch.Tensor] = None):
        device = s_feat.device
        _, c, _, _ = s_feat.shape

        S_all = self._flatten_feat(s_feat).detach()
        m_flat = s_mask.reshape(-1).float()
        fg_idx = torch.nonzero(m_flat > 0.5, as_tuple=True)[0]
        bg_idx = torch.nonzero(m_flat < 0.5, as_tuple=True)[0]

        if bg_idx.numel() > 0:
            p_bg = l2norm(_safe_mean(S_all[bg_idx]), dim=1)
        else:
            if q_prior is not None:
                B = q_feat.shape[0]
                qf = q_feat.detach()
                prior = q_prior.detach()
                prior_flat = prior.view(B, -1)
                k_bg = max(1, int(0.2 * prior_flat.shape[1]))
                _, idx = torch.topk(prior_flat, k=k_bg, dim=1, largest=False)
                q_flat = qf.flatten(2).permute(0, 2, 1)
                bg_feats = [q_flat[b][idx[b]] for b in range(B)]
                bg_feats = torch.cat(bg_feats, dim=0)
                p_bg = l2norm(_safe_mean(bg_feats), dim=1)
            else:
                p_bg = l2norm(torch.zeros(1, c, device=device), dim=1)

        Q_all = self._flatten_feat(q_feat)
        reliable_idx, _ = self._mutual_nn_select(S_all, Q_all, m_flat)
        if reliable_idx.numel() == 0:
            reliable_idx = fg_idx

        if reliable_idx.numel() == 0:
            fg_centers = torch.zeros(self.n_parts, c, device=device)
            meta = {
                'cluster_sizes': torch.zeros(self.n_parts, device=device),
                'cluster_sims': torch.zeros(self.n_parts, device=device),
                'p_bg': p_bg,
            }
            return fg_centers, meta

        X = l2norm(S_all[reliable_idx], dim=1)
        Nr = X.shape[0]

        if Nr < self.min_reliable:
            centers = l2norm(_safe_mean(X), dim=1)
            assign = torch.zeros((Nr,), device=device, dtype=torch.long)
        else:
            k = max(1, Nr // max(1, self.cluster_step) + 1)
            k = min(self.n_parts, k)
            centers, assign = self._kmeans(X, k=k, iters=self.kmeans_iters)

        k_eff = centers.shape[0]
        sizes = torch.zeros((k_eff,), device=device)
        sims = torch.zeros((k_eff,), device=device)

        with torch.no_grad():
            sim_to_center = X @ centers.t()
            for ci in range(k_eff):
                m = (assign == ci)
                if m.any():
                    sizes[ci] = m.sum().float()
                    sims[ci] = sim_to_center[m, ci].mean()

        P = self.n_parts
        if k_eff < P:
            rep = P - k_eff
            centers = torch.cat([centers, centers[:1].repeat(rep, 1)], dim=0)
            sizes = torch.cat([sizes, sizes[:1].repeat(rep)], dim=0)
            sims = torch.cat([sims, sims[:1].repeat(rep)], dim=0)
        elif k_eff > P:
            centers = centers[:P]
            sizes = sizes[:P]
            sims = sims[:P]

        meta = {'cluster_sizes': sizes, 'cluster_sims': sims, 'p_bg': p_bg}
        return centers, meta


class DPANet_PML(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        n_way: int = 1,
        n_parts: int = 6,
        alpha: float = 1.0,
        tau: float = 0.2,
        temperature: float = 20.0,
        use_tcc_head: bool = True,
        tcc_fuse_skel: float = 0.25,
        tcc_fuse_edge: float = 0.15,
    ):
        super().__init__()
        self.backbone = backbone
        self.n_way = int(n_way)
        self.n_parts = int(n_parts)
        self.temperature = float(temperature)
        self.use_tcc_head = bool(use_tcc_head)
        self.tcc_fuse_skel = float(tcc_fuse_skel)
        self.tcc_fuse_edge = float(tcc_fuse_edge)

        self.reduce = nn.Sequential(
            nn.Conv2d(2048, 512, 1, bias=False),
            GroupNorm(512),
            nn.ReLU(inplace=True),
        )
        self.aspp = ASPP(512, out_ch=512, dilations=(1, 6, 12, 18))
        self.tcc_head = TCCHead(in_ch=512, mid_ch=256) if self.use_tcc_head else None

        self.spm = SPM(alpha=alpha, fuse='mean', detach=True)
        self.scpm_list = nn.ModuleList([
            SelectionClusteringPrototypeModule(n_parts=n_parts, tau=tau) for _ in range(self.n_way)
        ])

    @staticmethod
    def _pick_last_valid(feat_out):
        if isinstance(feat_out, (list, tuple)):
            for t in reversed(feat_out):
                if t is not None:
                    return t
            return feat_out[0]
        return feat_out

    @staticmethod
    def _downsample_mask(mask: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
        if mask.dim() == 4:
            mask = mask.squeeze(1)
        if mask.shape[-2:] != size_hw:
            mask = F.interpolate(mask.unsqueeze(1).float(), size=size_hw, mode='nearest').squeeze(1)
        return mask.long()

    def forward(self, support_imgs: torch.Tensor, support_masks: torch.Tensor, query_imgs: torch.Tensor):
        K = support_imgs.shape[0]
        B = query_imgs.shape[0]

        s_feat_raw = self.backbone(support_imgs)
        q_feat_raw = self.backbone(query_imgs)
        s_feat_h = self._pick_last_valid(s_feat_raw)
        q_feat_h = self._pick_last_valid(q_feat_raw)

        s_feat_m = self.aspp(self.reduce(s_feat_h))
        q_feat_m = self.aspp(self.reduce(q_feat_h))

        y_Q, prior, gate = self.spm(
            s_feat_h=s_feat_h,
            s_mask=support_masks.long(),
            q_feat_h=q_feat_h,
            q_feat_m=q_feat_m,
        )

        h, w = s_feat_m.shape[-2:]
        if y_Q.shape[-2:] != (h, w):
            y_Q = F.interpolate(y_Q, size=(h, w), mode='bilinear', align_corners=False)
            prior = F.interpolate(prior, size=(h, w), mode='bilinear', align_corners=False)
            gate = F.interpolate(gate, size=(h, w), mode='bilinear', align_corners=False)

        s_mask_ds = self._downsample_mask(support_masks, (h, w))

        skel_logits = None
        edge_logits = None
        skel_prob = None
        edge_prob = None
        if self.use_tcc_head and self.tcc_head is not None:
            skel_logits, edge_logits = self.tcc_head(y_Q)
            skel_prob = torch.sigmoid(skel_logits)
            edge_prob = torch.sigmoid(edge_logits)

        bg_feats = []
        for k in range(K):
            m = (s_mask_ds[k] == 0)
            if m.any():
                bg_feats.append(s_feat_m[k].permute(1, 2, 0)[m])
        if bg_feats:
            bg_proto = l2norm(_safe_mean(torch.cat(bg_feats, dim=0)), dim=1)
        else:
            q_flat = y_Q.detach().flatten(2).permute(0, 2, 1)
            prior_flat = prior.detach().view(B, -1)
            k_bg = max(1, int(0.2 * prior_flat.shape[1]))
            _, idx = torch.topk(prior_flat, k=k_bg, dim=1, largest=False)
            bgf = [q_flat[b][idx[b]] for b in range(B)]
            bg_proto = l2norm(_safe_mean(torch.cat(bgf, dim=0)), dim=1)

        C = y_Q.shape[1]
        q_flat = l2norm(y_Q.flatten(2).permute(0, 2, 1).reshape(-1, C), dim=1)
        bg_logit = (q_flat @ bg_proto.t()).view(B, 1, h, w)

        fg_logits = []
        fg_weights_dbg = []
        for way in range(self.n_way):
            way_mask = (s_mask_ds == (way + 1)).long()
            fg_protos, meta = self.scpm_list[way](s_feat_m, way_mask, y_Q, q_prior=prior)

            sizes = meta['cluster_sizes'].clamp(min=0.0)
            sims = meta['cluster_sims'].clamp(min=0.0)
            if sizes.sum() > 0:
                size_n = sizes / sizes.sum()
            else:
                size_n = torch.ones_like(sizes) / max(1, sizes.numel())

            score = size_n * sims
            if score.sum() > 0:
                w_proto = score / score.sum()
            else:
                w_proto = torch.ones_like(score) / max(1, score.numel())
            fg_weights_dbg.append(w_proto.detach().cpu())

            p = l2norm(fg_protos, dim=1)
            sim = q_flat @ p.t()
            fg = (sim * w_proto.view(1, -1)).sum(dim=1).view(B, 1, h, w)

            if skel_prob is not None and edge_prob is not None:
                structure_gain = 1.0 + self.tcc_fuse_skel * skel_prob + self.tcc_fuse_edge * edge_prob
                fg = fg * structure_gain

            fg_logits.append(fg)

        logits = torch.cat([bg_logit] + fg_logits, dim=1) * self.temperature
        aux: Dict[str, torch.Tensor] = {
            'q_feat': y_Q,
            'prior': prior,
            'gate': gate,
        }
        if skel_logits is not None and edge_logits is not None:
            aux['skel_logits'] = skel_logits
            aux['edge_logits'] = edge_logits
        aux['proto_weights'] = torch.stack([w for w in fg_weights_dbg], dim=0) if fg_weights_dbg else torch.empty(0)
        return logits, aux
