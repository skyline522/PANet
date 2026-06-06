from collections import OrderedDict

from .vgg import Encoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)  # 使用ReLU6实现

    def forward(self, x):
        return self.relu(x + 3) / 6  # 公式为ReLU6(x+3)/6，模拟Sigmoid激活函数

# 定义h_swish激活函数，这是基于h_sigmoid的Swish函数变体
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)  # 使用上面定义的h_sigmoid

    def forward(self, x):
        return x * self.sigmoid(x)  # 公式为x * h_sigmoid(x)

# 定义Coordinate Attention模块
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        # 定义水平和垂直方向的自适应平均池化
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 水平方向
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 垂直方向

        mip = max(8, inp // reduction)  # 计算中间层的通道数

        # 1x1卷积用于降维
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)  # 批归一化
        self.act = h_swish()  # 激活函数

        # 两个1x1卷积，分别对应水平和垂直方向
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x  # 保存输入作为残差连接

        n, c, h, w = x.size()  # 获取输入的尺寸
        x_h = self.pool_h(x)  # 水平方向池化
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # 垂直方向池化并交换维度以适应拼接

        y = torch.cat([x_h, x_w], dim=2)  # 拼接水平和垂直方向的特征
        y = self.conv1(y)  # 通过1x1卷积降维
        y = self.bn1(y)  # 批归一化
        y = self.act(y)  # 激活函数

        x_h, x_w = torch.split(y, [h, w], dim=2)  # 将特征拆分回水平和垂直方向
        x_w = x_w.permute(0, 1, 3, 2)  # 恢复x_w的原始维度

        a_h = self.conv_h(x_h).sigmoid()  # 通过1x1卷积并应用Sigmoid获取水平方向的注意力权重
        a_w = self.conv_w(x_w).sigmoid()  # 通过1x1卷积并应用Sigmoid获取垂直方向的注意力权重

        out = identity * a_w * a_h  # 应用注意力权重到输入特征，并与残差连接相乘

        return out  # 返回输出
    
class PPIE_CoordAtt(nn.Module):
    """
    带 CoordAttention 的 Part-based Prompt Initialization Encoder.
    """
    def __init__(self, in_channels, prompt_dim=64, num_parts=2, ca_reduction=32):
        super(PPIE_CoordAtt, self).__init__()
        self.in_channels = in_channels
        self.prompt_dim = prompt_dim
        self.num_parts = num_parts

        # prompt编码器
        self.prompt_encoder = nn.Sequential(
            nn.Linear(in_channels, prompt_dim),
            nn.ReLU(),
            nn.Linear(prompt_dim, prompt_dim)
        )
        # 融合层
        self.fusion_layer = nn.Conv2d(in_channels + prompt_dim, in_channels, 1)
        # CoordAttention
        self.coordatt = CoordAtt(in_channels, in_channels, reduction=ca_reduction)

    def masked_avg_pool(self, feat, mask):
        B, C, H, W = feat.shape
        masked = feat * mask
        sum_feat = masked.flatten(2).sum(dim=-1)      # (B, C)
        sum_mask = mask.flatten(2).sum(dim=-1) + 1e-6 # (B, 1)
        pooled = sum_feat / sum_mask                  # (B, C)
        return pooled

    def forward(self, support_feat, support_mask, query_feat):
        B, C, H, W = support_feat.shape
        device = support_feat.device

        # Step 1: 构造part掩码（例：前景、背景）
        part_masks = []
        for cls in range(self.num_parts):
            part_masks.append((support_mask == cls).float())
        part_masks = torch.cat(part_masks, dim=1)   # (B, N, H, W)

        # Step 2: 每个part做masked avg pool
        part_prompts = []
        for i in range(self.num_parts):
            pm = part_masks[:, i:i+1, :, :]         # (B, 1, H, W)
            pf = self.masked_avg_pool(support_feat, pm)  # (B, C)
            part_prompts.append(pf)
        part_prompts = torch.stack(part_prompts, dim=1)  # (B, N, C)

        # Step 3: prompt编码
        prompt_embs = self.prompt_encoder(part_prompts)  # (B, N, prompt_dim)
        prompt_emb = prompt_embs.mean(dim=1)             # (B, prompt_dim)

        # Step 4: 融合进query/support特征
        prompt_expand = prompt_emb.unsqueeze(-1).unsqueeze(-1).expand(B, self.prompt_dim, H, W)
        new_query = torch.cat([query_feat, prompt_expand], dim=1)        # (B, C+prompt_dim, H, W)
        new_support = torch.cat([support_feat, prompt_expand], dim=1)
        new_query = self.fusion_layer(new_query)        # (B, C, H, W)
        new_support = self.fusion_layer(new_support)

        # Step 5: CoordAttention空间增强
        new_query = self.coordatt(new_query)
        new_support = self.coordatt(new_support)

        return new_query, new_support