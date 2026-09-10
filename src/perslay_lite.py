#!/usr/bin/env python3
"""
PersLay-lite：可微拓扑向量化层（阶段 2C）
=========================================
简化版 PersLay（Carrière et al., AISTATS 2020）：
- 固定向量化（PI）= 手工高斯核 + 手工权重函数
- PersLay-lite  = 可学习权重 w_i = sigmoid(a·pers_i + b) + 可学习带宽 σ 的高斯核

输入：一批持久图（变长点集，padding 到统一长度）
输出：G×G 图像特征（展平），可接任意下游网络

不可微边界：ripser PH 计算不可微，PD 由上游预计算缓存（PersLay 原论文标准做法）。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def diagrams_to_padded(diagrams, max_points=None):
    """把变长持久图列表 padding 成统一张量。

    Args:
        diagrams: list of (n_i, 2) ndarray，每个是一维持久图 (birth, death)
        max_points: padding 长度，None 则取批内最大

    Returns:
        pts:   (B, N, 2) tensor，padding 点为 (0,0)
        mask:  (B, N) bool tensor，True = 真实点
    """
    finite = []
    for dgm in diagrams:
        d = np.asarray(dgm, dtype=np.float64)
        if len(d) > 0:
            d = d[~np.isinf(d[:, 1])]      # 去掉无穷死亡点
            d = d[d[:, 1] > d[:, 0]]       # 去掉零持久点
        finite.append(d if len(d) > 0 else np.zeros((0, 2)))
    if max_points is None:
        max_points = max(len(d) for d in finite)
        max_points = max(max_points, 1)
    B = len(finite)
    pts = np.zeros((B, max_points, 2), dtype=np.float32)
    mask = np.zeros((B, max_points), dtype=bool)
    for i, d in enumerate(finite):
        n = min(len(d), max_points)
        if n > 0:
            pts[i, :n] = d[:n]
            mask[i, :n] = True
    return torch.from_numpy(pts), torch.from_numpy(mask)


class PersLayLite(nn.Module):
    """可微向量化层：PD → G×G 图像特征。

    参数（全部可学习）：
        a, b: 权重函数 w = sigmoid(a·pers + b) 的斜率/截距
        log_sigma: 高斯核带宽（log 参数化保证正性）
    """

    def __init__(self, grid_size=20, init_sigma=0.1, pers_range=(0.0, 3.0)):
        super().__init__()
        self.G = grid_size
        self.register_buffer(
            "grid_b",  # 网格 birth 轴
            torch.linspace(pers_range[0], pers_range[1], grid_size),
        )
        self.register_buffer(
            "grid_p",  # 网格 persistence 轴
            torch.linspace(0.0, pers_range[1], grid_size),
        )
        # 可学习参数
        self.a = nn.Parameter(torch.tensor(2.0))
        self.b = nn.Parameter(torch.tensor(-1.0))
        self.log_sigma = nn.Parameter(torch.tensor(float(np.log(init_sigma))))

    def forward(self, pts, mask):
        """
        Args:
            pts:  (B, N, 2) 持久点 (birth, death)
            mask: (B, N) bool
        Returns:
            img:  (B, G*G) 展平图像特征
        """
        B, N, _ = pts.shape
        birth = pts[:, :, 0]                          # (B, N)
        pers = pts[:, :, 1] - pts[:, :, 0]            # (B, N)

        # 可学习权重
        w = torch.sigmoid(self.a * pers + self.b)     # (B, N)
        w = w * mask.float()                          # padding 点权重置 0

        sigma = torch.exp(self.log_sigma).clamp(min=1e-3, max=5.0)

        # 高斯核投影到网格: (B, N, G, G)
        # img[b,i,j] = Σ_n w_n · exp(-((birth_n-gb_i)² + (pers_n-gp_j)²)/(2σ²))
        db = birth[:, :, None, None] - self.grid_b[None, None, :, None]   # (B,N,G,1)
        dp = pers[:, :, None, None] - self.grid_p[None, None, None, :]    # (B,N,1,G)
        kernel = torch.exp(-(db ** 2 + dp ** 2) / (2 * sigma ** 2))
        img = (w[:, :, None, None] * kernel).sum(dim=1)                   # (B,G,G)

        return img.reshape(B, -1)


class TopoClassifier(nn.Module):
    """PersLayLite + MLP 分类头（可选拼接时频特征，支持二分类/多分类）。"""

    def __init__(self, grid_size=20, tf_dim=0, hidden=32, dropout=0.3,
                 init_sigma=0.1, pers_range=(0.0, 3.0), num_classes=1):
        super().__init__()
        self.perslay = PersLayLite(grid_size, init_sigma, pers_range)
        in_dim = grid_size * grid_size + tf_dim
        self.tf_dim = tf_dim
        self.num_classes = num_classes
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, pts, mask, tf_feats=None):
        topo = self.perslay(pts, mask)
        if self.tf_dim > 0 and tf_feats is not None:
            x = torch.cat([topo, tf_feats], dim=1)
        else:
            x = topo
        out = self.head(x)
        # 二分类(num_classes=1)输出 logit 标量；多分类输出 (B, C) logits
        return out.squeeze(-1) if self.num_classes == 1 else out
