"""
============================================================
ph_pipeline.py — 持续同调（PH）核心工具模块
============================================================
SRTP Pipeline 复现项目 · 核心工具函数
包含所有论文共享的 PH 计算、可视化、向量化等基础功能。

参考文献:
  - Paper 1 (2505.06583): 教学入门 + DNA 结构分析
  - Paper 2 (2601.01450): 喷注标记 (Jet Tagging)
  - Paper 3 (2602.14228): TopoGAT 显著特征选择
  - Paper 4 (2604.04299): 3DPHDL 设计空间
  - Paper 5 (NeurIPS 2023): 自适应过滤学习
============================================================
"""

import numpy as np
from typing import Tuple, Optional, List, Callable
import warnings

# ============================================================
# 1. 点云生成函数 (Paper 1 & 2 & 基础)
# ============================================================

def generate_circle(
    n_points: int = 100,
    radius: float = 1.0,
    noise: float = 0.05,
    dim: int = 3
) -> np.ndarray:
    """生成带噪声的圆环点云（有 H1 环）。

    对应 Paper 1 的示例点云、Paper 2 的喷注点云类比。

    Args:
        n_points: 点的数量
        radius: 圆环半径
        noise: 高斯噪声标准差
        dim: 嵌入维度 (2 或 3)

    Returns:
        shape (n_points, dim) 的点云数组
    """
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    x = radius * np.cos(theta) + np.random.normal(0, noise, n_points)
    y = radius * np.sin(theta) + np.random.normal(0, noise, n_points)
    if dim == 2:
        return np.column_stack((x, y))
    z = np.random.normal(0, noise, n_points)
    return np.column_stack((x, y, z))


def generate_disk(
    n_points: int = 100,
    radius: float = 1.0,
    noise: float = 0.05,
    dim: int = 3
) -> np.ndarray:
    """生成带噪声的圆盘点云（无 H1 环）。

    Args:
        n_points: 点的数量
        radius: 圆盘半径
        noise: 高斯噪声标准差
        dim: 嵌入维度 (2 或 3)

    Returns:
        shape (n_points, dim) 的点云数组
    """
    r = radius * np.sqrt(np.random.uniform(0, 1, n_points))
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    x = r * np.cos(theta) + np.random.normal(0, noise, n_points)
    y = r * np.sin(theta) + np.random.normal(0, noise, n_points)
    if dim == 2:
        return np.column_stack((x, y))
    z = np.random.normal(0, noise, n_points)
    return np.column_stack((x, y, z))


def generate_torus(
    n_points: int = 500,
    major_radius: float = 2.0,
    minor_radius: float = 1.0,
    noise: float = 0.05
) -> np.ndarray:
    """生成带噪声的环面点云（有 H1×2 + H2）。

    用于 Paper 4 的 3D 拓扑分析实验。

    Returns:
        shape (n_points, 3) 的点云数组
    """
    u = np.random.uniform(0, 2 * np.pi, n_points)
    v = np.random.uniform(0, 2 * np.pi, n_points)
    x = (major_radius + minor_radius * np.cos(v)) * np.cos(u)
    y = (major_radius + minor_radius * np.cos(v)) * np.sin(u)
    z = minor_radius * np.sin(v)
    x += np.random.normal(0, noise, n_points)
    y += np.random.normal(0, noise, n_points)
    z += np.random.normal(0, noise, n_points)
    return np.column_stack((x, y, z))


def generate_sphere(
    n_points: int = 500,
    radius: float = 1.0,
    noise: float = 0.05
) -> np.ndarray:
    """生成带噪声的球面点云（有 H2，无 H1）。

    Returns:
        shape (n_points, 3) 的点云数组
    """
    phi = np.random.uniform(0, 2 * np.pi, n_points)
    theta = np.arccos(np.random.uniform(-1, 1, n_points))
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    x += np.random.normal(0, noise, n_points)
    y += np.random.normal(0, noise, n_points)
    z += np.random.normal(0, noise, n_points)
    return np.column_stack((x, y, z))


# ============================================================
# 2. 持续同调计算 (Paper 1-5 通用)
# ============================================================

def compute_persistence_diagrams(
    points: np.ndarray,
    maxdim: int = 2,
    distance_matrix: bool = False
) -> List[np.ndarray]:
    """计算点云的持久同调，返回各维度的持久图。

    使用 Ripser 库（Paper 1, 2, 3 均使用）。
    支持距离矩阵输入（Paper 1 Remark 3.1, Paper 5 蛋白质数据）。

    Args:
        points: 点云数组 (n, d) 或距离矩阵 (n, n)
        maxdim: 最大同调维数
        distance_matrix: 是否已将 points 作为距离矩阵传入

    Returns:
        diagrams: 列表，diagrams[k] 为第 k 维的 (n_k, 2) 持久图数组
    """
    import ripser
    result = ripser.ripser(
        points,
        maxdim=maxdim,
        distance_matrix=distance_matrix
    )
    return result['dgms']


# ============================================================
# 3. PH 向量化方法 (Paper 2, 3, 4)
# ============================================================

def compute_top_k_persistences(
    diagrams: List[np.ndarray],
    dim: int = 1,
    top_k: int = 10
) -> np.ndarray:
    """提取 Top-K 持久性特征（距离对角线的寿命排序）。

    对应 Paper 1/2 的 baseline 方法和 SRTP test.py 的原始方法。

    Args:
        diagrams: 持久图列表
        dim: 同调维数
        top_k: 保留的 Top 特征数

    Returns:
        shape (top_k,) 的持久性特征向量
    """
    dgm = diagrams[dim]
    if len(dgm) == 0:
        return np.zeros(top_k)
    finite_pers = [(b, d) for b, d in dgm if not np.isinf(d)]
    pers = sorted([d - b for b, d in finite_pers], reverse=True)[:top_k]
    if len(pers) < top_k:
        pers = pers + [0] * (top_k - len(pers))
    return np.array(pers)


def compute_persistence_image(
    diagrams: List[np.ndarray],
    dim: int = 1,
    resolution: int = 40,
    sigma: float = 0.001,
    birth_range: Optional[Tuple[float, float]] = None,
    death_range: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """计算持久图像（Persistence Image）。

    对应 Paper 2 的核心向量化方法（H0/H1 持久图像 + CNN）。

    Args:
        diagrams: 持久图列表
        dim: 同调维数
        resolution: 网格分辨率
        sigma: 高斯核带宽
        birth_range: 出生范围 (min, max)
        death_range: 死亡范围 (min, max)

    Returns:
        shape (resolution, resolution) 的持久图像矩阵
    """
    from persim import PersistenceImager
    dgm = diagrams[dim]
    if len(dgm) == 0:
        return np.zeros((resolution, resolution))

    # 过滤无穷大点
    finite_mask = ~np.isinf(dgm[:, 1])
    dgm_finite = dgm[finite_mask]

    if len(dgm_finite) == 0:
        return np.zeros((resolution, resolution))

    # persim 的 PersistenceImager:
    #   birth_range = (b_min, b_max)  — x 轴（出生时间）
    #   pers_range  = (p_min, p_max)  — y 轴（持久性 = death - birth）
    #   pixel_size  — 控制网格分辨率
    # 必须保证 (b_max - b_min) / pixel_size ≈ (p_max - p_min) / pixel_size ≈ resolution
    # 才能得到 resolution × resolution 的方阵

    finite_pers = dgm_finite[:, 1] - dgm_finite[:, 0]  # 持久性 (death - birth)

    if birth_range is None:
        b_min, b_max = dgm_finite[:, 0].min(), dgm_finite[:, 0].max()
        if b_max == b_min:
            b_max = b_min + 1.0
        b_pad = (b_max - b_min) * 0.1
        birth_range = (b_min - b_pad, b_max + b_pad)
    if death_range is None:
        p_min, p_max = finite_pers.min(), finite_pers.max()
        if p_max == p_min:
            p_max = p_min + 1.0
        p_pad = (p_max - p_min) * 0.1
        death_range = (p_min - p_pad, p_max + p_pad)

    b_span = birth_range[1] - birth_range[0]
    p_span = death_range[1] - death_range[0]
    calc_pixel_size = max(b_span, p_span) / resolution

    pimgr = PersistenceImager(
        pixel_size=calc_pixel_size,
        birth_range=birth_range,
        pers_range=death_range
    )
    img = pimgr.transform(dgm_finite)
    return img


def compute_persistence_landscape(
    diagrams: List[np.ndarray],
    dim: int = 1,
    num_landscapes: int = 5,
    num_points: int = 100
) -> np.ndarray:
    """计算持久景观（Persistence Landscape）。

    对应 Paper 3 (TopoGAT) 和 Paper 4 的向量化方案。

    Args:
        diagrams: 持久图列表
        dim: 同调维数
        num_landscapes: 景观层数
        num_points: 采样点数

    Returns:
        shape (num_landscapes, num_points) 的景观函数矩阵
    """
    dgm = diagrams[dim]
    if len(dgm) == 0:
        return np.zeros((num_landscapes, num_points))

    finite_mask = ~np.isinf(dgm[:, 1])
    dgm_finite = dgm[finite_mask]

    if len(dgm_finite) == 0:
        return np.zeros((num_landscapes, num_points))

    # 计算持久对
    pers_pairs = []
    for b, d in dgm_finite:
        if d > b:
            pers_pairs.append((b, d))

    if not pers_pairs:
        return np.zeros((num_landscapes, num_points))

    # 确定采样范围
    births = np.array([b for b, d in pers_pairs])
    deaths = np.array([d for b, d in pers_pairs])
    t_min = max(0, births.min() - 0.1)
    t_max = deaths.max() + 0.1
    t_vals = np.linspace(t_min, t_max, num_points)

    # 计算景观函数
    landscapes = np.zeros((num_landscapes, num_points))
    for i, (b, d) in enumerate(pers_pairs):
        mid = (b + d) / 2
        half_len = (d - b) / 2
        for j, t in enumerate(t_vals):
            val = max(0, half_len - abs(t - mid))
            if val > 0:
                # 插入到正确的层
                inserted = False
                for k in range(num_landscapes):
                    if landscapes[k, j] < val:
                        # 下移
                        if k < num_landscapes - 1:
                            landscapes[k+1:, j] = landscapes[k:-1, j]
                        landscapes[k, j] = val
                        inserted = True
                        break
                if not inserted and num_landscapes > 0:
                    last_val = landscapes[-1, j]
                    if val > last_val:
                        landscapes[-1, j] = val

    return landscapes


# ============================================================
# 4. 显著特征选择 (Paper 3: TopoGAT)
# ============================================================

def _sigmoid_soft_mask(
    persistence: np.ndarray,
    threshold: float,
    eta: float = 10.0
) -> np.ndarray:
    """Sigmoid 软掩码，用于可微过滤持久图。

    对应 Paper 3 (TopoGAT) 公式 (3):
        mask_i = σ(η * (p_i - λ))

    Args:
        persistence: 持久性数组 (d - b)
        threshold: 阈值 λ
        eta: 软掩码控制参数

    Returns:
        mask: [0,1] 之间的掩码值
    """
    return 1.0 / (1.0 + np.exp(-eta * (persistence - threshold)))


def select_significant_features(
    diagram: np.ndarray,
    threshold: float = 0.1,
    method: str = 'topo',
    eta: float = 10.0
) -> Tuple[np.ndarray, np.ndarray]:
    """选择显著拓扑特征。

    支持多种方法（Paper 3 对比实验）:
    - 'topo': TopoGAT 式软掩码
    - 'hard': 硬阈值截断 (Top-K 风格)
    - 'stat': 统计置信带方法 (Fasy et al.)

    Args:
        diagram: 持久图 (n, 2)
        threshold: 阈值
        method: 选择方法
        eta: 软掩码参数

    Returns:
        (selected, mask): 显著特征和掩码
    """
    finite_mask = ~np.isinf(diagram[:, 1])
    dgm = diagram[finite_mask]

    if len(dgm) == 0:
        return np.array([]), np.array([])

    pers = dgm[:, 1] - dgm[:, 0]

    if method == 'topo':
        mask = _sigmoid_soft_mask(pers, threshold, eta)
        selected_mask = mask > 0.5
        return dgm[selected_mask], mask

    elif method == 'hard':
        selected_mask = pers > threshold
        return dgm[selected_mask], selected_mask.astype(float)

    elif method == 'stat':
        # 统计方法: 基于对角线的距离
        dist_to_diag = pers / np.sqrt(2)
        selected_mask = dist_to_diag > threshold
        return dgm[selected_mask], selected_mask.astype(float)

    else:
        raise ValueError(f"Unknown method: {method}")


# ============================================================
# 5. 加权过滤 (Paper 5: NeurIPS 2023)
# ============================================================

def compute_weighted_filtration(
    points: np.ndarray,
    weights: np.ndarray,
    max_edge: Optional[float] = None
) -> List[np.ndarray]:
    """计算加权 Rips 过滤的持久同调。

    对应 Paper 5 (NeurIPS 2023) 的加权过滤:
    - 为每个点分配权重 w(x)，半径 rx(t) = max(0, t - w(x))
    - 权重大的点膨胀得慢，权重小的点膨胀得快
    - 边 (i,j) 的出生时间: t = (d_ij + w_i + w_j) / 2
    - 顶点 i 的出生时间: t = w_i

    注意: ripser 只能处理顶点出生时间为 0 的情况，
    因此 H0 的出生时间会偏移，但 H1 的相对持久性正确。

    Args:
        points: 点云 (n, d)
        weights: 每个点的权重 (n,)
        max_edge: 最大边阈值（用于加速）

    Returns:
        diagrams: 持久图列表
    """
    import ripser

    n = len(points)
    # 加权 Rips 过滤的边出生时间: t_ij = (d_ij + w_i + w_j) / 2
    # 用距离矩阵 D'[i,j] = (d_ij + w_i + w_j) / 2 来模拟
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            raw_dist = np.linalg.norm(points[i] - points[j])
            effective_dist = (raw_dist + weights[i] + weights[j]) / 2.0
            dist_matrix[i, j] = effective_dist
            dist_matrix[j, i] = effective_dist

    return compute_persistence_diagrams(
        dist_matrix,
        maxdim=1,
        distance_matrix=True
    )


# ============================================================
# 6. 特征提取 Pipeline (Paper 1-5 通用)
# ============================================================

def ph_feature_pipeline(
    points: np.ndarray,
    method: str = 'top_k',
    maxdim: int = 2,
    **kwargs
) -> np.ndarray:
    """统一的 PH 特征提取 Pipeline。

    支持多种向量化方法:
    - 'top_k': Top-K 持久性
    - 'persistence_image': 持久图像
    - 'persistence_landscape': 持久景观
    - 'betti_curve': Betti 曲线

    Args:
        points: 点云 (n, d)
        method: 向量化方法
        maxdim: 最大同调维数
        **kwargs: 传递给具体方法的参数

    Returns:
        特征向量
    """
    diagrams = compute_persistence_diagrams(points, maxdim=maxdim)

    if method == 'top_k':
        dim = kwargs.get('dim', 1)
        k = kwargs.get('top_k', 10)
        return compute_top_k_persistences(diagrams, dim=dim, top_k=k)

    elif method == 'persistence_image':
        dim = kwargs.get('dim', 1)
        resolution = kwargs.get('resolution', 40)
        sigma = kwargs.get('sigma', 0.001)
        img = compute_persistence_image(
            diagrams, dim=dim, resolution=resolution, sigma=sigma
        )
        return img.flatten()

    elif method == 'persistence_landscape':
        dim = kwargs.get('dim', 1)
        num_landscapes = kwargs.get('num_landscapes', 5)
        num_points = kwargs.get('num_points', 100)
        landscape = compute_persistence_landscape(
            diagrams, dim=dim,
            num_landscapes=num_landscapes,
            num_points=num_points
        )
        return landscape.flatten()

    elif method == 'betti_curve':
        dim = kwargs.get('dim', 1)
        num_bins = kwargs.get('num_bins', 50)
        return compute_betti_curve(diagrams, dim=dim, num_bins=num_bins)

    else:
        raise ValueError(f"Unknown method: {method}")


def compute_betti_curve(
    diagrams: List[np.ndarray],
    dim: int = 1,
    num_bins: int = 50
) -> np.ndarray:
    """计算 Betti 曲线（各过滤参数下的 Betti 数）。

    对应 Paper 4 的简单向量化方法。

    Args:
        diagrams: 持久图列表
        dim: 同调维数
        num_bins: 采样点数

    Returns:
        shape (num_bins,) 的 Betti 曲线
    """
    dgm = diagrams[dim]
    if len(dgm) == 0:
        return np.zeros(num_bins)

    finite_mask = ~np.isinf(dgm[:, 1])
    dgm_finite = dgm[finite_mask]

    if len(dgm_finite) == 0:
        return np.zeros(num_bins)

    # 确定范围
    all_vals = np.concatenate([dgm_finite[:, 0], dgm_finite[:, 1]])
    t_min = max(0, all_vals.min() - 0.1)
    t_max = all_vals.max() + 0.1
    t_vals = np.linspace(t_min, t_max, num_bins)

    betti = np.zeros(num_bins)
    for b, d in dgm_finite:
        betti[(t_vals >= b) & (t_vals < d)] += 1

    return betti


# ============================================================
# 7. 评估指标
# ============================================================

def wasserstein_distance(
    dgm1: np.ndarray,
    dgm2: np.ndarray,
    p: int = 2
) -> float:
    """计算两个持久图之间的 p-Wasserstein 距离。

    对应 Paper 3 (TopoGAT) 的拓扑损失组成部分。

    Args:
        dgm1: 持久图 1
        dgm2: 持久图 2
        p: Wasserstein 距离的阶数

    Returns:
        Wasserstein 距离值
    """
    from persim import wasserstein
    # 过滤无穷大
    dgm1_finite = dgm1[~np.isinf(dgm1[:, 1])]
    dgm2_finite = dgm2[~np.isinf(dgm2[:, 1])]
    if len(dgm1_finite) == 0 and len(dgm2_finite) == 0:
        return 0.0
    return wasserstein(dgm1_finite, dgm2_finite, matching=False)


def bottleneck_distance(dgm1: np.ndarray, dgm2: np.ndarray) -> float:
    """计算两个持久图之间的 Bottleneck 距离。

    Args:
        dgm1: 持久图 1
        dgm2: 持久图 2

    Returns:
        Bottleneck 距离值
    """
    from persim import bottleneck
    dgm1_finite = dgm1[~np.isinf(dgm1[:, 1])]
    dgm2_finite = dgm2[~np.isinf(dgm2[:, 1])]
    if len(dgm1_finite) == 0 and len(dgm2_finite) == 0:
        return 0.0
    return bottleneck(dgm1_finite, dgm2_finite)


# ============================================================
# 8. 数据集批处理
# ============================================================

def batch_compute_ph_features(
    point_clouds: List[np.ndarray],
    labels: np.ndarray,
    method: str = 'top_k',
    maxdim: int = 2,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """批量计算点云数据集的 PH 特征。

    对应 Paper 1 (test.py) 和 Paper 2 的数据集 Pipeline。

    Args:
        point_clouds: 点云列表
        labels: 标签数组
        method: 向量化方法
        maxdim: 最大同调维数
        **kwargs: 传递给 ph_feature_pipeline 的参数

    Returns:
        (X_features, y_labels): 特征矩阵和标签
    """
    features = []
    for pts in point_clouds:
        feat = ph_feature_pipeline(pts, method=method, maxdim=maxdim, **kwargs)
        features.append(feat)
    return np.array(features), labels


def train_test_split_ph(
    point_clouds: List[np.ndarray],
    labels: np.ndarray,
    test_size: float = 0.3,
    random_state: int = 42,
    method: str = 'top_k',
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """划分训练/测试集并计算 PH 特征。

    Returns:
        (X_train, X_test, y_train, y_test)
    """
    from sklearn.model_selection import train_test_split as tts
    X_features, y_labels = batch_compute_ph_features(
        point_clouds, labels, method=method, **kwargs
    )
    return tts(X_features, y_labels, test_size=test_size, random_state=random_state)


# ============================================================
# 补充生成器：多空腔结构（阶段 1 子任务 A，验证 H2 空腔特征）
# ============================================================

def generate_porous_block(
    n_points: int = 2000,
    n_cavities: int = 4,
    box_size: float = 2.0,
    cavity_radius: float = 0.35,
    noise: float = 0.02
) -> np.ndarray:
    """生成立方体内部挖多个球形空腔的点云（有 H2 空腔特征）。

    采样点落在立方体 [-box/2, box/2]^3 内部、但所有空腔球体之外。
    空腔在立方体内随机放置（允许轻微重叠），每个空腔在 H2 持久图上
    表现为一个显著的、长生命的空腔特征。

    Args:
        n_points: 点的数量
        n_cavities: 空腔个数 (3~5)
        box_size: 立方体边长
        cavity_radius: 空腔球体半径
        noise: 高斯噪声标准差

    Returns:
        shape (n_points, 3) 的点云数组
    """
    half = box_size / 2.0
    pts = np.random.uniform(-half, half, (n_points, 3))

    # 随机放置空腔中心（在立方体内，留出边缘距离）
    margin = cavity_radius * 1.2
    cavities = np.random.uniform(-half + margin, half - margin, (n_cavities, 3))

    # 保留位于所有空腔之外的点（拒绝采样）
    keep = np.ones(n_points, dtype=bool)
    for c in cavities:
        dist2 = np.sum((pts - c) ** 2, axis=1)
        keep &= dist2 > cavity_radius ** 2

    accepted = pts[keep]
    # 若拒绝采样后点数不足，循环补采至满足要求
    while len(accepted) < n_points:
        extra = np.random.uniform(-half, half, (n_points, 3))
        ekeep = np.ones(n_points, dtype=bool)
        for c in cavities:
            dist2 = np.sum((extra - c) ** 2, axis=1)
            ekeep &= dist2 > cavity_radius ** 2
        accepted = np.vstack((accepted, extra[ekeep]))

    out = accepted[:n_points]
    out += np.random.normal(0, noise, out.shape)
    return out