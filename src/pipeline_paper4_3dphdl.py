"""
============================================================
pipeline_paper4_3dphdl.py — Paper 4: 3DPHDL 设计空间
============================================================
基于 2604.04299v1:
  A Persistent Homology Design Space for 3D Point Cloud
  Deep Learning

Pipeline:
  1. 多种复形构造 (VR, Alpha, Witness)
  2. 多种向量化方法 (PI, PL, Betti曲线)
  3. 不同 backbone 的对比实验
  4. 噪声鲁棒性分析
  5. 计算效率对比

核心创新: 系统化对比 PH+DL 各设计选择的效果
============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path
import sys
import time
from typing import List, Dict, Tuple

rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

sys.path.insert(0, str(Path(__file__).parent))
from ph_pipeline import (
    compute_persistence_diagrams,
    compute_persistence_image,
    compute_persistence_landscape,
    compute_betti_curve,
    compute_top_k_persistences,
    generate_circle, generate_disk, generate_sphere, generate_torus,
    wasserstein_distance,
)

RESULTS_DIR = Path(__file__).parent.parent / "results"


# ============================================================
# 1. 数据生成: 多种拓扑形状
# ============================================================

def generate_all_shapes(n_points: int = 300) -> Dict[str, np.ndarray]:
    """生成不同拓扑类型的 3D 形状。

    对应 Paper 4 Figure 1 的 Betti 数标记。
    """
    shapes = {}
    shapes['sphere'] = generate_sphere(n_points, noise=0.03)
    shapes['torus'] = generate_torus(n_points, noise=0.03)
    shapes['circle'] = generate_circle(n_points, noise=0.03)
    shapes['disk'] = generate_disk(n_points, noise=0.03)
    return shapes


# ============================================================
# 2. 向量化方法对比
# ============================================================

def compare_vectorization_methods(
    diagrams: List[np.ndarray],
    dim: int = 1
) -> Dict[str, Tuple[float, np.ndarray]]:
    """比较不同向量化方法的性能。

    对应 Paper 4 Table 4 的实验设计。

    Returns:
        dict: {method_name: (compute_time, feature_vector)}
    """
    results = {}

    # Top-K 持久性
    t0 = time.time()
    topk = compute_top_k_persistences(diagrams, dim=dim, top_k=10)
    results['Top-K'] = (time.time() - t0, topk)

    # 持久图像
    t0 = time.time()
    pi = compute_persistence_image(diagrams, dim=dim, resolution=40, sigma=0.001)
    results['PI'] = (time.time() - t0, pi.flatten())

    # 持久景观
    t0 = time.time()
    pl = compute_persistence_landscape(diagrams, dim=dim, num_landscapes=5, num_points=100)
    results['PL'] = (time.time() - t0, pl.flatten())

    # Betti 曲线
    t0 = time.time()
    bc = compute_betti_curve(diagrams, dim=dim, num_bins=50)
    results['Betti Curve'] = (time.time() - t0, bc)

    return results


# ============================================================
# 3. 噪声鲁棒性分析
# ============================================================

def test_noise_robustness(
    points: np.ndarray,
    noise_levels: List[float],
    dim: int = 1
) -> Dict[str, List[float]]:
    """测试不同噪声水平下的拓扑稳定性。

    对应 Paper 4 Table 2-3 和图 9 的稳定性分析。

    Args:
        points: 原始点云
        noise_levels: 噪声标准差列表
        dim: 同调维数

    Returns:
        dict: {noise_level: wasserstein_distance}
    """
    original_diagrams = compute_persistence_diagrams(points, maxdim=dim)
    original_dgm = original_diagrams[dim]
    original_finite = original_dgm[~np.isinf(original_dgm[:, 1])]

    if len(original_finite) == 0:
        return {n: 0.0 for n in noise_levels}

    results = {'noise_level': [], 'W2': [], 'W2_H0': []}
    for noise in noise_levels:
        noisy_points = points + np.random.normal(0, noise, points.shape)
        noisy_diagrams = compute_persistence_diagrams(noisy_points, maxdim=dim)

        # H0 稳定性
        noisy_dgm0 = noisy_diagrams[0]
        noisy_finite0 = noisy_dgm0[~np.isinf(noisy_dgm0[:, 1])]
        w2_h0 = wasserstein_distance(
            original_diagrams[0][~np.isinf(original_diagrams[0][:, 1])],
            noisy_finite0, p=2
        ) if len(noisy_finite0) > 0 else 0

        # H1 稳定性
        noisy_dgm = noisy_diagrams[dim]
        noisy_finite = noisy_dgm[~np.isinf(noisy_dgm[:, 1])]
        w2 = wasserstein_distance(original_finite, noisy_finite, p=2) if len(noisy_finite) > 0 else 0

        results['noise_level'].append(noise)
        results['W2'].append(w2)
        results['W2_H0'].append(w2_h0)

    return results


# ============================================================
# 4. 简单分类器 (MLP)
# ============================================================

def train_mlp_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> float:
    """训练 MLP 分类器并返回测试准确率。

    对应 Paper 4 的 backbone 对比实验 (PointNet 简化版)。
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class MLP(nn.Module):
        def __init__(self, in_dim, num_classes):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(64, num_classes)
            )
        def forward(self, x):
            return self.net(x)

    model = MLP(X_train.shape[1], len(np.unique(y_train))).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    X_tr = torch.FloatTensor(X_train).to(device)
    y_tr = torch.LongTensor(y_train).to(device)
    X_te = torch.FloatTensor(X_test).to(device)
    y_te = torch.LongTensor(y_test).to(device)

    train_ds = TensorDataset(X_tr, y_tr)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)

    for epoch in range(50):
        model.train()
        for xb, yb in train_dl:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_te).argmax(1)
        acc = (preds == y_te).float().mean().item()

    return acc


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("Paper 4: 3DPHDL 设计空间 Pipeline")
    print("=" * 60)

    np.random.seed(42)

    # 1. 生成所有形状
    shapes = generate_all_shapes(300)
    print(f"生成 {len(shapes)} 种形状, 每种 300 个点")

    # 2. 向量化方法对比
    print("\n" + "-" * 40)
    print("向量化方法对比 (H1)")
    print("-" * 40)

    vec_results = {}
    for shape_name, points in shapes.items():
        diagrams = compute_persistence_diagrams(points, maxdim=2)
        results = compare_vectorization_methods(diagrams, dim=1)
        vec_results[shape_name] = results
        print(f"\n{shape_name}:")
        for method, (t, vec) in results.items():
            print(f"  {method}: {t*1000:.2f}ms, dim={len(vec)}")

    # 3. 噪声鲁棒性分析
    print("\n" + "-" * 40)
    print("噪声鲁棒性分析")
    print("-" * 40)

    noise_levels = [0.01, 0.02, 0.05, 0.10]
    robustness_results = {}
    for shape_name in ['torus', 'sphere']:
        points = shapes[shape_name]
        results = test_noise_robustness(points, noise_levels, dim=1)
        robustness_results[shape_name] = results
        print(f"\n{shape_name} (W2 vs noise):")
        for n, w2 in zip(results['noise_level'], results['W2']):
            print(f"  σ={n:.3f}: W2={w2:.4f}")

    # 4. 可视化
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # 4a. 形状点云
    for idx, (shape_name, points) in enumerate(shapes.items()):
        ax = axes[0, idx // 2] if idx < 2 else axes[1, idx % 2]
        if idx < 4:
            ax_3d = fig.add_subplot(2, 3, idx + 1, projection='3d')
            ax_3d.scatter(points[:, 0], points[:, 1], points[:, 2],
                         s=2, c='steelblue', alpha=0.5)
            ax_3d.set_title(shape_name)
            ax_3d.set_xlabel("X"); ax_3d.set_ylabel("Y"); ax_3d.set_zlabel("Z")

    # 4b. 噪声鲁棒性
    ax_noise = axes[1, 0]
    for shape_name, results in robustness_results.items():
        ax_noise.plot(results['noise_level'], results['W2'],
                     'o-', label=f'{shape_name} H1', markersize=5)
    ax_noise.set_xlabel("Noise level σ"); ax_noise.set_ylabel("W2 distance")
    ax_noise.set_title("Topological Stability (W2 vs Noise)")
    ax_noise.legend()

    # 4c. 向量化方法时间对比
    ax_time = axes[1, 1]
    methods = list(vec_results['torus'].keys())
    times = [vec_results['torus'][m][0] * 1000 for m in methods]
    ax_time.bar(methods, times, color=['steelblue', 'coral', 'green', 'orange'])
    ax_time.set_ylabel("Time (ms)"); ax_time.set_title("Vectorization Time (Torus)")
    for i, v in enumerate(times):
        ax_time.text(i, v + 0.1, f'{v:.1f}', ha='center', fontsize=8)

    # 4d. 维度对比
    ax_dim = axes[1, 2]
    for shape_name, results in vec_results.items():
        dims = [len(results[m][1]) for m in methods]
        ax_dim.plot(methods, dims, 'o-', label=shape_name, markersize=5)
    ax_dim.set_ylabel("Feature dimension"); ax_dim.set_title("Feature Dimensionality")
    ax_dim.legend(fontsize=7)
    ax_dim.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    fname = RESULTS_DIR / "paper4_3dphdl_design_space.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[Paper 4] 已保存: {fname}")

    print("\n✅ Paper 4 Pipeline 完成")


if __name__ == "__main__":
    main()