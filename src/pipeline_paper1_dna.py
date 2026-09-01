"""
============================================================
pipeline_paper1_dna.py — Paper 1: DNA 结构 PH 分析
============================================================
基于 2505.06583v1:
  Persistent Homology: A Pedagogical Introduction with
  Biological Applications

Pipeline:
  1. 生成/加载点云 (模拟 DNA 结构的 3D 坐标)
  2. 计算 Vietoris-Rips 过滤的持久同调
  3. 可视化点云和持久图
  4. 计算 Betti 数

注意: 原论文使用真实的 AlphaFold 蛋白质结构数据。
此处生成模拟的螺旋结构点云作为替代演示。
============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path
import sys

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'DejaVu Serif']
rcParams['axes.unicode_minus'] = False

sys.path.insert(0, str(Path(__file__).parent))
from ph_pipeline import (
    compute_persistence_diagrams,
    compute_top_k_persistences,
    compute_persistence_image,
    compute_persistence_landscape,
    compute_betti_curve,
)

RESULTS_DIR = Path(__file__).parent.parent / "results"


def generate_helix_3d(
    n_points: int = 200,
    height: float = 10.0,
    radius: float = 1.5,
    turns: float = 3.0,
    noise: float = 0.1
) -> np.ndarray:
    """生成模拟 DNA 双螺旋结构的 3D 点云。

    对应 Paper 1 第 5 节的 3-1 超螺旋 DNA 结构分析。
    """
    t = np.linspace(0, height, n_points)
    # 双螺旋：两条链
    theta = 2 * np.pi * turns * t / height
    # 链 1
    x1 = radius * np.cos(theta) + np.random.normal(0, noise, n_points)
    y1 = radius * np.sin(theta) + np.random.normal(0, noise, n_points)
    z1 = t + np.random.normal(0, noise, n_points)
    # 链 2 (相位差 π)
    x2 = radius * np.cos(theta + np.pi) + np.random.normal(0, noise, n_points)
    y2 = radius * np.sin(theta + np.pi) + np.random.normal(0, noise, n_points)
    z2 = t + np.random.normal(0, noise, n_points)
    return np.column_stack((
        np.concatenate([x1, x2]),
        np.concatenate([y1, y2]),
        np.concatenate([z1, z2])
    ))


def plot_pointcloud_and_diagrams(points, diagrams, save_dir):
    """绘制点云和持久图（对应 Paper 1 Figure 11, 15）。"""
    fig = plt.figure(figsize=(16, 5))

    # 点云
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], s=2, c='steelblue', alpha=0.6)
    ax1.set_title("Point Cloud (Simulated Helix)")
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")

    # 持久图
    ax2 = fig.add_subplot(132)
    colors = {0: 'red', 1: 'blue', 2: 'green'}
    labels = {0: 'H0 (components)', 1: 'H1 (loops)', 2: 'H2 (voids)'}
    max_val = 0
    for dim in range(3):
        dgm = diagrams[dim]
        if len(dgm) == 0:
            continue
        finite = dgm[~np.isinf(dgm[:, 1])]
        if len(finite) > 0:
            max_val = max(max_val, finite.max())
            ax2.scatter(finite[:, 0], finite[:, 1],
                       c=colors[dim], label=labels[dim], s=20, alpha=0.7)
        inf = dgm[np.isinf(dgm[:, 1])]
        for p in inf:
            ax2.scatter(p[0], max_val * 1.1, c=colors[dim],
                       marker='^', s=50, label=f'{labels[dim]} (inf)')

    max_val = max(max_val, 1.0)
    ax2.plot([0, max_val], [0, max_val], 'k--', alpha=0.3)
    ax2.set_xlabel("Birth"); ax2.set_ylabel("Death")
    ax2.set_title("Persistence Diagram")
    ax2.legend(fontsize=8)
    ax2.set_aspect('equal')

    # 条形码
    ax3 = fig.add_subplot(133)
    y_offset = 0
    for dim in range(2):
        dgm = diagrams[dim]
        if len(dgm) == 0:
            continue
        finite = dgm[~np.isinf(dgm[:, 1])]
        for b, d in sorted(finite, key=lambda x: x[1] - x[0], reverse=True)[:20]:
            ax3.plot([b, d], [y_offset, y_offset],
                    color=colors[dim], linewidth=2, alpha=0.7)
            y_offset += 1
    ax3.set_xlabel("Filtration value"); ax3.set_ylabel("Feature index")
    ax3.set_title("Persistence Barcode")

    plt.tight_layout()
    fname = save_dir / "paper1_dna_ph_analysis.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Paper 1] 已保存: {fname}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("Paper 1: DNA 结构 PH 分析 Pipeline")
    print("=" * 60)

    # 1. 生成模拟螺旋点云
    np.random.seed(42)
    points = generate_helix_3d(n_points=200, height=10.0, radius=1.5,
                               turns=3.0, noise=0.1)
    print(f"点云: {points.shape[0]} 个点, {points.shape[1]} 维")

    # 2. 计算持久同调
    diagrams = compute_persistence_diagrams(points, maxdim=2)
    for dim, dgm in enumerate(diagrams):
        finite = dgm[~np.isinf(dgm[:, 1])]
        print(f"  H{dim}: {len(dgm)} 个特征 ({len(finite)} 个有限)")

    # 3. 计算 Betti 数
    betti = []
    for dim in range(3):
        dgm = diagrams[dim]
        finite = dgm[~np.isinf(dgm[:, 1])]
        # 在某个固定过滤值下的 Betti 数（近似）
        betti.append(len(finite))
    print(f"Betti 数 (近似): β0={betti[0]}, β1={betti[1]}, β2={betti[2]}")

    # 4. 可视化
    plot_pointcloud_and_diagrams(points, diagrams, RESULTS_DIR)

    # 5. 向量化特征
    top_k = compute_top_k_persistences(diagrams, dim=1, top_k=5)
    print(f"Top-5 H1 持久性: {top_k}")

    # 6. 持久图像
    pi = compute_persistence_image(diagrams, dim=1, resolution=40)
    print(f"持久图像 (H1) 形状: {pi.shape}")

    # 7. 持久景观
    pl = compute_persistence_landscape(diagrams, dim=1, num_landscapes=3, num_points=50)
    print(f"持久景观 (H1) 形状: {pl.shape}")

    print("\n✅ Paper 1 Pipeline 完成")


if __name__ == "__main__":
    main()