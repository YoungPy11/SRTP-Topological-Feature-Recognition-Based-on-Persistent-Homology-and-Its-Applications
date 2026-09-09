#!/usr/bin/env python3
"""
阶段 2B 任务 2：旋转/扰动连续性对比
- 环面点云，SO(3) 随机旋转（角度 ∈ {5°, 15°, 30°, 45°}）
- 四种向量化：top-k / PI / PL / Betti
- 度量：相对变化率 = ||rotated - clean||₂ / ||clean||₂
- 重点：top-k 天然旋转不变 vs PI/PL 需要插值
"""
import sys
import numpy as np
from pathlib import Path
import warnings
import csv

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ph_pipeline import (
    generate_torus,
    compute_persistence_diagrams,
    compute_top_k_persistences,
    compute_persistence_image,
    compute_betti_curve,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "exp2b_vectorization"

# 向量化参数（与任务 1 一致）
TOP_K = 10
PI_RESOLUTION = 40
PL_NUM_LANDSCAPES = 5
PL_NUM_POINTS = 100
BETTI_NUM_BINS = 50
MAXDIM = 2

# 旋转角度（度）
ANGLES = [5, 15, 30, 45]
N_REPEAT = 10


def random_so3(angle_deg):
    """生成随机 SO(3) 旋转矩阵（绕随机轴旋转固定角度）"""
    angle = np.deg2rad(angle_deg)
    # 随机单位轴
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    # Rodrigues 旋转公式
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R


def compute_fixed_landscape(dgm, num_landscapes=5, num_points=100, b_min=0, b_max=1):
    """用固定范围计算持久景观"""
    if len(dgm) == 0:
        return np.zeros((num_landscapes, num_points))
    finite_mask = ~np.isinf(dgm[:, 1])
    dgm_f = dgm[finite_mask]
    if len(dgm_f) == 0:
        return np.zeros((num_landscapes, num_points))
    # 归一化到 [0, 1]
    dgm_norm = (dgm_f - b_min) / (b_max - b_min) if b_max > b_min else dgm_f
    pers_pairs = [(b, d) for b, d in dgm_norm if d > b]
    if not pers_pairs:
        return np.zeros((num_landscapes, num_points))
    t_vals = np.linspace(0, 1, num_points)
    landscapes = np.zeros((num_landscapes, num_points))
    for b, d in pers_pairs:
        mid = (b + d) / 2
        half_len = (d - b) / 2
        for j, t in enumerate(t_vals):
            val = max(0, half_len - abs(t - mid))
            if val > 0:
                inserted = False
                for k in range(num_landscapes):
                    if landscapes[k, j] < val:
                        if k < num_landscapes - 1:
                            landscapes[k+1:, j] = landscapes[k:-1, j]
                        landscapes[k, j] = val
                        inserted = True
                        break
                if not inserted and num_landscapes > 0:
                    if val > landscapes[-1, j]:
                        landscapes[-1, j] = val
    return landscapes


def compute_fixed_betti_curve(dgm, num_bins=50, b_min=0, b_max=1):
    """用固定范围计算 Betti 曲线"""
    if len(dgm) == 0:
        return np.zeros(num_bins)
    finite_mask = ~np.isinf(dgm[:, 1])
    dgm_f = dgm[finite_mask]
    if len(dgm_f) == 0:
        return np.zeros(num_bins)
    dgm_norm = (dgm_f - b_min) / (b_max - b_min) if b_max > b_min else dgm_f
    t_vals = np.linspace(0, 1, num_bins)
    betti = np.zeros(num_bins)
    for b, d in dgm_norm:
        if d > b:
            betti[(t_vals >= b) & (t_vals < d)] += 1
    return betti


def vectorize(dgm, method, b_min=0, b_max=1):
    """对持久图做四种向量化（固定范围）"""
    if method == "top_k":
        return compute_top_k_persistences([dgm], dim=0, top_k=TOP_K)
    elif method == "pi":
        img = compute_persistence_image(
            [dgm], dim=0, resolution=PI_RESOLUTION, sigma=0.001,
            birth_range=(b_min, b_max), death_range=(b_min, b_max)
        )
        return img.flatten()
    elif method == "pl":
        land = compute_fixed_landscape(
            dgm, num_landscapes=PL_NUM_LANDSCAPES, num_points=PL_NUM_POINTS,
            b_min=b_min, b_max=b_max
        )
        return land.flatten()
    elif method == "betti":
        return compute_fixed_betti_curve(dgm, num_bins=BETTI_NUM_BINS, b_min=b_min, b_max=b_max)
    else:
        raise ValueError(f"Unknown method: {method}")


def main():
    sys.stdout.reconfigure(line_buffering=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # [1] 计算无噪声环面的持久图，记录范围
    print("[1/4] 计算参考持久图（无噪声环面）...")
    np.random.seed(42)
    pc_clean = generate_torus(n_points=500, major_radius=2.0, minor_radius=1.0, noise=0.0)
    dgm_clean = compute_persistence_diagrams(pc_clean, maxdim=MAXDIM)[0]
    h1_f = dgm_clean[~np.isinf(dgm_clean[:, 1])]
    if len(h1_f) == 0:
        b_min, b_max = 0.0, 1.0
    else:
        b_min = h1_f[:, 0].min() - 0.1
        b_max = h1_f[:, 1].max() + 0.1
    print(f"  H0 birth 范围: [{b_min:.3f}, {b_max:.3f}]")
    print(f"  持久对数量: {len(h1_f)}")

    # 计算参考向量化
    ref_vecs = {}
    for method in ["top_k", "pi", "pl", "betti"]:
        ref_vecs[method] = vectorize(dgm_clean, method, b_min, b_max)
        print(f"  {method:10s}: {ref_vecs[method].shape} 维")

    # [2] 对不同旋转角度计算向量化连续性
    print(f"\n[2/4] 旋转连续性测试（{N_REPEAT} 次重复）...")
    methods = ["top_k", "pi", "pl", "betti"]
    results = {m: {a: [] for a in ANGLES} for m in methods}

    for angle in ANGLES:
        print(f"\n  旋转角度 = {angle}°")
        for rep in range(N_REPEAT):
            np.random.seed(42 + rep)
            pc_clean = generate_torus(n_points=500, major_radius=2.0, minor_radius=1.0, noise=0.0)
            R = random_so3(angle)
            pc_rotated = pc_clean @ R.T
            dgm_rotated = compute_persistence_diagrams(pc_rotated, maxdim=MAXDIM)[0]
            for method in methods:
                vec_rotated = vectorize(dgm_rotated, method, b_min, b_max)
                vec_clean = ref_vecs[method]
                rel_change = np.linalg.norm(vec_rotated - vec_clean) / (np.linalg.norm(vec_clean) + 1e-9)
                results[method][angle].append(rel_change)
        for method in methods:
            mean_change = np.mean(results[method][angle])
            std_change = np.std(results[method][angle])
            print(f"    {method:10s}: {mean_change:.4f} ± {std_change:.4f}")

    # [3] 保存结果
    print("\n[3/4] 保存结果...")
    csv_path = RESULTS_DIR / "vectorization_rotation.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "angle_deg", "mean_rel_change", "std_rel_change"])
        for method in methods:
            for angle in ANGLES:
                mean_change = np.mean(results[method][angle])
                std_change = np.std(results[method][angle])
                w.writerow([method, angle, f"{mean_change:.6f}", f"{std_change:.6f}"])
    print(f"  [已保存] {csv_path}")

    # [4] 可视化
    print("\n[4/4] 可视化...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    colors = {"top_k": "#2196F3", "pi": "#F44336", "pl": "#FF9800", "betti": "#4CAF50"}
    markers = {"top_k": "o", "pi": "s", "pl": "^", "betti": "D"}
    for method in methods:
        means = [np.mean(results[method][a]) for a in ANGLES]
        stds = [np.std(results[method][a]) for a in ANGLES]
        ax.errorbar(ANGLES, means, yerr=stds, label=method,
                    color=colors[method], marker=markers[method],
                    linewidth=2, markersize=8, capsize=4)
    ax.set_xlabel("Rotation Angle (deg)", fontsize=12)
    ax.set_ylabel("Relative Change Rate", fontsize=12)
    ax.set_title("Vectorization Continuity vs Rotation", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    method_names = {"top_k": "Top-K", "pi": "PI", "pl": "PL", "betti": "Betti"}
    changes_at_max_rot = {m: np.mean(results[m][45]) for m in methods}
    sorted_methods = sorted(methods, key=lambda m: changes_at_max_rot[m])
    names = [method_names[m] for m in sorted_methods]
    vals = [changes_at_max_rot[m] for m in sorted_methods]
    bars = ax.barh(names, vals, color=[colors[m] for m in sorted_methods], edgecolor="black")
    ax.set_xlabel("Relative Change Rate (45°)", fontsize=12)
    ax.set_title("Continuity Ranking at Max Rotation", fontsize=13)
    for i, (bar, v) in enumerate(zip(bars, vals)):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=10)
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    fig_path = RESULTS_DIR / "vectorization_rotation.png"
    plt.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  [已保存] {fig_path}")

    # 结论
    print("\n" + "=" * 60)
    print("旋转连续性结论")
    print("=" * 60)
    print(f"{'方法':<10} {'5°':<12} {'15°':<12} {'30°':<12} {'45°':<12}")
    for method in methods:
        row = [method_names[method]]
        for angle in ANGLES:
            mean_change = np.mean(results[method][angle])
            row.append(f"{mean_change:.4f}")
        print("  ".join(row))
    print()
    sorted_at_max = sorted(methods, key=lambda m: changes_at_max_rot[m])
    print(f"连续性排序（45°，从好到差）:")
    for i, m in enumerate(sorted_at_max, 1):
        print(f"  {i}. {method_names[m]}: {changes_at_max_rot[m]:.4f}")
    print()
    if changes_at_max_rot["top_k"] < changes_at_max_rot["pi"]:
        print("✓ top-k 比 PI 更连续（旋转不变，预期）")
    else:
        print("✗ top-k 不如 PI 连续（意外）")


if __name__ == "__main__":
    main()