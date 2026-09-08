#!/usr/bin/env python3
"""
子任务 B：拓扑特征 vs 传统几何特征对比实验
- 5 种合成对象：circle, disk, torus, sphere, porous_block
- 拓扑特征：WD/BD/top-k/持久点统计（来自 ph_pipeline）
- 几何特征：FPFH + 法线直方图 + 凸包统计 + PCA 主轴
- 分类器：SVM / RandomForest
- 鲁棒性：噪声（0%, 5%, 10%, 20%）、旋转（0°, 45°, 90°, 180°）
"""
import os
import sys
import time
import warnings
import numpy as np
from pathlib import Path
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# 将 src/ 加入路径
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ph_pipeline import (
    generate_circle, generate_disk, generate_torus,
    generate_sphere, generate_porous_block,
    compute_persistence_diagrams,
    wasserstein_distance, bottleneck_distance,
    compute_top_k_persistences,
)
import open3d as o3d

warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# 1. 几何特征提取
# ============================================================
def geometric_features_open3d(points: np.ndarray) -> np.ndarray:
    """用 Open3D 提取几何特征：FPFH(33) + 法线直方图(10) + 凸包统计(4)"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
    normals = np.asarray(pcd.normals)

    # 法线直方图（xyz 各 10 bin）
    hist_normals = []
    for axis in range(3):
        h, _ = np.histogram(normals[:, axis], bins=10, range=(-1.0, 1.0), density=True)
        hist_normals.append(h)
    normals_feat = np.concatenate(hist_normals)  # 30

    # FPFH（简化：只算 33 维）
    try:
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd, o3d.geometry.KDTreeSearchParamKNN(knn=20)
        )
        fpfh_arr = np.asarray(fpfh)  # (33, N) 或空
        if fpfh_arr.size == 0 or fpfh_arr.ndim != 2 or fpfh_arr.shape[1] < 1:
            fpfh_mean = np.zeros(33)
        else:
            fpfh_data = fpfh_arr.T  # (N, 33)
            fpfh_mean = np.mean(fpfh_data, axis=0)
            if np.any(np.isnan(fpfh_mean)):
                fpfh_mean = np.zeros(33)
    except Exception:
        fpfh_mean = np.zeros(33)

    # 凸包统计
    try:
        hull = o3d.geometry.PointCloud(pcd).compute_convex_hull()[0]
        hull_pts = np.asarray(hull.vertices)
        ch_vol = hull.volume if hasattr(hull, "volume") else 0.0
        ch_area = hull.area if hasattr(hull, "area") else 0.0
        ch_ratio = ch_area / (ch_vol + 1e-9)
    except Exception:
        ch_vol, ch_area, ch_ratio = 0.0, 0.0, 0.0
    convex_feat = np.array([ch_vol, ch_area, ch_ratio, len(points)])

    return np.concatenate([fpfh_mean, normals_feat, convex_feat])


def geometric_features_numpy(points: np.ndarray) -> np.ndarray:
    """纯 numpy/scipy 几何特征（open3d 不可用时回退）：主轴 + 凸包统计"""
    # PCA 主轴
    centered = points - points.mean(axis=0)
    cov = centered.T @ centered / max(len(points) - 1, 1)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(eigvals)[::-1]
    if eigvals[2] > 1e-9:
        aniso = (eigvals[0] - eigvals[2]) / eigvals[0]
        planarity = (eigvals[1] - eigvals[2]) / eigvals[0]
        sphericity = (eigvals[2] / eigvals[0]) ** 0.5
    else:
        aniso = planarity = sphericity = 0.0
    pca_feat = np.array([*eigvals, aniso, planarity, sphericity])  # 6

    # 凸包
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points)
        ch_vol = hull.volume
        ch_area = hull.area
        ch_ratio = ch_area / (ch_vol + 1e-9)
    except Exception:
        ch_vol, ch_area, ch_ratio = 0.0, 0.0, 0.0
    convex_feat = np.array([ch_vol, ch_area, ch_ratio, points.shape[0]])  # 4

    # 点云范围
    spread = points.max(axis=0) - points.min(axis=0)  # 3

    return np.concatenate([pca_feat, convex_feat, spread])


# ============================================================
# 2. 拓扑特征提取
# ============================================================
# 基准 PD 缓存（只算一次，避免每个样本重复计算）
_ref_dgms_cache = None


def _get_ref_dgms(maxdim: int):
    """缓存基准（单位圆）持久图，只算一次"""
    global _ref_dgms_cache
    if _ref_dgms_cache is None:
        ref = generate_circle(200)
        ref_diagrams = compute_persistence_diagrams(ref, maxdim=maxdim)
        _ref_dgms_cache = [_get_dgm(ref_diagrams, d) for d in range(maxdim + 1)]
    return _ref_dgms_cache


def _get_dgm(diagrams, dim):
    dgm = diagrams[dim] if dim < len(diagrams) else np.empty((0, 2))
    finite = dgm[~np.isinf(dgm[:, 1])]
    return finite if len(finite) > 0 else np.array([[0.0, 0.0]])


def topological_features(points: np.ndarray, maxdim: int = 2) -> np.ndarray:
    """拓扑特征：WD/BD（与单位圆基准比较）+ top-k + 持久点统计"""
    diagrams = compute_persistence_diagrams(points, maxdim=maxdim)

    ref_dgms = _get_ref_dgms(maxdim)
    dgms = [_get_dgm(diagrams, d) for d in range(maxdim + 1)]

    wd_feats = []
    bd_feats = []
    for d in range(maxdim + 1):
        try:
            wd_feats.append(wasserstein_distance(dgms[d], ref_dgms[d]))
        except Exception:
            wd_feats.append(0.0)
        try:
            bd_feats.append(bottleneck_distance(dgms[d], ref_dgms[d]))
        except Exception:
            bd_feats.append(0.0)

    # top-k 持久性（H1）
    try:
        topk = compute_top_k_persistences(diagrams, dim=1, top_k=10)
    except Exception:
        topk = np.zeros(10)

    # 持久点统计（每维度点数和平均持久度）
    stats = []
    for d in range(maxdim + 1):
        dgm = dgms[d]
        pers = dgm[:, 1] - dgm[:, 0]
        stats.extend([len(dgm), np.mean(pers) if len(pers) > 0 else 0.0, np.max(pers) if len(pers) > 0 else 0.0])

    return np.concatenate([wd_feats, bd_feats, topk, stats])


# ============================================================
# 3. 实验设置
# ============================================================
N_POINTS = 300
N_REPEAT = 3
OBJECTS = {
    "circle": lambda n: generate_circle(n),
    "disk": lambda n: generate_disk(n),
    "torus": lambda n: generate_torus(n),
    "sphere": lambda n: generate_sphere(n),
    "porous_block": lambda n: generate_porous_block(n),
}
NOISE_LEVELS = [0.0, 0.05, 0.10, 0.20]
ROTATIONS = [0, 45, 90, 180]
CLASSIFIERS = {
    "SVM": lambda: SVC(kernel="rbf", C=10, gamma="scale"),
    "RF": lambda: RandomForestClassifier(n_estimators=200, random_state=42),
}


def add_noise(points: np.ndarray, level: float) -> np.ndarray:
    if level == 0.0:
        return points
    scale = level * (points.max(axis=0) - points.min(axis=0)).mean()
    return points + np.random.normal(0, scale, size=points.shape)


def rotate_points(points: np.ndarray, degree: float) -> np.ndarray:
    if degree == 0:
        return points
    theta = np.radians(degree)
    # 绕 z 轴旋转（加上随机轴混合，使旋转更有效）
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1],
    ])
    return points @ R.T


# ============================================================
# 4. 主实验
# ============================================================
def main():
    sys.stdout.reconfigure(line_buffering=True)  # 实时刷新增量到日志
    results_dir = Path(__file__).resolve().parent.parent / "results" / "exp1b_topo_vs_geo"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_topo = []
    all_geo = []
    all_labels = []
    meta = []  # (obj, noise, rot, rep)

    print("=" * 70)
    print("子任务 B：拓扑特征 vs 几何特征 对比实验")
    print("=" * 70)

    for obj_name, gen_fn in OBJECTS.items():
        base_pts = gen_fn(N_POINTS)
        print(f"\n[对象] {obj_name} ({base_pts.shape[0]} 点)")
        for noise in NOISE_LEVELS:
            for rot in ROTATIONS:
                for rep in range(N_REPEAT):
                    pts = rotate_points(add_noise(base_pts.copy(), noise), rot)
                    try:
                        topo_f = topological_features(pts, maxdim=2)
                    except Exception as e:
                        print(f"  [跳过] {obj_name} n{noise} r{rot} rep{rep}: {e}")
                        continue
                    try:
                        geo_f = geometric_features_open3d(pts)
                    except Exception as e:
                        print(f"  [回退numpy] {obj_name}: {e}")
                        geo_f = geometric_features_numpy(pts)
                    all_topo.append(topo_f)
                    all_geo.append(geo_f)
                    all_labels.append(obj_name)
                    meta.append((obj_name, noise, rot, rep))
        print(f"  [进度] {obj_name} 完成")
        sys.stdout.flush()

    X_topo = np.array(all_topo)
    X_geo = np.array(all_geo)
    y = np.array(all_labels)

    print(f"\n数据集：{len(y)} 样本，{len(OBJECTS)} 类")
    print(f"  拓扑特征维度：{X_topo.shape[1]}")
    print(f"  几何特征维度：{X_geo.shape[1]}")

    # 保存特征矩阵
    np.save(results_dir / "X_topo.npy", X_topo)
    np.save(results_dir / "X_geo.npy", X_geo)
    np.save(results_dir / "labels.npy", y)

    # ============================================================
    # 5. 分类准确率对比
    # ============================================================
    summary_rows = []

    for feat_name, X in [("拓扑特征", X_topo), ("几何特征", X_geo)]:
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)

        for clf_name, clf_fn in CLASSIFIERS.items():
            accs = []
            for seed in range(5):
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X_s, y, test_size=0.3, random_state=seed, stratify=y
                )
                clf = clf_fn()
                clf.fit(X_tr, y_tr)
                accs.append(accuracy_score(y_te, clf.predict(X_te)))

            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            row = {
                "特征类型": feat_name,
                "分类器": clf_name,
                "均值准确率": f"{mean_acc:.4f}",
                "标准差": f"{std_acc:.4f}",
                "特征维度": X.shape[1],
            }
            summary_rows.append(row)
            print(f"  {feat_name:8s} + {clf_name:12s}: {mean_acc:.4f} ± {std_acc:.4f}")

    # 保存汇总
    import csv
    csv_path = results_dir / "topo_vs_geo_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\n[已保存] {csv_path}")

    # ============================================================
    # 6. 可视化
    # ============================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    topo_svm = [r for r in summary_rows if r["特征类型"] == "拓扑特征" and r["分类器"] == "SVM"][0]
    topo_rf = [r for r in summary_rows if r["特征类型"] == "拓扑特征" and r["分类器"] == "RF"][0]
    geo_svm = [r for r in summary_rows if r["特征类型"] == "几何特征" and r["分类器"] == "SVM"][0]
    geo_rf = [r for r in summary_rows if r["特征类型"] == "几何特征" and r["分类器"] == "RF"][0]

    cats = ["Topo\nSVM", "Topo\nRF", "Geo\nSVM", "Geo\nRF"]
    vals = [float(topo_svm["均值准确率"]), float(topo_rf["均值准确率"]),
            float(geo_svm["均值准确率"]), float(geo_rf["均值准确率"])]
    stds = [float(topo_svm["标准差"]), float(topo_rf["标准差"]),
            float(geo_svm["标准差"]), float(geo_rf["标准差"])]
    colors = ["#2196F3", "#64B5F6", "#F44336", "#EF9A9A"]

    axes[0].bar(cats, vals, yerr=stds, capsize=6, color=colors, edgecolor="black")
    axes[0].set_ylim(0, 1.1)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Topological vs Geometric Features: Classification Accuracy")
    for i, (v, s) in enumerate(zip(vals, stds)):
        axes[0].text(i, v + s + 0.02, f"{v:.2%}", ha="center", fontsize=10)

    # 噪声鲁棒性（SVM）
    noise_acc_topo = []
    noise_acc_geo = []
    noise_labels_plot = []
    for noise in NOISE_LEVELS:
        mask = [m[1] == noise for m in meta]
        if sum(mask) == 0:
            continue
        noise_labels_plot.append(f"{noise*100:.0f}%")
        for feat_name, X in [("topo", X_topo), ("geo", X_geo)]:
            X_sub = X[mask]
            y_sub = y[mask]
            scaler = StandardScaler()
            X_s = scaler.fit_transform(X_sub)
            accs = []
            for seed in range(5):
                X_tr, X_te, y_tr, y_te = train_test_split(X_s, y_sub, test_size=0.3, random_state=seed, stratify=y_sub)
                clf = SVC(kernel="rbf", C=10, gamma="scale")
                clf.fit(X_tr, y_tr)
                accs.append(accuracy_score(y_te, clf.predict(X_te)))
            if feat_name == "topo":
                noise_acc_topo.append(np.mean(accs))
            else:
                noise_acc_geo.append(np.mean(accs))

    axes[1].plot(noise_labels_plot, noise_acc_topo, "o-", color="#2196F3", label="Topological", linewidth=2)
    axes[1].plot(noise_labels_plot, noise_acc_geo, "s-", color="#F44336", label="Geometric", linewidth=2)
    axes[1].set_ylim(0, 1.1)
    axes[1].set_xlabel("Noise Level")
    axes[1].set_ylabel("Accuracy (SVM)")
    axes[1].set_title("Noise Robustness")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 旋转鲁棒性（SVM）
    rot_acc_topo = []
    rot_acc_geo = []
    rot_labels_plot = []
    for rot in ROTATIONS:
        mask = [m[2] == rot for m in meta]
        if sum(mask) == 0:
            continue
        rot_labels_plot.append(f"{rot}°")
        for feat_name, X in [("topo", X_topo), ("geo", X_geo)]:
            X_sub = X[mask]
            y_sub = y[mask]
            scaler = StandardScaler()
            X_s = scaler.fit_transform(X_sub)
            accs = []
            for seed in range(5):
                X_tr, X_te, y_tr, y_te = train_test_split(X_s, y_sub, test_size=0.3, random_state=seed, stratify=y_sub)
                clf = SVC(kernel="rbf", C=10, gamma="scale")
                clf.fit(X_tr, y_tr)
                accs.append(accuracy_score(y_te, clf.predict(X_te)))
            if feat_name == "topo":
                rot_acc_topo.append(np.mean(accs))
            else:
                rot_acc_geo.append(np.mean(accs))

    axes[2].plot(rot_labels_plot, rot_acc_topo, "o-", color="#2196F3", label="Topological", linewidth=2)
    axes[2].plot(rot_labels_plot, rot_acc_geo, "s-", color="#F44336", label="Geometric", linewidth=2)
    axes[2].set_ylim(0, 1.1)
    axes[2].set_xlabel("Rotation Angle")
    axes[2].set_ylabel("Accuracy (SVM)")
    axes[2].set_title("Rotation Robustness")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # 特征维度对比
    axes[3].axis("off")
    info_text = (
        f"Feature Dimension Comparison\n"
        f"{'─' * 40}\n"
        f"Topological features : {X_topo.shape[1]} dims\n"
        f"  - WD/BD (3 dims)\n"
        f"  - Top-k persistences (10 dims)\n"
        f"  - Per-dim stats (9 dims)\n"
        f"\nGeometric features   : {X_geo.shape[1]} dims\n"
        f"  - FPFH mean (33 dims)\n"
        f"  - Normal histograms (30 dims)\n"
        f"  - Convex hull stats (4 dims)\n"
        f"\nClassification: SVM (RBF) + RF\n"
        f"Noise: 0% / 5% / 10% / 20%\n"
        f"Rotation: 0° / 45° / 90° / 180°\n"
        f"Samples per config: {N_REPEAT} reps"
    )
    axes[3].text(0.05, 0.95, info_text, transform=axes[3].transAxes,
                 fontsize=10, verticalalignment="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    fig_path = results_dir / "topo_vs_geo_comparison.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[已保存] {fig_path}")

    # ============================================================
    # 7. 结论输出
    # ============================================================
    print("\n" + "=" * 70)
    print("实验结论")
    print("=" * 70)
    topo_mean = np.mean([float(r["均值准确率"]) for r in summary_rows if r["特征类型"] == "拓扑特征"])
    geo_mean = np.mean([float(r["均值准确率"]) for r in summary_rows if r["特征类型"] == "几何特征"])
    print(f"  拓扑特征平均准确率：{topo_mean:.2%}")
    print(f"  几何特征平均准确率：{geo_mean:.2%}")
    if topo_mean > geo_mean:
        print("  ✓ 拓扑特征整体优于几何特征")
    else:
        print("  ✗ 几何特征整体优于拓扑特征（合成数据场景下可能因几何特征更直接）")


if __name__ == "__main__":
    main()
