#!/usr/bin/env python3
"""
子任务 C：ModelNet10 三维真实物体分类
- 数据：ModelNet10（10 类 CAD 模型，.off 格式）
- 预处理：trimesh 加载网格 → 表面均匀采样 512 点
- 特征：拓扑(25维) / 几何(67维) / 融合(拓扑+几何)
- 分类：SVM + RF + 轻量 MLP
- 防泄漏：GroupKFold（按物体 ID 分组，同一物体不跨训练/测试）
"""
import os
import sys
import time
import numpy as np
from pathlib import Path
import warnings
import argparse

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ph_pipeline import compute_persistence_diagrams, compute_top_k_persistences
from experiment_1b_topo_vs_geo import (
    topological_features, geometric_features_open3d, geometric_features_numpy,
    _get_ref_dgms,
)
import trimesh

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "modelnet10" / "ModelNet10"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "exp1c_modelnet10"

# ModelNet10 类别
CLASSES = ["bathtub", "bed", "chair", "desk", "dresser", "monitor", "night_stand", "sofa", "table", "toilet"]

N_POINTS = 256
CACHE_FILE = RESULTS_DIR / "modelnet10_sampled_cache.npz"


def load_and_sample(split: str, max_per_class: int = None, use_cache: bool = True):
    """加载 ModelNet10 并采样点云。返回 (points_list, labels, obj_ids)"""
    if use_cache and CACHE_FILE.exists():
        data = np.load(CACHE_FILE, allow_pickle=True)
        print(f"[缓存] 从 {CACHE_FILE} 加载 {len(data['points'])} 个采样点云")
        return data["points"], data["labels"], data["obj_ids"]

    points_list, labels, obj_ids = [], [], []
    for cls_idx, cls in enumerate(CLASSES):
        cls_dir = DATA_DIR / cls / split
        off_files = sorted(cls_dir.glob("*.off"))
        if max_per_class:
            off_files = off_files[:max_per_class]
        n_loaded = 0
        for i, f in enumerate(off_files):
            try:
                mesh = trimesh.load_mesh(f)
                if mesh.is_empty:
                    continue
                pts = mesh.sample(N_POINTS)
                if len(pts) < 100:  # 采样失败
                    continue
            except Exception as e:
                print(f"  [跳过] {f.name}: {e}")
                continue
            points_list.append(pts)
            labels.append(cls_idx)
            obj_ids.append(f"{cls}_{i}")  # 物体唯一 ID
            n_loaded += 1
        print(f"  {cls:<14} {split:>6}: {len(off_files)} 文件, 实际加载 {n_loaded} 个")

    points_arr = np.array(points_list, dtype=object)
    return points_arr, np.array(labels), np.array(obj_ids)


def extract_features(points, maxdim=2):
    """对一批点云提取特征。返回 (X_topo, X_geo)"""
    X_topo, X_geo = [], []
    t0 = time.time()
    n = len(points)
    for i, p in enumerate(points):
        # 拓扑特征
        try:
            xt = topological_features(p, maxdim=maxdim)
        except Exception as e:
            print(f"  [拓扑失败] sample {i}: {e}")
            xt = np.zeros(25)
        # 几何特征
        try:
            xg = geometric_features_open3d(p)
        except Exception:
            xg = geometric_features_numpy(p)
        X_topo.append(xt)
        X_geo.append(xg)
        if (i + 1) % 20 == 0:
            el = time.time() - t0
            print(f"  [特征] {i+1}/{n} 完成, 已用 {el:.1f}s, 预计 {el/(i+1)*n:.1f}s")
    return np.array(X_topo), np.array(X_geo)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-class", type=int, default=None, help="每类最多用多少物体(试点)")
    parser.add_argument("--no-cache", action="store_true", help="忽略缓存重新采样")
    parser.add_argument("--load-from-split", default="train", choices=["train", "test"])
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sys.stdout.reconfigure(line_buffering=True)

    print("=" * 70)
    print("子任务 C：ModelNet10 三维物体分类（拓扑 vs 几何 vs 融合）")
    print("=" * 70)
    print(f"数据目录: {DATA_DIR}")
    print(f"每类物体上限: {args.max_per_class or '全部'}")
    print(f"采样点数: {N_POINTS}")

    # 1. 加载 + 采样
    print("\n[1/4] 加载并采样点云...")
    points, labels, obj_ids = load_and_sample(
        args.load_from_split, max_per_class=args.max_per_class, use_cache=not args.no_cache
    )
    print(f"  共 {len(points)} 个点云，{len(set(labels))} 类")

    # 保存缓存
    if not CACHE_FILE.exists() or args.no_cache:
        print(f"  [保存] 采样缓存到 {CACHE_FILE}")
        np.savez_compressed(CACHE_FILE, points=points, labels=labels, obj_ids=obj_ids)

    # 2. 提取特征
    print("\n[2/4] 提取特征...")
    X_topo, X_geo = extract_features(points, maxdim=2)
    print(f"  拓扑特征: {X_topo.shape[1]} 维")
    print(f"  几何特征: {X_geo.shape[1]} 维")

    # 融合特征
    X_fuse = np.concatenate([X_topo, X_geo], axis=1)
    print(f"  融合特征: {X_fuse.shape[1]} 维")

    # 3. 分类（GroupKFold 防泄漏）
    print("\n[3/4] 分类（GroupKFold 按物体分组）...")
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.decomposition import PCA

    gkf = GroupKFold(n_splits=5)
    feature_sets = {
        "拓扑特征": X_topo,
        "几何特征": X_geo,
        "融合特征": X_fuse,
    }
    # 几何降维到同维（公平对比）
    pca_geo = PCA(n_components=X_topo.shape[1])
    feature_sets["几何特征(PCA25)"] = pca_geo.fit_transform(X_geo)

    classifiers = {
        "SVM": lambda: SVC(kernel="rbf", C=10, gamma="scale"),
        "RF": lambda: RandomForestClassifier(n_estimators=200, random_state=42),
    }

    results = {}
    for feat_name, X in feature_sets.items():
        results[feat_name] = {}
        scaler = StandardScaler().fit(X)
        X_s = scaler.transform(X)
        for clf_name, clf_fn in classifiers.items():
            accs = []
            for train_idx, test_idx in gkf.split(X_s, labels, groups=obj_ids):
                clf = clf_fn()
                clf.fit(X_s[train_idx], labels[train_idx])
                accs.append(accuracy_score(labels[test_idx], clf.predict(X_s[test_idx])))
            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            results[feat_name][clf_name] = (mean_acc, std_acc)
            print(f"  {feat_name:16s} + {clf_name:5s}: {mean_acc:.4f} ± {std_acc:.4f}")

    # 保存汇总
    import csv
    csv_path = RESULTS_DIR / "modelnet10_summary.csv"
    rows = []
    for feat_name, clf_res in results.items():
        for clf_name, (m, s) in clf_res.items():
            rows.append({"特征类型": feat_name, "分类器": clf_name,
                         "均值准确率": f"{m:.4f}", "标准差": f"{s:.4f}",
                         "特征维度": feature_sets[feat_name].shape[1]})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\n[已保存] {csv_path}")

    # 4. 可视化
    print("\n[4/4] 可视化...")
    fig, ax = plt.subplots(figsize=(10, 6))
    labels_bar = []
    vals_bar = []
    stds_bar = []
    colors_bar = []
    color_map = {"拓扑特征": "#2196F3", "几何特征": "#F44336",
                 "融合特征": "#4CAF50", "几何特征(PCA25)": "#FF9800"}
    for feat_name in ["拓扑特征", "几何特征", "几何特征(PCA25)", "融合特征"]:
        for clf_name in ["SVM", "RF"]:
            m, s = results[feat_name][clf_name]
            labels_bar.append(f"{feat_name}\n{clf_name}")
            vals_bar.append(m)
            stds_bar.append(s)
            colors_bar.append(color_map[feat_name])
    ax.bar(labels_bar, vals_bar, yerr=stds_bar, capsize=4, color=colors_bar, edgecolor="black")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Accuracy")
    ax.set_title("ModelNet10: Topological vs Geometric vs Fused (GroupKFold)")
    for i, (v, s) in enumerate(zip(vals_bar, stds_bar)):
        ax.text(i, v + s + 0.02, f"{v:.1%}", ha="center", fontsize=9)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig_path = RESULTS_DIR / "modelnet10_comparison.png"
    plt.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[已保存] {fig_path}")

    # 结论
    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    for feat in ["拓扑特征", "几何特征", "几何特征(PCA25)", "融合特征"]:
        svm_acc = results[feat]["SVM"][0]
        rf_acc = results[feat]["RF"][0]
        print(f"  {feat:16s}: SVM {svm_acc:.1%}  RF {rf_acc:.1%}  avg {(svm_acc+rf_acc)/2:.1%}")
    fuse_best = max(results["融合特征"]["SVM"][0], results["融合特征"]["RF"][0])
    topo_best = max(results["拓扑特征"]["SVM"][0], results["拓扑特征"]["RF"][0])
    print(f"\n  融合特征最佳: {fuse_best:.1%} vs 拓扑特征最佳: {topo_best:.1%}")
    if fuse_best > topo_best:
        print("  ✓ 融合特征优于纯拓扑，拓扑特征提供补充价值")
    print(f"  {'✅ ModelNet10 达到 85% 目标' if fuse_best >= 0.85 else '⚠️ 未达 85%，需进一步调优'}")

    # 保存特征矩阵
    np.save(RESULTS_DIR / "X_topo.npy", X_topo)
    np.save(RESULTS_DIR / "X_geo.npy", X_geo)
    np.save(RESULTS_DIR / "X_fuse.npy", X_fuse)
    np.save(RESULTS_DIR / "labels.npy", labels)
    np.save(RESULTS_DIR / "obj_ids.npy", obj_ids)
    print(f"\n[已保存] 特征矩阵到 {RESULTS_DIR}/")


if __name__ == "__main__":
    main()