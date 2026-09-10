#!/usr/bin/env python3
"""
阶段 2B 任务 3：ECG200 上的向量化效果对比
- 四种向量化：top-k(10D) / PI(1600D→PCA降维) / PL(500D→PCA降维) / Betti(50D)
- 分类器：RF（与 ECG200 冲刺版一致）
- 度量：test 准确率 + 5折 CV 稳定性（方差）
- 重点：哪种向量化在小数据集上最稳定/最准确
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
from experiment_2a_embedding import time_delay_embedding
from ph_pipeline import (
    compute_persistence_diagrams,
    compute_top_k_persistences,
    compute_persistence_image,
    compute_persistence_landscape,
    compute_betti_curve,
)
from experiment_2a_ecg200 import load_arff  # 复用 ARFF 加载

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "exp2b_vectorization"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "ecg200"

# 嵌入参数（用任务 1 最优参数）
M_FIXED = 3
TAU_FIXED = 8
MAXDIM = 1

# 向量化参数
TOP_K = 10
PI_RESOLUTION = 40
PL_NUM_LANDSCAPES = 5
PL_NUM_POINTS = 100
BETTI_NUM_BINS = 50
PCA_DIM = 50  # 高维向量化降维目标


def vectorize(dgm, method):
    """对持久图做四种向量化"""
    if method == "top_k":
        return compute_top_k_persistences([dgm], dim=0, top_k=TOP_K)
    if method == "pi":
        img = compute_persistence_image(
            [dgm], dim=0, resolution=PI_RESOLUTION, sigma=0.001,
            birth_range=(0, 1), death_range=(0, 1)
        )
        return img.flatten()
    elif method == "pl":
        land = compute_persistence_landscape(
            [dgm], dim=0, num_landscapes=PL_NUM_LANDSCAPES, num_points=PL_NUM_POINTS
        )
        return land.flatten()
    elif method == "betti":
        return compute_betti_curve([dgm], dim=0, num_bins=BETTI_NUM_BINS)
    else:
        raise ValueError(f"Unknown method: {method}")


def main():
    sys.stdout.reconfigure(line_buffering=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # [1] 加载 ECG200 数据
    print("[1/4] 加载 ECG200 数据...")
    train_path = DATA_DIR / "ECG200_TRAIN.arff"
    test_path = DATA_DIR / "ECG200_TEST.arff"
    X_tr, y_tr = load_arff(train_path)
    X_te, y_te = load_arff(test_path)
    print(f"  TRAIN: {X_tr.shape}, TEST: {X_te.shape}")
    print(f"  类别分布 train: 0={sum(y_tr==0)}, 1={sum(y_tr==1)}; test: 0={sum(y_te==0)}, 1={sum(y_te==1)}")

    # [2] 特征提取（四种向量化）
    print(f"\n[2/4] 特征提取（嵌入 m={M_FIXED}, τ={TAU_FIXED}）...")
    methods = ["top_k", "pi", "pl", "betti"]
    feat_tr = {}
    feat_te = {}
    for method in methods:
        vecs_tr = [vectorize(
            compute_persistence_diagrams(
                time_delay_embedding(s, m=M_FIXED, tau=TAU_FIXED),
                maxdim=MAXDIM
            )[0],
            method
        ) for s in X_tr]
        vecs_te = [vectorize(
            compute_persistence_diagrams(
                time_delay_embedding(s, m=M_FIXED, tau=TAU_FIXED),
                maxdim=MAXDIM
            )[0],
            method
        ) for s in X_te]
        feat_tr[method] = np.array(vecs_tr)
        feat_te[method] = np.array(vecs_te)
        print(f"  {method:10s}: train {feat_tr[method].shape}, test {feat_te[method].shape}")

    # [3] 分类（高维向量化用 PCA 降维）
    print(f"\n[3/4] 分类（RF，高维向量化 PCA→{PCA_DIM}D）...")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import StratifiedKFold

    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for method in methods:
        xtr = feat_tr[method]
        xte = feat_te[method]
        # 高维向量化降维
        if xtr.shape[1] > PCA_DIM:
            pca = PCA(n_components=PCA_DIM)
            xtr = pca.fit_transform(xtr)
            xte = pca.transform(xte)
        # 标准化
        scaler = StandardScaler().fit(xtr)
        xtr_s = scaler.transform(xtr)
        xte_s = scaler.transform(xte)
        # 5折 CV
        cv_scores = []
        for tr_idx, va_idx in cv.split(xtr_s, y_tr):
            clf.fit(xtr_s[tr_idx], y_tr[tr_idx])
            cv_scores.append(accuracy_score(y_tr[va_idx], clf.predict(xtr_s[va_idx])))
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        # Test
        clf.fit(xtr_s, y_tr)
        test_acc = accuracy_score(y_te, clf.predict(xte_s))
        results[method] = {
            "cv_mean": float(cv_mean),
            "cv_std": float(cv_std),
            "test_acc": float(test_acc),
            "dim": int(feat_tr[method].shape[1]),
        }
        print(f"  {method:10s}: CV={cv_mean:.3f}±{cv_std:.3f}, test={test_acc:.3f}")

    # [4] 保存 + 可视化
    print("\n[4/4] 保存结果...")
    csv_path = RESULTS_DIR / "vectorization_ecg200.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "dim", "cv_mean", "cv_std", "test_acc"])
        for method, res in results.items():
            w.writerow([method, res["dim"], f"{res['cv_mean']:.4f}", f"{res['cv_std']:.4f}", f"{res['test_acc']:.4f}"])
    print(f"  [已保存] {csv_path}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    method_names = {"top_k": "Top-K(10D)", "pi": "PI(1600D→PCA)", "pl": "PL(500D→PCA)", "betti": "Betti(50D)"}
    colors = {"top_k": "#2196F3", "pi": "#F44336", "pl": "#FF9800", "betti": "#4CAF50"}

    # 左图：CV 准确率 + 方差
    ax = axes[0]
    names = [method_names[m] for m in methods]
    cv_means = [results[m]["cv_mean"] for m in methods]
    cv_stds = [results[m]["cv_std"] for m in methods]
    bars = ax.bar(names, cv_means, yerr=cv_stds, capsize=4,
                  color=[colors[m] for m in methods], edgecolor="black")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("5-Fold CV Accuracy")
    ax.set_title("ECG200: Vectorization Comparison (CV)")
    for i, (bar, v) in enumerate(zip(bars, cv_means)):
        ax.text(i, v + 0.03, f"{v:.1%}", ha="center", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # 右图：Test 准确率
    ax = axes[1]
    test_accs = [results[m]["test_acc"] for m in methods]
    bars = ax.bar(names, test_accs, color=[colors[m] for m in methods], edgecolor="black")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Test Accuracy")
    ax.set_title("ECG200: Vectorization Comparison (Test)")
    for i, (bar, v) in enumerate(zip(bars, test_accs)):
        ax.text(i, v + 0.03, f"{v:.1%}", ha="center", fontsize=10)
    ax.axhline(0.85, color="red", linestyle="--", alpha=0.7, label="85% 目标")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig_path = RESULTS_DIR / "vectorization_ecg200.png"
    plt.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  [已保存] {fig_path}")

    # 结论
    print("\n" + "=" * 60)
    print("ECG200 向量化效果结论")
    print("=" * 60)
    best_method = max(methods, key=lambda m: results[m]["test_acc"])
    print(f"  最佳向量化: {method_names[best_method]} (test={results[best_method]['test_acc']:.1%})")
    print(f"  稳定性排序（CV 方差，从稳到差）:")
    sorted_by_stability = sorted(methods, key=lambda m: results[m]["cv_std"])
    for i, m in enumerate(sorted_by_stability, 1):
        print(f"    {i}. {method_names[m]}: CV={results[m]['cv_mean']:.1%}±{results[m]['cv_std']:.3f}")
    print(f"\n  与 ECG200 基线对比（融合 85%）:")
    for method in methods:
        print(f"    {method_names[m]}: {results[method]['test_acc']:.1%}")


if __name__ == "__main__":
    main()