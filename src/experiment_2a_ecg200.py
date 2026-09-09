#!/usr/bin/env python3
"""
阶段 2A：ECG200 真实时序二分类实证
- 数据：UCR ECG200（100 train / 100 test，96 长度心拍序列，正常 vs 心肌梗死）
- 特征：拓扑(延迟嵌入+比例判据+top-k) / 时频统计 / 融合
- 分类：SVM + RF，固定 train/test 官方划分
- 目标：验证拓扑特征在真实时序上的补充价值
"""
import sys
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment_2a_embedding import time_delay_embedding
from ph_pipeline import compute_persistence_diagrams

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "exp2a_ecg200"
DATA_DIR = Path("/tmp/ecg200")  # 数据下载位置
# 也支持从仓库 data 目录找
ALT_DATA = Path(__file__).resolve().parent.parent / "data" / "ecg200"

# 嵌入参数（合成验证确认的默认）
M_FIXED = 4
TAU_FIXED = 4
MAXDIM = 1


def load_arff(path: Path):
    """解析 UCR ARFF 文件。返回 (X, y)，X: (n, len), y: 0/1"""
    with open(path) as f:
        lines = f.readlines()
    X, y = [], []
    in_data = False
    for line in lines:
        line = line.strip()
        if not line or line.startswith("%") or line.startswith("@"):
            if line.lower().startswith("@data"):
                in_data = True
            continue
        if in_data:
            parts = line.split(",")
            if len(parts) < 2:
                continue
            # UCR ARFF 格式：标签在行尾，值为 -1 / 1
            label_str = parts[-1].strip()
            label = 1 if label_str == "1" else 0  # -1 -> 0 (正常), 1 -> 1 (心梗)
            vals = np.array([float(x) for x in parts[1:-1]])  # 排除标签
            X.append(vals)
            y.append(label)
    return np.array(X), np.array(y)


def topo_features(sig: np.ndarray, m: int, tau: int) -> np.ndarray:
    """从 H1 持续图提取拓扑特征：比例判据环数 + top-k 持久度 + 环数统计"""
    pts = time_delay_embedding(sig, m=m, tau=tau)
    dgms = compute_persistence_diagrams(pts, maxdim=MAXDIM)
    h1 = dgms[1]
    h1_f = h1[~np.isinf(h1[:, 1])]
    if len(h1_f) == 0:
        return np.zeros(8)
    pers = h1_f[:, 1] - h1_f[:, 0]
    mx = pers.max()
    # 比例判据环数
    n_strong = int(np.sum(pers > 0.3 * mx)) if mx > 0 else 0
    # top-k 持久度
    top_k = np.sort(pers)[::-1][:5]
    top_k_pad = np.zeros(5)
    top_k_pad[:len(top_k)] = top_k
    # 环数统计
    n_loops = len(h1_f)
    return np.concatenate([[n_strong, n_loops, mx, mx / (pers.mean() + 1e-9)], top_k_pad])


def timefreq_features(sig: np.ndarray) -> np.ndarray:
    """时频统计特征：均值/方差/偏度/峰度 + FFT 峰值/能量分布"""
    x = np.asarray(sig, dtype=np.float64)
    feats = []
    # 时域
    feats += [x.mean(), x.std(), float(np.mean(np.diff(x))), float(np.std(np.diff(x)))]
    # 偏度峰度
    n = len(x)
    mu = x.mean()
    sd = x.std() + 1e-9
    feats.append(float(np.mean((x - mu) ** 3) / sd ** 3))  # 偏度
    feats.append(float(np.mean((x - mu) ** 4) / sd ** 4 - 3))  # 峰度
    # 频域
    fft = np.fft.rfft(x - mu)
    power = np.abs(fft) ** 2
    power_n = power / (power.sum() + 1e-9)
    freq = np.fft.rfftfreq(n, d=1.0)
    # 主频位置 + 频域能量集中度
    feats.append(float(freq[np.argmax(power)]))
    feats.append(float(power_n.max()))
    # 频带能量（分 3 段）
    seg = len(power_n) // 3
    for i in range(3):
        feats.append(float(power_n[i * seg:(i + 1) * seg].sum()))
    return np.array(feats)


def main():
    sys.stdout.reconfigure(line_buffering=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 定位数据
    data_dir = DATA_DIR if DATA_DIR.exists() else ALT_DATA
    if not data_dir.exists():
        print(f"错误: 数据目录不存在 {data_dir}")
        sys.exit(1)
    train_path = data_dir / "ECG200_TRAIN.arff"
    test_path = data_dir / "ECG200_TEST.arff"
    print(f"数据: {data_dir}")

    X_tr, y_tr = load_arff(train_path)
    X_te, y_te = load_arff(test_path)
    print(f"TRAIN: {X_tr.shape}, TEST: {X_te.shape}")
    print(f"类别分布 train: 0={sum(y_tr==0)}, 1={sum(y_tr==1)}; test: 0={sum(y_te==0)}, 1={sum(y_te==1)}")

    # 嵌入参数微扫描（m∈{3,4,5}, τ∈{2,4,8}）选稳定性好的
    print("\n[1/3] 嵌入参数微扫描（选择稳定性）...")
    best_params = (M_FIXED, TAU_FIXED)
    best_sep = -1
    for m in [3, 4, 5]:
        for tau in [2, 4, 8]:
            # 用 train 集算两类拓扑特征均值差异
            feat_0 = [topo_features(s, m, tau) for s in X_tr[y_tr == 0][:10]]
            feat_1 = [topo_features(s, m, tau) for s in X_tr[y_tr == 1][:10]]
            m0 = np.mean(np.array(feat_0)[:, 0])  # 比例判据环数
            m1 = np.mean(np.array(feat_1)[:, 0])
            sep = abs(m1 - m0) / (np.std(np.array(feat_0)[:, 0]) + np.std(np.array(feat_1)[:, 0]) + 1e-9)
            print(f"  m={m}, tau={tau}: 类0环数={m0:.1f}, 类1环数={m1:.1f}, 区分度={sep:.2f}")
            if sep > best_sep:
                best_sep = sep
                best_params = (m, tau)
    m, tau = best_params
    print(f"  选择 m={m}, tau={tau} (区分度 {best_sep:.2f})")

    # 特征提取
    print("\n[2/3] 特征提取...")
    topo_tr = np.array([topo_features(s, m, tau) for s in X_tr])
    topo_te = np.array([topo_features(s, m, tau) for s in X_te])
    tf_tr = np.array([timefreq_features(s) for s in X_tr])
    tf_te = np.array([timefreq_features(s) for s in X_te])
    fuse_tr = np.concatenate([topo_tr, tf_tr], axis=1)
    fuse_te = np.concatenate([topo_te, tf_te], axis=1)
    print(f"  拓扑特征: {topo_tr.shape[1]} 维, 时频: {tf_tr.shape[1]} 维, 融合: {fuse_tr.shape[1]} 维")

    # 分类
    print("\n[3/3] 分类（固定 train/test）...")
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    from sklearn.decomposition import PCA

    feature_sets = {
        "拓扑特征": (topo_tr, topo_te),
        "时频特征": (tf_tr, tf_te),
        "融合特征": (fuse_tr, fuse_te),
    }
    # 时频降维到拓扑同维
    pca = PCA(n_components=topo_tr.shape[1])
    feature_sets["时频(PCA)"] = (pca.fit_transform(tf_tr), pca.transform(tf_te))

    classifiers = {
        "SVM": lambda: SVC(kernel="rbf", C=10, gamma="scale"),
        "RF": lambda: RandomForestClassifier(n_estimators=200, random_state=42),
    }

    results = {}
    for feat_name, (xtr, xte) in feature_sets.items():
        scaler = StandardScaler().fit(xtr)
        xtr_s = scaler.transform(xtr)
        xte_s = scaler.transform(xte)
        results[feat_name] = {}
        for clf_name, clf_fn in classifiers.items():
            clf = clf_fn()
            clf.fit(xtr_s, y_tr)
            acc = accuracy_score(y_te, clf.predict(xte_s))
            results[feat_name][clf_name] = acc
            print(f"  {feat_name:12s} + {clf_name:4s}: {acc:.4f}")

    # 保存汇总
    import csv
    csv_path = RESULTS_DIR / "ecg200_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["特征类型", "分类器", "准确率"])
        for feat_name, clf_res in results.items():
            for clf_name, acc in clf_res.items():
                w.writerow([feat_name, clf_name, f"{acc:.4f}"])
    print(f"\n[已保存] {csv_path}")

    # 可视化
    fig, ax = plt.subplots(figsize=(9, 5))
    names, vals = [], []
    for feat in ["拓扑特征", "时频特征", "时频(PCA)", "融合特征"]:
        for clf in ["SVM", "RF"]:
            names.append(f"{feat}\n{clf}")
            vals.append(results[feat][clf])
    colors = ["#2196F3", "#64B5F6", "#F44336", "#EF9A9A", "#FF9800", "#FFCC80", "#4CAF50", "#81C784"]
    ax.bar(names, vals, color=colors, edgecolor="black")
    ax.axhline(0.85, color="red", linestyle="--", alpha=0.6, label="85% 目标")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Accuracy")
    ax.set_title("ECG200: Topological vs Time-Frequency vs Fused")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.1%}", ha="center", fontsize=9)
    plt.xticks(rotation=30, ha="right")
    ax.legend()
    plt.tight_layout()
    fig_path = RESULTS_DIR / "ecg200_comparison.png"
    plt.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[已保存] {fig_path}")

    # 结论
    print("\n" + "=" * 60)
    print("ECG200 结论")
    print("=" * 60)
    for feat in ["拓扑特征", "时频特征", "时频(PCA)", "融合特征"]:
        svm = results[feat]["SVM"]
        rf = results[feat]["RF"]
        print(f"  {feat:12s}: SVM {svm:.1%}  RF {rf:.1%}  avg {(svm+rf)/2:.1%}")
    topo_avg = (results["拓扑特征"]["SVM"] + results["拓扑特征"]["RF"]) / 2
    tf_avg = (results["时频特征"]["SVM"] + results["时频特征"]["RF"]) / 2
    fuse_avg = (results["融合特征"]["SVM"] + results["融合特征"]["RF"]) / 2
    print(f"\n  融合({fuse_avg:.1%}) - 时频({tf_avg:.1%}) = {fuse_avg - tf_avg:+.1%}")
    if fuse_avg > tf_avg:
        print(f"  ✓ 融合优于纯时频，拓扑特征提供补充价值")
    else:
        print(f"  ✗ 融合未超纯时频（拓扑增量有限）")
    if fuse_avg >= 0.85:
        print(f"  ✅ ECG200 达到 85% 目标")
    else:
        print(f"  ⚠️ 未达 85%（{fuse_avg:.1%}）")

    # 保存特征矩阵
    np.save(RESULTS_DIR / "X_topo.npy", topo_tr)
    np.save(RESULTS_DIR / "X_tf.npy", tf_tr)
    np.save(RESULTS_DIR / "X_fuse.npy", fuse_tr)
    np.save(RESULTS_DIR / "y_tr.npy", y_tr)
    print(f"\n[已保存] 特征矩阵到 {RESULTS_DIR}/")


if __name__ == "__main__":
    main()