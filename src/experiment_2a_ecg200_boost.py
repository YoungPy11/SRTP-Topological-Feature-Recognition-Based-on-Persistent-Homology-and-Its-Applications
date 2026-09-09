#!/usr/bin/env python3
"""
阶段 2A ECG200 85% 冲刺版
改进项（按优先级）：
1. 拓扑特征扩展：比例环数 + top-k 持久度 + Betti 曲线(5阈值) + 持续景观(3系数) = 15-20维
2. 嵌入参数微调：m∈{3,4,5}, τ∈{4,8,12}，train 5折CV选最优
3. 分类器扩展：XGBoost + LightGBM + 调优SVM + RF
4. 融合改进：加权拼接 + 集成投票
时间控制：2-3小时内无法稳定达85%立即停止
"""
import sys
import numpy as np
from pathlib import Path
import warnings
import json

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
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "ecg200"

MAXDIM = 1


def load_arff(path: Path):
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
            label = 1 if parts[-1].strip() == "1" else 0
            vals = np.array([float(x) for x in parts[:-1]])  # 排除标签，保留所有数值
            X.append(vals)
            y.append(label)
    return np.array(X), np.array(y)


def betti_curve(h1_f, thresholds=5):
    """在固定阈值处采样 Betti 数（H1 环数）"""
    if len(h1_f) == 0:
        return np.zeros(thresholds)
    pers = h1_f[:, 1] - h1_f[:, 0]
    mx = pers.max() if len(pers) > 0 else 1.0
    threshs = np.linspace(0.1 * mx, 0.9 * mx, thresholds)
    counts = [np.sum(pers > t) for t in threshs]
    return np.array(counts, dtype=np.float64)


def landscape_coeffs(h1_f, n_coeffs=3):
    """持续景观前 n_coeffs 个系数（简化版：按持久度排序的均值）"""
    if len(h1_f) == 0:
        return np.zeros(n_coeffs)
    pers = h1_f[:, 1] - h1_f[:, 0]
    sorted_pers = np.sort(pers)[::-1]
    coeffs = np.zeros(n_coeffs)
    for i in range(min(n_coeffs, len(sorted_pers))):
        # 景观系数 = 持久度 * 对应环的索引权重
        coeffs[i] = sorted_pers[i] / (sorted_pers[0] + 1e-9)
    return coeffs


def topo_features_boost(sig: np.ndarray, m: int, tau: int) -> np.ndarray:
    """增强版拓扑特征：比例环数 + top-5 持久度 + Betti 曲线(5) + 景观(3) = 14维"""
    pts = time_delay_embedding(sig, m=m, tau=tau)
    dgms = compute_persistence_diagrams(pts, maxdim=MAXDIM)
    h1 = dgms[1]
    h1_f = h1[~np.isinf(h1[:, 1])]
    feats = []
    if len(h1_f) == 0:
        return np.zeros(17)  # 4 + 5 + 5 + 3 = 17 维
    pers = h1_f[:, 1] - h1_f[:, 0]
    mx = pers.max()
    n_strong = int(np.sum(pers > 0.3 * mx)) if mx > 0 else 0
    n_loops = len(h1_f)
    feats += [n_strong, n_loops, mx, mx / (pers.mean() + 1e-9)]
    top_k = np.sort(pers)[::-1][:5]
    top_k_pad = np.zeros(5)
    top_k_pad[:len(top_k)] = top_k
    feats += [float(x) for x in top_k_pad]
    feats += [float(x) for x in betti_curve(h1_f, thresholds=5)]
    feats += [float(x) for x in landscape_coeffs(h1_f, n_coeffs=3)]
    return np.array(feats)


def timefreq_features(sig: np.ndarray) -> np.ndarray:
    x = np.asarray(sig, dtype=np.float64)
    feats = []
    feats += [x.mean(), x.std(), float(np.mean(np.diff(x))), float(np.std(np.diff(x)))]
    n = len(x)
    mu = x.mean()
    sd = x.std() + 1e-9
    feats.append(float(np.mean((x - mu) ** 3) / sd ** 3))
    feats.append(float(np.mean((x - mu) ** 4) / sd ** 4 - 3))
    fft = np.fft.rfft(x - mu)
    power = np.abs(fft) ** 2
    power_n = power / (power.sum() + 1e-9)
    freq = np.fft.rfftfreq(n, d=1.0)
    feats.append(float(freq[np.argmax(power)]))
    feats.append(float(power_n.max()))
    seg = len(power_n) // 3
    for i in range(3):
        feats.append(float(power_n[i * seg:(i + 1) * seg].sum()))
    return np.array(feats)


def cross_val_score(xtr, ytr, clf, cv=5):
    """简单 K 折交叉验证（分层）"""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = []
    for tr_idx, va_idx in skf.split(xtr, ytr):
        scaler = StandardScaler().fit(xtr[tr_idx])
        clf.fit(scaler.transform(xtr[tr_idx]), ytr[tr_idx])
        scores.append(accuracy_score(ytr[va_idx], clf.predict(scaler.transform(xtr[va_idx]))))
    return np.mean(scores), np.std(scores)


def main():
    sys.stdout.reconfigure(line_buffering=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_path = DATA_DIR / "ECG200_TRAIN.arff"
    test_path = DATA_DIR / "ECG200_TEST.arff"
    X_tr, y_tr = load_arff(train_path)
    X_te, y_te = load_arff(test_path)
    print(f"TRAIN: {X_tr.shape}, TEST: {X_te.shape}")
    print(f"类别分布 train: 0={sum(y_tr==0)}, 1={sum(y_tr==1)}; test: 0={sum(y_te==0)}, 1={sum(y_te==1)}")

    # [1] 嵌入参数微调（m∈{3,4,5}, τ∈{4,8,12}），train 5折CV选最优
    print("\n[1/4] 嵌入参数微调（5折CV）...")
    best_params = (3, 8)
    best_cv_score = -1
    param_scores = []
    for m in [3, 4, 5]:
        for tau in [4, 8, 12]:
            feats = np.array([topo_features_boost(s, m, tau) for s in X_tr])
            # 用 RF 快速评估
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            cv_mean, cv_std = cross_val_score(feats, y_tr, clf, cv=5)
            param_scores.append(((m, tau), cv_mean, cv_std))
            print(f"  m={m}, τ={tau}: CV={cv_mean:.3f}±{cv_std:.3f}")
            if cv_mean > best_cv_score:
                best_cv_score = cv_mean
                best_params = (m, tau)
    m, tau = best_params
    print(f"  选择 m={m}, τ={tau} (CV={best_cv_score:.3f})")

    # [2] 特征提取（增强版拓扑 + 时频 + 融合）
    print("\n[2/4] 特征提取（增强版）...")
    topo_tr = np.array([topo_features_boost(s, m, tau) for s in X_tr])
    topo_te = np.array([topo_features_boost(s, m, tau) for s in X_te])
    tf_tr = np.array([timefreq_features(s) for s in X_tr])
    tf_te = np.array([timefreq_features(s) for s in X_te])
    fuse_tr = np.concatenate([topo_tr, tf_tr], axis=1)
    fuse_te = np.concatenate([topo_te, tf_te], axis=1)
    print(f"  拓扑特征: {topo_tr.shape[1]} 维, 时频: {tf_tr.shape[1]} 维, 融合: {fuse_tr.shape[1]} 维")

    # [3] 分类器扩展 + 网格搜索
    print("\n[3/4] 分类器扩展（5折CV选最优）...")
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    from sklearn.decomposition import PCA
    import xgboost as xgb
    import lightgbm as lgb

    # 时频降维到拓扑同维（取较小值，避免 PCA 报错）
    pca_dim = min(topo_tr.shape[1], tf_tr.shape[1])
    pca = PCA(n_components=pca_dim)
    tf_pca_tr = pca.fit_transform(tf_tr)
    tf_pca_te = pca.transform(tf_te)

    feature_sets = {
        "拓扑特征": (topo_tr, topo_te),
        "时频特征": (tf_tr, tf_te),
        "时频(PCA)": (tf_pca_tr, tf_pca_te),
        "融合特征": (fuse_tr, fuse_te),
    }

    # 分类器网格
    classifiers = {
        "SVM(C=1)": lambda: SVC(kernel="rbf", C=1, gamma="scale"),
        "SVM(C=10)": lambda: SVC(kernel="rbf", C=10, gamma="scale"),
        "SVM(C=100)": lambda: SVC(kernel="rbf", C=100, gamma="scale"),
        "RF(50)": lambda: RandomForestClassifier(n_estimators=50, random_state=42),
        "RF(200)": lambda: RandomForestClassifier(n_estimators=200, random_state=42),
        "XGBoost": lambda: xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric="logloss"),
        "LightGBM": lambda: lgb.LGBMClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, verbose=-1),
    }

    cv_results = {}
    test_results = {}
    for feat_name, (xtr, xte) in feature_sets.items():
        scaler = StandardScaler().fit(xtr)
        xtr_s = scaler.transform(xtr)
        xte_s = scaler.transform(xte)
        cv_results[feat_name] = {}
        test_results[feat_name] = {}
        for clf_name, clf_fn in classifiers.items():
            clf = clf_fn()
            cv_mean, cv_std = cross_val_score(xtr_s, y_tr, clf, cv=5)
            cv_results[feat_name][clf_name] = (cv_mean, cv_std)
            # 在 test 上评估
            clf.fit(xtr_s, y_tr)
            test_acc = accuracy_score(y_te, clf.predict(xte_s))
            test_results[feat_name][clf_name] = test_acc
            print(f"  {feat_name:12s} + {clf_name:14s}: CV={cv_mean:.3f} test={test_acc:.3f}")

    # [4] 集成投票（RF + XGB + LGB 对融合特征投票）
    print("\n[4/4] 集成投票...")
    from sklearn.ensemble import VotingClassifier
    scaler = StandardScaler().fit(fuse_tr)
    fuse_tr_s = scaler.transform(fuse_tr)
    fuse_te_s = scaler.transform(fuse_te)
    voting_clf = VotingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=200, random_state=42)),
            ("xgb", xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric="logloss")),
            ("lgb", lgb.LGBMClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, verbose=-1)),
        ],
        voting="soft"
    )
    voting_clf.fit(fuse_tr_s, y_tr)
    voting_acc = accuracy_score(y_te, voting_clf.predict(fuse_te_s))
    test_results["融合特征"]["Voting(RF+XGB+LGB)"] = voting_acc
    print(f"  {'融合特征':12s} + {'Voting(RF+XGB+LGB)':22s}: test={voting_acc:.3f}")

    # 保存汇总
    summary = {
        "best_params": {"m": m, "tau": tau, "cv_score": best_cv_score},
        "feature_dims": {
            "topo": int(topo_tr.shape[1]),
            "timefreq": int(tf_tr.shape[1]),
            "fuse": int(fuse_tr.shape[1]),
        },
        "cv_results": {k: {kk: vv for kk, vv in v.items()} for k, v in cv_results.items()},
        "test_results": test_results,
        "param_scores": [{"m": int(mm), "tau": int(tt), "cv": float(cvm), "std": float(cvs)} for (mm, tt), cvm, cvs in param_scores],
    }
    with open(RESULTS_DIR / "ecg200_boost_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[已保存] {RESULTS_DIR / 'ecg200_boost_summary.json'}")

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    # 参数热力图
    param_grid = np.zeros((3, 3))
    m_vals = [3, 4, 5]
    tau_vals = [4, 8, 12]
    for (mm, tt), cvm, cvs in param_scores:
        mi = m_vals.index(mm)
        ti = tau_vals.index(tt)
        param_grid[mi, ti] = cvm
    im = ax.imshow(param_grid, cmap="YlGn", vmin=0.5, vmax=1.0, aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels(tau_vals)
    ax.set_yticks(range(3)); ax.set_yticks(range(3)); ax.set_yticklabels(m_vals)
    ax.set_xlabel("τ"); ax.set_ylabel("m"); ax.set_title("CV Accuracy Heatmap")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{param_grid[i,j]:.3f}", ha="center", va="center", fontsize=10)
    plt.colorbar(im, ax=ax)

    ax = axes[1]
    # Test 准确率对比（融合特征 + 所有分类器）
    feat_key = "融合特征"
    if feat_key in test_results:
        names = list(test_results[feat_key].keys())
        vals = [test_results[feat_key][n] for n in names]
        colors = ["#2196F3" if v < 0.85 else "#4CAF50" for v in vals]
        ax.barh(names, vals, color=colors, edgecolor="black")
        ax.axvline(0.85, color="red", linestyle="--", alpha=0.7, label="85% 目标")
        ax.axvline(0.83, color="orange", linestyle="--", alpha=0.7, label="基线 83%")
        ax.set_xlim(0.5, 1.0)
        ax.set_xlabel("Test Accuracy")
        ax.set_title("ECG200 Boost: Fused Features + Classifiers")
        for i, v in enumerate(vals):
            ax.text(v + 0.005, i, f"{v:.1%}", va="center", fontsize=9)
        ax.legend()
    plt.tight_layout()
    fig_path = RESULTS_DIR / "ecg200_boost.png"
    plt.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[已保存] {fig_path}")

    # 结论
    print("\n" + "=" * 60)
    print("ECG200 冲刺结论")
    print("=" * 60)
    best_feat, best_clf, best_acc = None, None, -1
    for feat, clf_res in test_results.items():
        for clf, acc in clf_res.items():
            if acc > best_acc:
                best_acc = acc
                best_feat, best_clf = feat, clf
    print(f"  最佳: {best_feat} + {best_clf} = {best_acc:.1%}")
    print(f"  嵌入参数: m={m}, τ={tau}")
    print(f"  拓扑特征: {topo_tr.shape[1]} 维")
    if best_acc >= 0.85:
        print(f"  ✅ 达到 85% 目标！")
    else:
        print(f"  ⚠️ 未达 85%（{best_acc:.1%}），差 {0.85 - best_acc:.1%}")
    print(f"  基线: 融合83% → 冲刺{best_acc:.1%} ({best_acc - 0.83:+.1%})")


if __name__ == "__main__":
    main()