#!/usr/bin/env python3
"""
阶段 2C-ECG5000：任务拓扑相关性维度的验证
==========================================
假设（2C 核心结论的另一半）：FordA 上可学习向量化未反超是因为其判别信息在时频域
（纯拓扑仅 50-62%）；ECG5000 是 5 类心律多分类，心拍形态差异大，拓扑相关性应更强，
预期：(a) 纯拓扑特征准确率明显高于 FordA；(b) 可学习向量化 ≥ 固定向量化。

数据：ECG5000（500 train / 4500 test，5 类，140 点心拍序列，官方固定划分）
协议：Takens m=5, τ=4，H1 持续图，train 5折CV 选 epoch，test 只评一次
对比矩阵（多分类版）：
  固定: Top-K(10D)+RF / Top-K+时频(21D)+RF
  可学习: PersLay纯拓扑 / PersLay+时频 MLP / PersLay特征→RF混合
"""
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from perslay_lite import diagrams_to_padded, TopoClassifier
from experiment_2a_ecg200 import timefreq_features
from experiment_2a_embedding import time_delay_embedding
from ph_pipeline import compute_persistence_diagrams

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "exp2c_ecg5000"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "ecg5000"

M_EMB, TAU_EMB = 5, 4
GRID, HIDDEN, DROPOUT = 20, 32, 0.3
LR_HEAD, LR_PERSLAY, WD = 1e-3, 1e-2, 1e-3
BATCH = 128
MAX_EPOCHS, PATIENCE = 80, 12
SEED = 42
TOP_K = 10
NUM_CLASSES = 5


def set_seed(s=SEED):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def load_arff_multi(path: Path):
    """解析 UCR ARFF 多分类文件：140 数值 + 行尾类别标签(1-5 → 0-4)"""
    X, y = [], []
    in_data = False
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            if line.startswith("@"):
                if line.lower().startswith("@data"):
                    in_data = True
                continue
            if in_data:
                parts = line.split(",")
                label = int(float(parts[-1])) - 1     # 1-5 → 0-4
                vals = [float(x) for x in parts[:-1]]  # 全部 140 个数值
                X.append(vals)
                y.append(label)
    return np.array(X), np.array(y)


def _ph_one(sig):
    pts = time_delay_embedding(np.asarray(sig), m=M_EMB, tau=TAU_EMB)
    return compute_persistence_diagrams(pts, maxdim=1)[1]


def compute_pd_cache(X, cache_path):
    if cache_path.exists():
        return list(np.load(cache_path, allow_pickle=True)["diagrams"])
    from concurrent.futures import ProcessPoolExecutor
    import os
    n_jobs = min(8, os.cpu_count() or 4)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        diagrams = list(ex.map(_ph_one, list(X), chunksize=32))
    print(f"  PH {len(X)} 样本用时 {time.time()-t0:.0f}s ({n_jobs} 进程)", flush=True)
    np.savez(cache_path, diagrams=np.array(diagrams, dtype=object))
    return diagrams


def topk_feats(diagrams, k=TOP_K):
    out = np.zeros((len(diagrams), k))
    for i, dgm in enumerate(diagrams):
        d = np.asarray(dgm)
        if len(d) == 0:
            continue
        d = d[~np.isinf(d[:, 1])]
        p = d[:, 1] - d[:, 0]
        p = np.sort(p[p > 0])[::-1][:k]
        out[i, :len(p)] = p
    return out


def pers_range_from(diagrams, q=99.0):
    births, perss = [], []
    for dgm in diagrams:
        d = np.asarray(dgm)
        if len(d) == 0:
            continue
        d = d[~np.isinf(d[:, 1])]
        d = d[d[:, 1] > d[:, 0]]
        if len(d) == 0:
            continue
        births.append(d[:, 0])
        perss.append(d[:, 1] - d[:, 0])
    rng = max(float(np.percentile(np.concatenate(births), q)),
              float(np.percentile(np.concatenate(perss), q))) * 1.1
    return (0.0, rng)


def train_fold_minibatch(model, pts, mask, tf, y, tr_idx, va_idx, device):
    """多分类 minibatch 训练 + 早停"""
    from sklearn.metrics import accuracy_score
    opt = torch.optim.Adam([
        {"params": list(model.perslay.parameters()), "lr": LR_PERSLAY},
        {"params": list(model.head.parameters()), "lr": LR_HEAD},
    ], weight_decay=WD)
    loss_fn = nn.CrossEntropyLoss()
    rng = np.random.RandomState(SEED)
    best_acc, best_ep, bad, best_state = -1.0, 0, 0, None
    n_tr = len(tr_idx)
    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = rng.permutation(n_tr)
        for s in range(0, n_tr, BATCH):
            bidx = tr_idx[perm[s:s + BATCH]]
            opt.zero_grad()
            logits = model(pts[bidx].to(device), mask[bidx].to(device),
                           tf[bidx].to(device) if tf is not None else None)
            loss = loss_fn(logits, y[bidx].to(device))
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            preds = []
            for s in range(0, len(va_idx), 512):
                bidx = va_idx[s:s + 512]
                vl = model(pts[bidx].to(device), mask[bidx].to(device),
                           tf[bidx].to(device) if tf is not None else None)
                preds.append(vl.argmax(1).cpu().numpy())
            va = accuracy_score(y[va_idx].numpy(), np.concatenate(preds))
        if va > best_acc:
            best_acc, best_ep, bad = va, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    return best_acc, best_ep, best_state


def cv_perslay(pts, mask, tf, y, device, pers_range, tag, n_splits=5):
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    tf_dim = tf.shape[1] if tf is not None else 0
    accs, eps = [], []
    y_np = y.numpy()
    for fi, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y_np)), y_np)):
        set_seed(SEED + fi)
        model = TopoClassifier(grid_size=GRID, tf_dim=tf_dim, hidden=HIDDEN,
                               dropout=DROPOUT, pers_range=pers_range,
                               num_classes=NUM_CLASSES).to(device)
        acc, ep, _ = train_fold_minibatch(model, pts, mask, tf, y, tr_idx, va_idx, device)
        accs.append(acc); eps.append(ep)
        print(f"    [{tag}] fold{fi+1}: val={acc:.3f} ep={ep}", flush=True)
    return accs, int(np.median(eps))


def final_train(pts_tr, mask_tr, tf_tr, y_tr, pts_te, mask_te, tf_te, y_te,
                device, pers_range, n_epochs):
    from sklearn.metrics import accuracy_score
    set_seed(SEED)
    tf_dim = tf_tr.shape[1] if tf_tr is not None else 0
    model = TopoClassifier(grid_size=GRID, tf_dim=tf_dim, hidden=HIDDEN,
                           dropout=DROPOUT, pers_range=pers_range,
                           num_classes=NUM_CLASSES).to(device)
    opt = torch.optim.Adam([
        {"params": list(model.perslay.parameters()), "lr": LR_PERSLAY},
        {"params": list(model.head.parameters()), "lr": LR_HEAD},
    ], weight_decay=WD)
    loss_fn = nn.CrossEntropyLoss()
    rng = np.random.RandomState(SEED)
    n = pts_tr.shape[0]
    for ep in range(n_epochs):
        model.train()
        perm = rng.permutation(n)
        for s in range(0, n, BATCH):
            bidx = perm[s:s + BATCH]
            opt.zero_grad()
            logits = model(pts_tr[bidx].to(device), mask_tr[bidx].to(device),
                           tf_tr[bidx].to(device) if tf_tr is not None else None)
            loss = loss_fn(logits, y_tr[bidx].to(device))
            loss.backward()
            opt.step()
    model.eval()
    preds = []
    with torch.no_grad():
        for s in range(0, pts_te.shape[0], 512):
            vl = model(pts_te[s:s+512].to(device), mask_te[s:s+512].to(device),
                       tf_te[s:s+512].to(device) if tf_te is not None else None)
            preds.append(vl.argmax(1).cpu().numpy())
    acc = accuracy_score(y_te.numpy(), np.concatenate(preds))
    learned = (float(model.perslay.a.detach().cpu()), float(model.perslay.b.detach().cpu()),
               float(np.exp(model.perslay.log_sigma.detach().cpu().item())))
    return acc, learned, model


def main():
    sys.stdout.reconfigure(line_buffering=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    # [1] 数据
    print("\n[1/6] 加载 ECG5000（5类多分类）...")
    X_tr, y_tr = load_arff_multi(DATA_DIR / "ECG5000_TRAIN.arff")
    X_te, y_te = load_arff_multi(DATA_DIR / "ECG5000_TEST.arff")
    print(f"  TRAIN {X_tr.shape}, 标签分布: {np.bincount(y_tr)}")
    print(f"  TEST  {X_te.shape}, 标签分布: {np.bincount(y_te)}")

    # [2] PH 缓存
    print("\n[2/6] PH 计算缓存 (原始140点, m=5, τ=4)...")
    dgm_tr = compute_pd_cache(X_tr, RESULTS_DIR / "pd_train.npz")
    dgm_te = compute_pd_cache(X_te, RESULTS_DIR / "pd_test.npz")
    print(f"  H1 持久点均值: train {np.mean([len(d) for d in dgm_tr]):.1f}, "
          f"test {np.mean([len(d) for d in dgm_te]):.1f}")

    pers_range = pers_range_from(dgm_tr + dgm_te)
    print(f"  网格范围(99%分位): {pers_range}")

    # [3] 固定向量化基线
    print("\n[3/6] 固定向量化基线 (Top-K + RF)...")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    tk_tr = topk_feats(dgm_tr)
    tk_te = topk_feats(dgm_te)
    tf_tr_np = np.array([timefreq_features(s) for s in X_tr])
    tf_te_np = np.array([timefreq_features(s) for s in X_te])

    rf = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)

    t0 = time.time()
    cv_topk = cross_val_score(rf, tk_tr, y_tr, cv=skf, scoring="accuracy", n_jobs=1)
    rf.fit(tk_tr, y_tr)
    test_topk = accuracy_score(y_te, rf.predict(tk_te))
    print(f"  Top-K(10D)+RF:      CV={cv_topk.mean():.3f}±{cv_topk.std():.3f} TEST={test_topk:.3f} ({time.time()-t0:.0f}s)")

    fuse_tr = np.concatenate([tk_tr, tf_tr_np], axis=1)
    fuse_te = np.concatenate([tk_te, tf_te_np], axis=1)
    cv_fuse = cross_val_score(rf, fuse_tr, y_tr, cv=skf, scoring="accuracy", n_jobs=1)
    rf.fit(fuse_tr, y_tr)
    test_fuse = accuracy_score(y_te, rf.predict(fuse_te))
    print(f"  Top-K+时频(21D)+RF: CV={cv_fuse.mean():.3f}±{cv_fuse.std():.3f} TEST={test_fuse:.3f}")

    # [4] 张量化
    print("\n[4/6] 张量化...")
    pts_tr, mask_tr = diagrams_to_padded(dgm_tr)
    pts_te, mask_te = diagrams_to_padded(dgm_te, max_points=pts_tr.shape[1])
    tf_tr_t = torch.tensor(tf_tr_np, dtype=torch.float32)
    tf_te_t = torch.tensor(tf_te_np, dtype=torch.float32)
    mu, sd = tf_tr_t.mean(0), tf_tr_t.std(0) + 1e-9
    tf_tr_n, tf_te_n = (tf_tr_t - mu) / sd, (tf_te_t - mu) / sd
    y_tr_t, y_te_t = torch.tensor(y_tr, dtype=torch.long), torch.tensor(y_te, dtype=torch.long)
    print(f"  pts_tr {tuple(pts_tr.shape)} (padding N={pts_tr.shape[1]})")

    results = {
        "dataset": "ECG5000", "task": "5-class multiclass",
        "train_n": int(X_tr.shape[0]), "test_n": int(X_te.shape[0]),
        "fixed_topk_rf": {"cv": float(cv_topk.mean()), "cv_std": float(cv_topk.std()), "test": float(test_topk)},
        "fixed_fuse_rf": {"cv": float(cv_fuse.mean()), "cv_std": float(cv_fuse.std()), "test": float(test_fuse)},
    }

    # [5] PersLay 纯拓扑
    print("\n[5/6] PersLay 纯拓扑 (5折CV)...")
    accs, ep = cv_perslay(pts_tr, mask_tr, None, y_tr_t, device, pers_range, "topo")
    cv_p = float(np.mean(accs))
    test_p, learn_p, model_p = final_train(pts_tr, mask_tr, None, y_tr_t,
                                           pts_te, mask_te, None, y_te_t, device, pers_range, ep)
    print(f"  CV={cv_p:.3f}±{np.std(accs):.3f} TEST={test_p:.3f} (ep={ep}) "
          f"a={learn_p[0]:.2f} b={learn_p[1]:.2f} σ={learn_p[2]:.3f}")
    results["perslay_topo"] = {"cv": cv_p, "cv_std": float(np.std(accs)), "test": float(test_p),
                               "epochs": ep, "learned": list(learn_p)}

    # [6] PersLay + 时频
    print("\n[6/6] PersLay + 时频融合 (5折CV)...")
    accs2, ep2 = cv_perslay(pts_tr, mask_tr, tf_tr_n, y_tr_t, device, pers_range, "fuse")
    cv_f = float(np.mean(accs2))
    test_f, learn_f, model_f = final_train(pts_tr, mask_tr, tf_tr_n, y_tr_t,
                                           pts_te, mask_te, tf_te_n, y_te_t, device, pers_range, ep2)
    print(f"  CV={cv_f:.3f}±{np.std(accs2):.3f} TEST={test_f:.3f} (ep={ep2}) "
          f"a={learn_f[0]:.2f} b={learn_f[1]:.2f} σ={learn_f[2]:.3f}")
    results["perslay_fused"] = {"cv": cv_f, "cv_std": float(np.std(accs2)), "test": float(test_f),
                                "epochs": ep2, "learned": list(learn_f)}

    # [6b] 混合：PersLay特征 → RF
    print("\n[6b] PersLay特征(400D)+时频 → RF 混合...")
    from sklearn.preprocessing import StandardScaler
    model_f.eval()
    with torch.no_grad():
        feat_tr, feat_te = [], []
        for s in range(0, pts_tr.shape[0], 512):
            feat_tr.append(model_f.perslay(pts_tr[s:s+512].to(device), mask_tr[s:s+512].to(device)).cpu().numpy())
        for s in range(0, pts_te.shape[0], 512):
            feat_te.append(model_f.perslay(pts_te[s:s+512].to(device), mask_te[s:s+512].to(device)).cpu().numpy())
    feat_tr = np.concatenate(feat_tr); feat_te = np.concatenate(feat_te)
    hyb_tr = np.concatenate([feat_tr, tf_tr_np], axis=1)
    hyb_te = np.concatenate([feat_te, tf_te_np], axis=1)
    sc = StandardScaler().fit(hyb_tr)
    rf2 = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    cv_h = cross_val_score(rf2, sc.transform(hyb_tr), y_tr, cv=skf, scoring="accuracy", n_jobs=1)
    rf2.fit(sc.transform(hyb_tr), y_tr)
    test_h = accuracy_score(y_te, rf2.predict(sc.transform(hyb_te)))
    print(f"  CV={cv_h.mean():.3f}±{cv_h.std():.3f} TEST={test_h:.3f}")
    results["perslay_hybrid_rf"] = {"cv": float(cv_h.mean()), "cv_std": float(cv_h.std()),
                                    "test": float(test_h)}

    # 汇总
    print("\n" + "=" * 68)
    print("ECG5000 结果汇总 (5类, train=500, test=4500)")
    print("=" * 68)
    rows = [
        ("固定 Top-K(10D)+RF", cv_topk.mean(), test_topk),
        ("固定 Top-K+时频(21D)+RF", cv_fuse.mean(), test_fuse),
        ("PersLay 纯拓扑 MLP", cv_p, test_p),
        ("PersLay+时频 MLP", cv_f, test_f),
        ("PersLay特征+时频 → RF", cv_h.mean(), test_h),
    ]
    for name, cv, te in rows:
        print(f"  {name:28s} CV={cv:.1%}  TEST={te:.1%}")

    best_learn = max([("PersLay topo", cv_p, test_p), ("PersLay+TF", cv_f, test_f),
                      ("PersLay+RF", cv_h.mean(), test_h)], key=lambda r: r[2])
    best_fixed = max([("Top-K+RF", cv_topk.mean(), test_topk),
                      ("Fuse+RF", cv_fuse.mean(), test_fuse)], key=lambda r: r[2])
    print(f"\n  最佳可学习: {best_learn[0]} TEST={best_learn[2]:.1%}")
    print(f"  最佳固定:   {best_fixed[0]} TEST={best_fixed[2]:.1%}")
    delta = best_learn[2] - best_fixed[2]
    print(f"  差值: {delta:+.1%}")

    # 任务拓扑相关性检验：纯拓扑准确率（对比 FordA 的 50-62%）
    print(f"\n  任务拓扑相关性检验:")
    print(f"    ECG5000 纯拓扑: Top-K+RF {test_topk:.1%} / PersLay {test_p:.1%}")
    print(f"    (对比 FordA 纯拓扑: 50.3% / 61.7%; ECG200 纯拓扑: 74% / 74%)")
    if test_topk >= 0.70 or test_p >= 0.70:
        print("    ✓ ECG5000 拓扑相关性确实更强（纯拓扑显著高于 FordA）")

    if delta >= 0:
        print(f"\n  ✅ 假设成立：拓扑强相关任务上可学习向量化 ≥ 固定向量化 ({delta:+.1%})")
    elif delta > -0.02:
        print(f"\n  △ 接近持平 ({delta:+.1%})")
    else:
        print(f"\n  ✗ 可学习向量化仍落后 ({delta:+.1%})")

    with open(RESULTS_DIR / "results_ecg5000.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 可视化：三数据集全景对比
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    ax = axes[0]
    names = [r[0].replace("+", "+\n") for r in rows]
    cvs = [r[1] for r in rows]
    tests = [r[2] for r in rows]
    x = np.arange(len(rows))
    ax.bar(x - 0.2, cvs, 0.4, label="5-fold CV (train)", color="#64B5F6", edgecolor="black")
    ax.bar(x + 0.2, tests, 0.4, label="Official Test (4500)", color="#1565C0", edgecolor="black")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy (5-class)")
    ax.set_title("ECG5000: Fixed vs Learnable Vectorization")
    for i, (c, t) in enumerate(zip(cvs, tests)):
        ax.text(i - 0.2, c + 0.01, f"{c:.1%}", ha="center", fontsize=8)
        ax.text(i + 0.2, t + 0.01, f"{t:.1%}", ha="center", fontsize=8)
    ax.legend(); ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    # 三数据集：任务拓扑相关性(纯拓扑acc) vs 可学习-固定差距
    datasets = ["FordA\n(噪声,弱拓扑)", "ECG5000\n(心律,强拓扑)", "ECG200\n(心拍,中拓扑)"]
    pure_topo = [0.503, test_topk, 0.74]          # 纯拓扑最佳 test
    gaps = [-0.016, delta, 0.81 - 0.85]           # 可学习-固定
    ax2 = ax.twinx()
    b1 = ax.bar(np.arange(3) - 0.2, pure_topo, 0.4, color="#FF9800", edgecolor="black",
                label="Pure-topo acc (task relevance)")
    b2 = ax2.bar(np.arange(3) + 0.2, gaps, 0.4, color="#4CAF50", edgecolor="black",
                 label="Learnable − Fixed gap")
    ax.axhline(0, color="black", lw=0.8)
    ax2.axhline(0, color="black", lw=0.8)
    ax.set_xticks(np.arange(3)); ax.set_xticklabels(datasets, fontsize=9)
    ax.set_ylabel("Pure-topo Test Acc", color="#FF9800")
    ax2.set_ylabel("Learnable − Fixed (pp)", color="#4CAF50")
    ax.set_ylim(0, 1.0); ax2.set_ylim(-0.10, 0.10)
    ax.set_title("Task Topological Relevance vs Learnable Advantage")
    for i, v in enumerate(pure_topo):
        ax.text(i - 0.2, v + 0.02, f"{v:.1%}", ha="center", fontsize=9)
    for i, v in enumerate(gaps):
        ax2.text(i + 0.2, v + (0.005 if v >= 0 else -0.012), f"{v:+.1%}", ha="center", fontsize=9)
    lines = [b1, b2]
    ax.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="upper left")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "ecg5000_comparison.png", dpi=140, bbox_inches="tight")
    plt.close()
    print(f"\n  [已保存] {RESULTS_DIR}/results_ecg5000.json, ecg5000_comparison.png")


if __name__ == "__main__":
    main()
