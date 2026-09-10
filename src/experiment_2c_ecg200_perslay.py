#!/usr/bin/env python3
"""
阶段 2C：ECG200 端到端可学习向量化分类
- PD 预计算缓存（m=5, τ=4，2A 冲刺最优嵌入参数）
- PersLayLite → MLP：纯拓扑 / 拓扑+时频融合 两种配置
- 协议：train 5折CV（早停+记录最优epoch）→ 最终 epoch=CV中位数 → test 只评一次
- 基线：固定向量化 Top-K+RF=74%（2B）、融合28D+RF(50)=85%（2A冲刺）
"""
import sys
import json
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
from experiment_2a_embedding import time_delay_embedding
from experiment_2a_ecg200 import load_arff, timefreq_features
from ph_pipeline import compute_persistence_diagrams

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "exp2c_perslay"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "ecg200"

M_EMB, TAU_EMB = 5, 4      # 2A 冲刺最优
GRID = 20                   # PersLay 网格 20×20=400D
HIDDEN, DROPOUT = 32, 0.3
LR, WD = 1e-3, 1e-3
MAX_EPOCHS, PATIENCE = 300, 30
SEED = 42


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_pd_cache(X, cache_path):
    """计算并缓存所有样本的 H1 持久图（变长，存 object npz）"""
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        return list(data["diagrams"])
    diagrams = []
    for i, sig in enumerate(X):
        pts = time_delay_embedding(sig, m=M_EMB, tau=TAU_EMB)
        dgms = compute_persistence_diagrams(pts, maxdim=1)
        diagrams.append(dgms[1])
        if (i + 1) % 50 == 0:
            print(f"    PH {i+1}/{len(X)}", flush=True)
    np.savez(cache_path, diagrams=np.array(diagrams, dtype=object))
    return diagrams


def pers_range_from(diagrams, q=99.0):
    """由数据分位数确定网格范围（避免手拍）"""
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
    if not births:
        return (0.0, 1.0)
    births = np.concatenate(births)
    perss = np.concatenate(perss)
    b_max = float(np.percentile(births, q))
    p_max = float(np.percentile(perss, q))
    rng_max = max(b_max, p_max) * 1.1
    return (0.0, rng_max)


def train_one_fold(model, pts, mask, tf, y, tr_idx, va_idx, device):
    """单折训练：早停，返回 (最优val_acc, 最优epoch, 训练后模型state)"""
    from sklearn.metrics import accuracy_score
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    loss_fn = nn.BCEWithLogitsLoss()

    def batch(idx):
        b = (pts[idx].to(device), mask[idx].to(device),
             tf[idx].to(device) if tf is not None else None,
             y[idx].to(device))
        return b

    best_acc, best_epoch, bad = -1.0, 0, 0
    best_state = None
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        p, m_, t, yy = batch(tr_idx)
        opt.zero_grad()
        logits = model(p, m_, t)
        loss = loss_fn(logits, yy.float())
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            p, m_, t, yy = batch(va_idx)
            va_logits = model(p, m_, t)
            va_pred = (torch.sigmoid(va_logits) > 0.5).cpu().numpy().astype(int)
            va_acc = accuracy_score(yy.cpu().numpy(), va_pred)
        if va_acc > best_acc:
            best_acc, best_epoch, bad = va_acc, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    return best_acc, best_epoch, best_state


def run_cv(pts, mask, tf, y, device, pers_range, n_splits=5):
    """5折CV：返回各折 (val_acc, best_epoch) 与平均学习参数"""
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    y_np = y.numpy()
    fold_accs, fold_epochs, learned = [], [], []
    for fi, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y_np)), y_np)):
        set_seed(SEED + fi)
        model = TopoClassifier(grid_size=GRID, tf_dim=(tf.shape[1] if tf is not None else 0),
                               hidden=HIDDEN, dropout=DROPOUT, pers_range=pers_range).to(device)
        acc, ep, _ = train_one_fold(model, pts, mask, tf, y, tr_idx, va_idx, device)
        fold_accs.append(acc)
        fold_epochs.append(ep)
        learned.append((float(model.perslay.a.detach()), float(model.perslay.b.detach()),
                        float(torch.exp(model.perslay.log_sigma).detach())))
        print(f"    fold{fi+1}: val_acc={acc:.3f}, best_epoch={ep}", flush=True)
    return fold_accs, fold_epochs, learned


def final_train_and_test(pts_tr, mask_tr, tf_tr, y_tr, pts_te, mask_te, tf_te, y_te,
                         device, pers_range, n_epochs):
    """用 CV 中位数 epoch 全量训练，test 只评一次"""
    from sklearn.metrics import accuracy_score
    set_seed(SEED)
    model = TopoClassifier(grid_size=GRID, tf_dim=(tf_tr.shape[1] if tf_tr is not None else 0),
                           hidden=HIDDEN, dropout=DROPOUT, pers_range=pers_range).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(n_epochs):
        opt.zero_grad()
        logits = model(pts_tr.to(device), mask_tr.to(device),
                       tf_tr.to(device) if tf_tr is not None else None)
        loss = loss_fn(logits, y_tr.to(device).float())
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        te_logits = model(pts_te.to(device), mask_te.to(device),
                          tf_te.to(device) if tf_te is not None else None)
        te_pred = (torch.sigmoid(te_logits) > 0.5).cpu().numpy().astype(int)
    acc = accuracy_score(y_te.numpy(), te_pred)
    learned = (float(model.perslay.a.detach()), float(model.perslay.b.detach()),
               float(torch.exp(model.perslay.log_sigma).detach()))
    return acc, learned, model


def visualize_weight_fn(learned, pers_range, save_path):
    """可视化学习到的权重函数 w(pers)=sigmoid(a·pers+b) vs 2A 手工 30% 判据"""
    a, b, sigma = learned
    pers = np.linspace(pers_range[0], pers_range[1], 200)
    w = 1.0 / (1.0 + np.exp(-(a * pers + b)))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(pers, w, color="#2196F3", lw=2,
            label=f"learned: sigmoid({a:.2f}·pers{b:+.2f})")
    # 2A 手工判据参照：30% 最大持久度处跳变（用范围中值近似"最大持久度"）
    p_ref = pers_range[1] * 0.5
    ax.axvline(0.3 * p_ref, color="red", ls="--", alpha=0.7,
               label=f"2A heuristic: pers > 0.3×max (≈{0.3*p_ref:.2f})")
    ax.set_xlabel("Persistence")
    ax.set_ylabel("Weight w(pers)")
    ax.set_title(f"Learned weight function (σ={sigma:.3f})")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close()


def main():
    sys.stdout.reconfigure(line_buffering=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    # [1] 数据 + PD 缓存
    print("\n[1/5] 加载数据 + PD 缓存 (m=5, τ=4)...")
    X_tr, y_tr = load_arff(DATA_DIR / "ECG200_TRAIN.arff")
    X_te, y_te = load_arff(DATA_DIR / "ECG200_TEST.arff")
    print(f"  TRAIN {X_tr.shape}, TEST {X_te.shape}")
    dgm_tr = compute_pd_cache(X_tr, RESULTS_DIR / "pd_train.npz")
    dgm_te = compute_pd_cache(X_te, RESULTS_DIR / "pd_test.npz")
    n_pts_tr = np.mean([len(d) for d in dgm_tr])
    print(f"  H1 持久点均值: {n_pts_tr:.1f}/样本")

    pers_range = pers_range_from(dgm_tr + dgm_te)
    print(f"  网格范围 (99%分位): {pers_range}")

    # [2] 张量化
    print("\n[2/5] 张量化...")
    pts_tr, mask_tr = diagrams_to_padded(dgm_tr)
    pts_te, mask_te = diagrams_to_padded(dgm_te, max_points=pts_tr.shape[1])
    tf_tr = torch.tensor(np.array([timefreq_features(s) for s in X_tr]), dtype=torch.float32)
    tf_te = torch.tensor(np.array([timefreq_features(s) for s in X_te]), dtype=torch.float32)
    # 时频特征标准化（用 train 统计量）
    tf_mu, tf_sd = tf_tr.mean(0), tf_tr.std(0) + 1e-9
    tf_tr_n, tf_te_n = (tf_tr - tf_mu) / tf_sd, (tf_te - tf_mu) / tf_sd
    y_tr_t, y_te_t = torch.tensor(y_tr), torch.tensor(y_te)
    print(f"  pts_tr {tuple(pts_tr.shape)}, tf {tuple(tf_tr.shape)}")

    results = {}

    # [3] 配置A：PersLay 纯拓扑
    print("\n[3/5] 配置A: PersLayLite 纯拓扑 (400D→MLP)...")
    accs_a, eps_a, learned_a = run_cv(pts_tr, mask_tr, None, y_tr_t, device, pers_range)
    cv_a = float(np.mean(accs_a))
    ep_a = int(np.median(eps_a))
    print(f"  CV = {cv_a:.3f} ± {np.std(accs_a):.3f}, 中位 epoch = {ep_a}")
    test_a, learn_a, model_a = final_train_and_test(
        pts_tr, mask_tr, None, y_tr_t, pts_te, mask_te, None, y_te_t, device, pers_range, ep_a)
    print(f"  TEST = {test_a:.3f} (epoch={ep_a})")
    results["perslay_topo"] = {"cv": cv_a, "cv_std": float(np.std(accs_a)),
                               "test": test_a, "epochs": ep_a,
                               "learned_a": learn_a[0], "learned_b": learn_a[1],
                               "learned_sigma": learn_a[2]}

    # [4] 配置B：PersLay + 时频融合
    print("\n[4/5] 配置B: PersLayLite + 时频融合 (400+11D→MLP)...")
    accs_b, eps_b, learned_b = run_cv(pts_tr, mask_tr, tf_tr_n, y_tr_t, device, pers_range)
    cv_b = float(np.mean(accs_b))
    ep_b = int(np.median(eps_b))
    print(f"  CV = {cv_b:.3f} ± {np.std(accs_b):.3f}, 中位 epoch = {ep_b}")
    test_b, learn_b, model_b = final_train_and_test(
        pts_tr, mask_tr, tf_tr_n, y_tr_t, pts_te, mask_te, tf_te_n, y_te_t, device, pers_range, ep_b)
    print(f"  TEST = {test_b:.3f} (epoch={ep_b})")
    results["perslay_fused"] = {"cv": cv_b, "cv_std": float(np.std(accs_b)),
                                "test": test_b, "epochs": ep_b,
                                "learned_a": learn_b[0], "learned_b": learn_b[1],
                                "learned_sigma": learn_b[2]}

    # [5] 保存 + 可视化
    print("\n[5/5] 保存结果 + 可视化...")
    # 基线（2A/2B 已测）
    results["baselines"] = {
        "topk_rf_2b": {"test": 0.74, "note": "2B任务3 固定Top-K+RF"},
        "fuse28_rf50_2a": {"test": 0.85, "note": "2A冲刺 固定融合28D+RF(50)"},
    }
    with open(RESULTS_DIR / "results_2c.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  [已保存] {RESULTS_DIR/'results_2c.json'}")

    visualize_weight_fn(learn_a, pers_range, RESULTS_DIR / "learned_weight_fn.png")
    print(f"  [已保存] {RESULTS_DIR/'learned_weight_fn.png'}")

    # 对比条形图
    fig, ax = plt.subplots(figsize=(9, 5))
    names = ["Top-K+RF\n(2B固定)", "PersLay\n(2C纯拓扑)", "融合28D+RF(50)\n(2A固定)", "PersLay+TF\n(2C融合)"]
    vals = [0.74, test_a, 0.85, test_b]
    colors = ["#90A4AE", "#2196F3", "#FF9800", "#4CAF50"]
    bars = ax.bar(names, vals, color=colors, edgecolor="black")
    ax.axhline(0.85, color="red", ls="--", alpha=0.7, label="85% target")
    ax.set_ylim(0.5, 0.95)
    ax.set_ylabel("ECG200 Test Accuracy")
    ax.set_title("Stage 2C: Learnable vs Fixed Vectorization")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.008, f"{v:.1%}", ha="center", fontsize=11)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "comparison_2c.png", dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  [已保存] {RESULTS_DIR/'comparison_2c.png'}")

    # 结论
    print("\n" + "=" * 60)
    print("阶段 2C 结论")
    print("=" * 60)
    print(f"  PersLay 纯拓扑:  CV {cv_a:.1%}  TEST {test_a:.1%}   (基线 Top-K+RF: 74%)")
    print(f"  PersLay+时频:    CV {cv_b:.1%}  TEST {test_b:.1%}   (基线 融合28D+RF: 85%)")
    print(f"  学习参数(融合): a={learn_b[0]:.2f}, b={learn_b[1]:.2f}, σ={learn_b[2]:.3f}")
    print(f"    → 权重半高点 pers* = {-learn_b[1]/learn_b[0]:.3f} (网格范围 {pers_range[1]:.2f})")
    if test_b >= 0.85:
        print("  ✅ 可学习向量化追平/超过固定向量化 (≥85%)")
    elif test_b > test_a:
        print(f"  △ 融合提升 {test_b-test_a:+.1%}，但未超固定基线 85%")
    else:
        print("  ✗ 未达基线，需检查模型容量/正则")


if __name__ == "__main__":
    main()
