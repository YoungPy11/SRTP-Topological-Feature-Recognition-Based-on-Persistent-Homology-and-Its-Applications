#!/usr/bin/env python3
"""
阶段 2C 改进实验：针对首轮两个问题的定向修复
- 变体A：PersLay 层单独 10× 学习率（修复"参数没动"的数据饥饿）
- 变体B：混合方案——训练后的 PersLay 特征(400D)+时频(11D) → RF
  （可学习向量化 + 2A 证明的小数据强分类器）
协议与首轮一致：5折CV选epoch，test只评一次。
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
from experiment_2a_ecg200 import load_arff, timefreq_features

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "exp2c_perslay"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "ecg200"

GRID, HIDDEN, DROPOUT = 20, 32, 0.3
LR_HEAD, LR_PERSLAY, WD = 1e-3, 1e-2, 1e-3   # 变体A：perslay 10× LR
MAX_EPOCHS, PATIENCE, SEED = 300, 30, 42


def set_seed(s=SEED):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def make_model(tf_dim, pers_range, device):
    return TopoClassifier(grid_size=GRID, tf_dim=tf_dim, hidden=HIDDEN,
                          dropout=DROPOUT, pers_range=pers_range).to(device)


def param_groups(model):
    return [
        {"params": list(model.perslay.parameters()), "lr": LR_PERSLAY},
        {"params": list(model.head.parameters()), "lr": LR_HEAD},
    ]


def train_fold(model, pts, mask, tf, y, tr_idx, va_idx, device):
    from sklearn.metrics import accuracy_score
    opt = torch.optim.Adam(param_groups(model), weight_decay=WD)
    loss_fn = nn.BCEWithLogitsLoss()
    best_acc, best_ep, bad, best_state = -1.0, 0, 0, None
    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        opt.zero_grad()
        logits = model(pts[tr_idx].to(device), mask[tr_idx].to(device),
                       tf[tr_idx].to(device) if tf is not None else None)
        loss = loss_fn(logits, y[tr_idx].to(device).float())
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            vl = model(pts[va_idx].to(device), mask[va_idx].to(device),
                       tf[va_idx].to(device) if tf is not None else None)
            vp = (torch.sigmoid(vl) > 0.5).cpu().numpy().astype(int)
            va = accuracy_score(y[va_idx].numpy(), vp)
        if va > best_acc:
            best_acc, best_ep, bad = va, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    return best_acc, best_ep, best_state


def cv_run(pts, mask, tf, y, device, pers_range, tag):
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    tf_dim = tf.shape[1] if tf is not None else 0
    accs, eps, learned, states = [], [], [], []
    for fi, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y)), y.numpy())):
        set_seed(SEED + fi)
        model = make_model(tf_dim, pers_range, device)
        acc, ep, state = train_fold(model, pts, mask, tf, y, tr_idx, va_idx, device)
        accs.append(acc); eps.append(ep)
        learned.append((float(state["perslay.a"].cpu()), float(state["perslay.b"].cpu()),
                        float(np.exp(state["perslay.log_sigma"].cpu().item()))))
        states.append(state)
        print(f"    [{tag}] fold{fi+1}: val={acc:.3f} ep={ep} "
              f"a={learned[-1][0]:.2f} b={learned[-1][1]:.2f} σ={learned[-1][2]:.3f}", flush=True)
    return accs, eps, learned, states


def extract_perslay_features(model, pts, mask, device):
    model.eval()
    with torch.no_grad():
        return model.perslay(pts.to(device), mask.to(device)).cpu().numpy()


def main():
    sys.stdout.reconfigure(line_buffering=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # 复用首轮 PD 缓存
    dgm_tr = list(np.load(RESULTS_DIR / "pd_train.npz", allow_pickle=True)["diagrams"])
    dgm_te = list(np.load(RESULTS_DIR / "pd_test.npz", allow_pickle=True)["diagrams"])
    X_tr, y_tr = load_arff(DATA_DIR / "ECG200_TRAIN.arff")
    X_te, y_te = load_arff(DATA_DIR / "ECG200_TEST.arff")

    pts_tr, mask_tr = diagrams_to_padded(dgm_tr)
    pts_te, mask_te = diagrams_to_padded(dgm_te, max_points=pts_tr.shape[1])
    tf_tr = torch.tensor(np.array([timefreq_features(s) for s in X_tr]), dtype=torch.float32)
    tf_te = torch.tensor(np.array([timefreq_features(s) for s in X_te]), dtype=torch.float32)
    mu, sd = tf_tr.mean(0), tf_tr.std(0) + 1e-9
    tf_tr_n, tf_te_n = (tf_tr - mu) / sd, (tf_te - mu) / sd
    y_tr_t, y_te_t = torch.tensor(y_tr), torch.tensor(y_te)

    # 网格范围与首轮一致
    births, perss = [], []
    for dgm in dgm_tr + dgm_te:
        d = np.asarray(dgm)
        if len(d):
            d = d[~np.isinf(d[:, 1])]
            d = d[d[:, 1] > d[:, 0]]
            if len(d):
                births.append(d[:, 0]); perss.append(d[:, 1] - d[:, 0])
    rng = max(float(np.percentile(np.concatenate(births), 99)),
              float(np.percentile(np.concatenate(perss), 99))) * 1.1
    pers_range = (0.0, rng)
    print(f"pers_range: {pers_range}")

    results = {}

    # ===== 变体A：分离学习率，纯拓扑 =====
    print("\n[变体A] PersLay LR=1e-2 (10×)，纯拓扑...")
    accs_a, eps_a, learned_a, _ = cv_run(pts_tr, mask_tr, None, y_tr_t, device, pers_range, "A-topo")
    cv_a = float(np.mean(accs_a)); ep_a = int(np.median(eps_a))
    print(f"  CV = {cv_a:.3f}±{np.std(accs_a):.3f}, 中位epoch={ep_a}")
    set_seed(SEED)
    model_a = make_model(0, pers_range, device)
    opt = torch.optim.Adam(param_groups(model_a), weight_decay=WD)
    loss_fn = nn.BCEWithLogitsLoss()
    model_a.train()
    for _ in range(ep_a):
        opt.zero_grad()
        lg = model_a(pts_tr.to(device), mask_tr.to(device), None)
        loss_fn(lg, y_tr_t.to(device).float()).backward()
        opt.step()
    model_a.eval()
    with torch.no_grad():
        tp = (torch.sigmoid(model_a(pts_te.to(device), mask_te.to(device), None)) > 0.5).cpu().numpy().astype(int)
    from sklearn.metrics import accuracy_score
    test_a = accuracy_score(y_te, tp)
    la = (float(model_a.perslay.a.detach()), float(model_a.perslay.b.detach()),
          float(torch.exp(model_a.perslay.log_sigma).detach()))
    print(f"  TEST = {test_a:.3f}; 学习参数 a={la[0]:.2f}, b={la[1]:.2f}, σ={la[2]:.3f}")
    results["variantA_topo_splitLR"] = {"cv": cv_a, "cv_std": float(np.std(accs_a)),
                                        "test": float(test_a), "epochs": ep_a,
                                        "learned": list(la)}

    # ===== 变体A2：分离学习率，融合 =====
    print("\n[变体A2] PersLay LR=1e-2，融合(+时频11D)...")
    accs_a2, eps_a2, learned_a2, _ = cv_run(pts_tr, mask_tr, tf_tr_n, y_tr_t, device, pers_range, "A2-fuse")
    cv_a2 = float(np.mean(accs_a2)); ep_a2 = int(np.median(eps_a2))
    print(f"  CV = {cv_a2:.3f}±{np.std(accs_a2):.3f}, 中位epoch={ep_a2}")
    set_seed(SEED)
    model_a2 = make_model(tf_tr_n.shape[1], pers_range, device)
    opt = torch.optim.Adam(param_groups(model_a2), weight_decay=WD)
    model_a2.train()
    for _ in range(ep_a2):
        opt.zero_grad()
        lg = model_a2(pts_tr.to(device), mask_tr.to(device), tf_tr_n.to(device))
        loss_fn(lg, y_tr_t.to(device).float()).backward()
        opt.step()
    model_a2.eval()
    with torch.no_grad():
        tp = (torch.sigmoid(model_a2(pts_te.to(device), mask_te.to(device), tf_te_n.to(device))) > 0.5).cpu().numpy().astype(int)
    test_a2 = accuracy_score(y_te, tp)
    la2 = (float(model_a2.perslay.a.detach()), float(model_a2.perslay.b.detach()),
           float(torch.exp(model_a2.perslay.log_sigma).detach()))
    print(f"  TEST = {test_a2:.3f}; 学习参数 a={la2[0]:.2f}, b={la2[1]:.2f}, σ={la2[2]:.3f}")
    results["variantA2_fused_splitLR"] = {"cv": cv_a2, "cv_std": float(np.std(accs_a2)),
                                          "test": float(test_a2), "epochs": ep_a2,
                                          "learned": list(la2)}

    # ===== 变体B：PersLay特征 → RF 混合 =====
    print("\n[变体B] PersLay特征(400D)+时频(11D) → RF(50)...")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler

    # 用变体A2最终模型抽取 PersLay 特征（在 train 上训练过的）
    feat_tr = extract_perslay_features(model_a2, pts_tr, mask_tr, device)
    feat_te = extract_perslay_features(model_a2, pts_te, mask_te, device)
    hyb_tr = np.concatenate([feat_tr, tf_tr_n.numpy()], axis=1)
    hyb_te = np.concatenate([feat_te, tf_te_n.numpy()], axis=1)
    sc = StandardScaler().fit(hyb_tr)
    hyb_tr_s, hyb_te_s = sc.transform(hyb_tr), sc.transform(hyb_te)
    rf = RandomForestClassifier(n_estimators=50, random_state=SEED)
    cv_b = cross_val_score(rf, hyb_tr_s, y_tr, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                           scoring="accuracy")
    rf.fit(hyb_tr_s, y_tr)
    test_b = accuracy_score(y_te, rf.predict(hyb_te_s))
    print(f"  CV = {cv_b.mean():.3f}±{cv_b.std():.3f}, TEST = {test_b:.3f}")
    results["variantB_hybrid_RF"] = {"cv": float(cv_b.mean()), "cv_std": float(cv_b.std()),
                                     "test": float(test_b)}

    # ===== 汇总对比 =====
    print("\n" + "=" * 64)
    print("阶段 2C 改进实验汇总")
    print("=" * 64)
    rows = [
        ("首轮 PersLay 纯拓扑 (统一LR)", 0.810, 0.740),
        ("首轮 PersLay+TF (统一LR)", 0.820, 0.780),
        ("A  分离LR 纯拓扑", cv_a, test_a),
        ("A2 分离LR 融合", cv_a2, test_a2),
        ("B  PersLay特征+RF 混合", cv_b.mean(), test_b),
        ("基线 固定Top-K+RF (2B)", 0.720, 0.740),
        ("基线 固定融合28D+RF50 (2A)", None, 0.850),
    ]
    for name, cv, te in rows:
        cv_s = f"{cv:.1%}" if cv is not None else "—"
        print(f"  {name:32s} CV={cv_s:8s} TEST={te:.1%}")

    best = max([("A", cv_a, test_a), ("A2", cv_a2, test_a2), ("B", cv_b.mean(), test_b)],
               key=lambda r: r[2])
    print(f"\n  本轮最佳: 变体{best[0]} TEST={best[2]:.1%}")
    if best[2] >= 0.85:
        print("  ✅ 追平/超过固定基线 85%")
    else:
        print(f"  △ 距固定基线 85% 还差 {0.85-best[2]:.1%}")

    # 参数移动量对比（首轮 vs 分离LR）
    print(f"\n  参数移动量（分离LR后是否真的动了）:")
    print(f"    首轮: a 2.00→2.00, b -1.00→-0.99, σ 0.100→0.101")
    print(f"    A2:   a 2.00→{la2[0]:.2f}, b -1.00→{la2[1]:.2f}, σ 0.100→{la2[2]:.3f}")

    with open(RESULTS_DIR / "results_2c_refine.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  [已保存] {RESULTS_DIR/'results_2c_refine.json'}")

    # 可视化
    fig, ax = plt.subplots(figsize=(10, 5.5))
    names = ["Top-K+RF\n(2B fixed)", "PersLay\n(2C-r1 topo)", "PersLay+TF\n(2C-r1)",
             "splitLR topo\n(2C-A)", "splitLR fused\n(2C-A2)", "PersLay+RF\n(2C-B hybrid)",
             "Fixed fuse+RF\n(2A best)"]
    vals = [0.74, 0.74, 0.78, test_a, test_a2, test_b, 0.85]
    colors = ["#90A4AE", "#64B5F6", "#2196F3", "#7986CB", "#3F51B5", "#4CAF50", "#FF9800"]
    bars = ax.bar(names, vals, color=colors, edgecolor="black")
    ax.axhline(0.85, color="red", ls="--", alpha=0.7, label="85% (2A fixed baseline)")
    ax.set_ylim(0.5, 0.95)
    ax.set_ylabel("ECG200 Test Accuracy")
    ax.set_title("Stage 2C: Learnable Vectorization — Round 1 vs Refinements")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.008, f"{v:.1%}", ha="center", fontsize=10)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "comparison_2c_refine.png", dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  [已保存] {RESULTS_DIR/'comparison_2c_refine.png'}")


if __name__ == "__main__":
    main()
