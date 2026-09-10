#!/usr/bin/env python3
"""
阶段 2C 稳定性确认（ECG5000）：+0.6pp 反超是否可靠？
=====================================================
按评审意见补三项：
1. 多种子重复（5 seeds）：PersLay+RF 的 test 准确率分布，判断 92.0% 是稳定还是种子运气
2. McNemar 检验：PersLay+RF vs 固定 Top-K+时频+RF 的配对预测差异是否显著
3. 类别不平衡诊断：ECG5000 五类极度不均衡 → 报告 per-class precision/recall/F1、
   macro-F1、balanced accuracy，并测 RF class_weight='balanced' 变体

额外变体 H2：原实验混合特征取自"融合MLP"模型（该配置在5类下发散，CV中位epoch=3，
PersLay层仅训练3轮）；H2 改用训练稳定的"纯拓扑"模型抽特征（epoch≈12），检验配置敏感性。
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
from experiment_2a_ecg200 import timefreq_features
from experiment_2c_ecg5000 import load_arff_multi, topk_feats, pers_range_from

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "exp2c_ecg5000"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "ecg5000"

GRID, HIDDEN, DROPOUT = 20, 32, 0.3
LR_HEAD, LR_PERSLAY, WD = 1e-3, 1e-2, 1e-3
BATCH = 128
MAX_EPOCHS, PATIENCE = 80, 12
NUM_CLASSES = 5
SEEDS = [42, 0, 1, 2, 3]
CLASS_NAMES = ["N(正常)", "S(室上性)", "V(室性)", "F(融合)", "Q(未知)"]


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def train_fold(model, pts, mask, tf, y, tr_idx, va_idx, device, seed):
    from sklearn.metrics import accuracy_score
    opt = torch.optim.Adam([
        {"params": list(model.perslay.parameters()), "lr": LR_PERSLAY},
        {"params": list(model.head.parameters()), "lr": LR_HEAD},
    ], weight_decay=WD)
    loss_fn = nn.CrossEntropyLoss()
    rng = np.random.RandomState(seed)
    best_acc, best_ep, bad = -1.0, 0, 0
    n_tr = len(tr_idx)
    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = rng.permutation(n_tr)
        for s in range(0, n_tr, BATCH):
            bidx = tr_idx[perm[s:s + BATCH]]
            opt.zero_grad()
            logits = model(pts[bidx].to(device), mask[bidx].to(device),
                           tf[bidx].to(device) if tf is not None else None)
            loss_fn(logits, y[bidx].to(device)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 防多分类发散
            opt.step()
        model.eval()
        preds = []
        with torch.no_grad():
            for s in range(0, len(va_idx), 512):
                bidx = va_idx[s:s + 512]
                vl = model(pts[bidx].to(device), mask[bidx].to(device),
                           tf[bidx].to(device) if tf is not None else None)
                preds.append(vl.argmax(1).cpu().numpy())
        va = accuracy_score(y[va_idx].numpy(), np.concatenate(preds))
        if va > best_acc:
            best_acc, best_ep, bad = va, ep, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    return best_acc, best_ep


def median_epoch_cv(pts, mask, tf, y, device, pers_range, seed):
    """复现原协议：5折CV取中位最优epoch"""
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    tf_dim = tf.shape[1] if tf is not None else 0
    eps = []
    for fi, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y.numpy())), y.numpy())):
        set_seed(seed + fi)
        model = TopoClassifier(grid_size=GRID, tf_dim=tf_dim, hidden=HIDDEN,
                               dropout=DROPOUT, pers_range=pers_range,
                               num_classes=NUM_CLASSES).to(device)
        _, ep = train_fold(model, pts, mask, tf, y, tr_idx, va_idx, device, seed + fi)
        eps.append(ep)
    return int(np.median(eps)), eps


def train_final(pts, mask, tf, y, device, pers_range, seed, n_epochs):
    set_seed(seed)
    tf_dim = tf.shape[1] if tf is not None else 0
    model = TopoClassifier(grid_size=GRID, tf_dim=tf_dim, hidden=HIDDEN,
                           dropout=DROPOUT, pers_range=pers_range,
                           num_classes=NUM_CLASSES).to(device)
    opt = torch.optim.Adam([
        {"params": list(model.perslay.parameters()), "lr": LR_PERSLAY},
        {"params": list(model.head.parameters()), "lr": LR_HEAD},
    ], weight_decay=WD)
    loss_fn = nn.CrossEntropyLoss()
    rng = np.random.RandomState(seed)
    n = pts.shape[0]
    for _ in range(n_epochs):
        model.train()
        perm = rng.permutation(n)
        for s in range(0, n, BATCH):
            bidx = perm[s:s + BATCH]
            opt.zero_grad()
            logits = model(pts[bidx].to(device), mask[bidx].to(device),
                           tf[bidx].to(device) if tf is not None else None)
            loss_fn(logits, y[bidx].to(device)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 防多分类发散
            opt.step()
    return model


def perslay_feats(model, pts, mask, device):
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, pts.shape[0], 512):
            out.append(model.perslay(pts[s:s+512].to(device), mask[s:s+512].to(device)).cpu().numpy())
    feats = np.concatenate(out)
    if not np.all(np.isfinite(feats)):
        n_bad = int(np.sum(~np.all(np.isfinite(feats), axis=1)))
        print(f"  [警告] PersLay特征含NaN/Inf（{n_bad}样本），已置0", flush=True)
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats


def mcnemar_test(y_true, pred_a, pred_b):
    """McNemar 精确二项检验：A=PersLay混合, B=固定基线"""
    from scipy.stats import binomtest
    b = int(np.sum((pred_a == y_true) & (pred_b != y_true)))  # A对B错
    c = int(np.sum((pred_a != y_true) & (pred_b == y_true)))  # A错B对
    n = b + c
    p = binomtest(min(b, c), n, 0.5).pvalue if n > 0 else 1.0
    return b, c, float(p)


def per_class_report(y_true, pred, n_classes=NUM_CLASSES):
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, pred, labels=list(range(n_classes)), zero_division=0)
    cm = confusion_matrix(y_true, pred, labels=list(range(n_classes)))
    return {"precision": prec.tolist(), "recall": rec.tolist(), "f1": f1.tolist(),
            "support": sup.tolist(), "confusion": cm.tolist()}


def main():
    sys.stdout.reconfigure(line_buffering=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

    # 数据 + 缓存
    X_tr, y_tr = load_arff_multi(DATA_DIR / "ECG5000_TRAIN.arff")
    X_te, y_te = load_arff_multi(DATA_DIR / "ECG5000_TEST.arff")
    dgm_tr = list(np.load(RESULTS_DIR / "pd_train.npz", allow_pickle=True)["diagrams"])
    dgm_te = list(np.load(RESULTS_DIR / "pd_test.npz", allow_pickle=True)["diagrams"])
    pers_range = pers_range_from(dgm_tr + dgm_te)

    print("\n[0] 类别不平衡诊断")
    cnt_tr, cnt_te = np.bincount(y_tr, minlength=5), np.bincount(y_te, minlength=5)
    for i in range(5):
        print(f"  类{i} {CLASS_NAMES[i]:10s}: train {cnt_tr[i]:4d} ({cnt_tr[i]/len(y_tr):5.1%})  "
              f"test {cnt_te[i]:5d} ({cnt_te[i]/len(y_te):5.1%})")
    print(f"  不平衡比(train 最大/最小): {cnt_tr.max()/max(cnt_tr.min(),1):.1f}×")

    pts_tr, mask_tr = diagrams_to_padded(dgm_tr)
    pts_te, mask_te = diagrams_to_padded(dgm_te, max_points=pts_tr.shape[1])
    tk_tr, tk_te = topk_feats(dgm_tr), topk_feats(dgm_te)
    tf_tr = np.array([timefreq_features(s) for s in X_tr])
    tf_te = np.array([timefreq_features(s) for s in X_te])
    tf_tr_t = torch.tensor(tf_tr, dtype=torch.float32)
    tf_te_t = torch.tensor(tf_te, dtype=torch.float32)
    mu, sd = tf_tr_t.mean(0), tf_tr_t.std(0) + 1e-9
    tf_tr_n, tf_te_n = (tf_tr_t - mu) / sd, (tf_te_t - mu) / sd
    y_tr_t, y_te_t = torch.tensor(y_tr, dtype=torch.long), torch.tensor(y_te, dtype=torch.long)

    # 固定基线特征（与种子无关，只有 RF 随种子）
    # 注意：下游是 RF（树模型），对特征缩放不敏感，无需 StandardScaler
    # （原实验用 scaler 在近零方差列上会放大到 4e41 导致 float32 溢出崩溃）
    fixed_tr = np.concatenate([tk_tr, tf_tr], axis=1)
    fixed_te = np.concatenate([tk_te, tf_te], axis=1)

    results = {"class_dist_train": cnt_tr.tolist(), "class_dist_test": cnt_te.tolist(), "seeds": {}}

    pred_fixed_42 = pred_h1_42 = pred_h2_42 = None
    for seed in SEEDS:
        print(f"\n===== seed {seed} =====")
        # 固定基线 RF
        rf_f = RandomForestClassifier(100, random_state=seed, n_jobs=-1)
        rf_f.fit(fixed_tr, y_tr)
        p_fixed = rf_f.predict(fixed_te)
        acc_fixed = accuracy_score(y_te, p_fixed)

        # H1 复现：融合MLP配置 → 抽 PersLay 特征 → RF
        ep1, eps1 = median_epoch_cv(pts_tr, mask_tr, tf_tr_n, y_tr_t, device, pers_range, seed)
        m1 = train_final(pts_tr, mask_tr, tf_tr_n, y_tr_t, device, pers_range, seed, ep1)
        f1_tr, f1_te = perslay_feats(m1, pts_tr, mask_tr, device), perslay_feats(m1, pts_te, mask_te, device)
        h1_tr = np.concatenate([f1_tr, tf_tr], axis=1); h1_te = np.concatenate([f1_te, tf_te], axis=1)
        rf1 = RandomForestClassifier(100, random_state=seed, n_jobs=-1)
        rf1.fit(h1_tr, y_tr)
        p_h1 = rf1.predict(h1_te)
        acc_h1 = accuracy_score(y_te, p_h1)

        # H2 变体：纯拓扑配置（训练稳定）→ 抽特征 → RF
        ep2, eps2 = median_epoch_cv(pts_tr, mask_tr, None, y_tr_t, device, pers_range, seed)
        m2 = train_final(pts_tr, mask_tr, None, y_tr_t, device, pers_range, seed, ep2)
        f2_tr, f2_te = perslay_feats(m2, pts_tr, mask_tr, device), perslay_feats(m2, pts_te, mask_te, device)
        h2_tr = np.concatenate([f2_tr, tf_tr], axis=1); h2_te = np.concatenate([f2_te, tf_te], axis=1)
        rf2 = RandomForestClassifier(100, random_state=seed, n_jobs=-1)
        rf2.fit(h2_tr, y_tr)
        p_h2 = rf2.predict(h2_te)
        acc_h2 = accuracy_score(y_te, p_h2)

        # RF balanced 变体（类别不平衡）
        rfb = RandomForestClassifier(100, random_state=seed, n_jobs=-1,
                                     class_weight="balanced_subsample")
        rfb.fit(h1_tr, y_tr)
        p_h1b = rfb.predict(h1_te)

        print(f"  H1 epochs(CV中位)={ep1} {eps1}")
        print(f"  H2 epochs(CV中位)={ep2} {eps2}")
        print(f"  固定 Top-K+TF+RF : acc={acc_fixed:.4f} macroF1={f1_score(y_te,p_fixed,average='macro'):.4f} "
              f"balAcc={balanced_accuracy_score(y_te,p_fixed):.4f}")
        print(f"  H1 PersLay+TF+RF : acc={acc_h1:.4f} macroF1={f1_score(y_te,p_h1,average='macro'):.4f} "
              f"balAcc={balanced_accuracy_score(y_te,p_h1):.4f}")
        print(f"  H2 PersLayTopo+RF: acc={acc_h2:.4f} macroF1={f1_score(y_te,p_h2,average='macro'):.4f} "
              f"balAcc={balanced_accuracy_score(y_te,p_h2):.4f}")
        print(f"  H1+RF-balanced   : acc={accuracy_score(y_te,p_h1b):.4f} "
              f"macroF1={f1_score(y_te,p_h1b,average='macro'):.4f} "
              f"balAcc={balanced_accuracy_score(y_te,p_h1b):.4f}")

        b, c, p_mcn = mcnemar_test(y_te, p_h1, p_fixed)
        print(f"  McNemar H1 vs 固定: H1独对={b} 固定独对={c} p={p_mcn:.4f}")

        results["seeds"][str(seed)] = {
            "ep_h1": ep1, "eps_h1": eps1, "ep_h2": ep2, "eps_h2": eps2,
            "acc_fixed": float(acc_fixed), "acc_h1": float(acc_h1), "acc_h2": float(acc_h2),
            "acc_h1_balanced": float(accuracy_score(y_te, p_h1b)),
            "macroF1_fixed": float(f1_score(y_te, p_fixed, average="macro")),
            "macroF1_h1": float(f1_score(y_te, p_h1, average="macro")),
            "macroF1_h2": float(f1_score(y_te, p_h2, average="macro")),
            "macroF1_h1_balanced": float(f1_score(y_te, p_h1b, average="macro")),
            "balAcc_fixed": float(balanced_accuracy_score(y_te, p_fixed)),
            "balAcc_h1": float(balanced_accuracy_score(y_te, p_h1)),
            "balAcc_h2": float(balanced_accuracy_score(y_te, p_h2)),
            "balAcc_h1_balanced": float(balanced_accuracy_score(y_te, p_h1b)),
            "mcnemar": {"h1_only_correct": b, "fixed_only_correct": c, "p_value": p_mcn},
        }
        if seed == 42:
            pred_fixed_42, pred_h1_42, pred_h2_42 = p_fixed, p_h1, p_h2
            results["per_class_seed42"] = {
                "fixed": per_class_report(y_te, p_fixed),
                "h1": per_class_report(y_te, p_h1),
                "h2": per_class_report(y_te, p_h2),
            }

    # ===== 汇总 =====
    print("\n" + "=" * 72)
    print("多种子稳定性汇总 (ECG5000, 5 seeds)")
    print("=" * 72)
    def col(k):
        return np.array([results["seeds"][str(s)][k] for s in SEEDS])
    for name, key in [("固定 Top-K+TF+RF", "acc_fixed"), ("H1 PersLay(融合配置)+TF+RF", "acc_h1"),
                      ("H2 PersLay(纯拓扑配置)+TF+RF", "acc_h2"), ("H1+RF balanced", "acc_h1_balanced")]:
        v = col(key)
        print(f"  {name:28s}: {v.mean():.4f} ± {v.std():.4f}  (min {v.min():.4f}, max {v.max():.4f})")
    print()
    for name, key in [("macroF1 固定", "macroF1_fixed"), ("macroF1 H1", "macroF1_h1"),
                      ("macroF1 H2", "macroF1_h2"), ("balAcc 固定", "balAcc_fixed"),
                      ("balAcc H1", "balAcc_h1"), ("balAcc H2", "balAcc_h2")]:
        v = col(key)
        print(f"  {name:16s}: {v.mean():.4f} ± {v.std():.4f}")

    # 配对比较：每种子 H1/H2 vs 固定
    print("\n  逐种子胜负（H1 − 固定 / H2 − 固定，百分点）:")
    wins1 = wins2 = 0
    for s in SEEDS:
        d1 = (results["seeds"][str(s)]["acc_h1"] - results["seeds"][str(s)]["acc_fixed"]) * 100
        d2 = (results["seeds"][str(s)]["acc_h2"] - results["seeds"][str(s)]["acc_fixed"]) * 100
        wins1 += d1 > 0; wins2 += d2 > 0
        print(f"    seed {s:3d}: H1 {d1:+.2f}pp   H2 {d2:+.2f}pp")
    print(f"  H1 胜出 {wins1}/{len(SEEDS)} 种子；H2 胜出 {wins2}/{len(SEEDS)} 种子")

    d_h1 = col("acc_h1") - col("acc_fixed")
    d_h2 = col("acc_h2") - col("acc_fixed")
    from scipy.stats import wilcoxon, ttest_rel
    try:
        p_w1 = wilcoxon(col("acc_h1"), col("acc_fixed")).pvalue
    except Exception:
        p_w1 = float("nan")
    try:
        p_w2 = wilcoxon(col("acc_h2"), col("acc_fixed")).pvalue
    except Exception:
        p_w2 = float("nan")
    p_t1 = ttest_rel(col("acc_h1"), col("acc_fixed")).pvalue
    p_t2 = ttest_rel(col("acc_h2"), col("acc_fixed")).pvalue
    print(f"\n  配对检验 H1 vs 固定: 均值差 {d_h1.mean()*100:+.2f}pp, Wilcoxon p={p_w1:.4f}, 配对t p={p_t1:.4f}")
    print(f"  配对检验 H2 vs 固定: 均值差 {d_h2.mean()*100:+.2f}pp, Wilcoxon p={p_w2:.4f}, 配对t p={p_t2:.4f}")
    results["summary"] = {
        "mean_diff_h1_pp": float(d_h1.mean() * 100), "std_diff_h1_pp": float(d_h1.std() * 100),
        "mean_diff_h2_pp": float(d_h2.mean() * 100), "std_diff_h2_pp": float(d_h2.std() * 100),
        "wilcoxon_p_h1": float(p_w1), "wilcoxon_p_h2": float(p_w2),
        "paired_t_p_h1": float(p_t1), "paired_t_p_h2": float(p_t2),
        "wins_h1": int(wins1), "wins_h2": int(wins2), "n_seeds": len(SEEDS),
    }

    # per-class 报告（seed 42）
    print("\n" + "=" * 72)
    print("Per-class 指标 (seed 42) — 类别不平衡诊断")
    print("=" * 72)
    for tag, key in [("固定", "fixed"), ("H1 PersLay混合", "h1"), ("H2 PersLay纯拓扑配置", "h2")]:
        r = results["per_class_seed42"][key]
        print(f"\n  [{tag}]")
        print(f"    {'类别':12s} {'support':>8s} {'precision':>10s} {'recall':>8s} {'F1':>8s}")
        for i in range(5):
            print(f"    {CLASS_NAMES[i]:12s} {r['support'][i]:8d} {r['precision'][i]:10.3f} "
                  f"{r['recall'][i]:8.3f} {r['f1'][i]:8.3f}")

    with open(RESULTS_DIR / "stability_ecg5000.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[已保存] {RESULTS_DIR/'stability_ecg5000.json'}")

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    ax = axes[0]
    data = [col("acc_fixed") * 100, col("acc_h1") * 100, col("acc_h2") * 100]
    bp = ax.boxplot(data, tick_labels=["Fixed\nTopK+TF+RF", "H1 PersLay\n(fused cfg)", "H2 PersLay\n(topo cfg)"],
                    patch_artist=True, widths=0.5)
    for patch, c in zip(bp["boxes"], ["#90A4AE", "#4CAF50", "#2196F3"]):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    for i, d in enumerate(data, 1):
        ax.scatter(np.full(len(d), i) + np.random.uniform(-0.08, 0.08, len(d)), d,
                   color="black", s=25, zorder=3)
    ax.set_ylabel("Test Accuracy (%)"); ax.set_ylim(88, 94)
    ax.set_title(f"Multi-seed Stability ({len(SEEDS)} seeds)\nH1−Fixed: {d_h1.mean()*100:+.2f}pp (p={p_w1:.3f})")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    x = np.arange(5); w = 0.27
    r42 = results["per_class_seed42"]
    ax.bar(x - w, r42["fixed"]["recall"], w, label="Fixed", color="#90A4AE", edgecolor="black")
    ax.bar(x, r42["h1"]["recall"], w, label="H1 PersLay", color="#4CAF50", edgecolor="black")
    ax.bar(x + w, r42["h2"]["recall"], w, label="H2 PersLay", color="#2196F3", edgecolor="black")
    ax.set_xticks(x); ax.set_xticklabels([f"{i}\n(n={cnt_te[i]})" for i in range(5)], fontsize=9)
    ax.set_ylabel("Per-class Recall"); ax.set_ylim(0, 1.05)
    ax.set_xlabel("Class (test support)")
    ax.set_title("Per-class Recall under Class Imbalance (seed 42)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    ax = axes[2]
    metrics = ["Accuracy", "Macro-F1", "Balanced Acc"]
    fixed_v = [col("acc_fixed").mean(), col("macroF1_fixed").mean(), col("balAcc_fixed").mean()]
    h1_v = [col("acc_h1").mean(), col("macroF1_h1").mean(), col("balAcc_h1").mean()]
    h2_v = [col("acc_h2").mean(), col("macroF1_h2").mean(), col("balAcc_h2").mean()]
    h1b_v = [col("acc_h1_balanced").mean(), col("macroF1_h1_balanced").mean(), col("balAcc_h1_balanced").mean()]
    xx = np.arange(3); w = 0.2
    ax.bar(xx - 1.5*w, np.array(fixed_v)*100, w, label="Fixed", color="#90A4AE", edgecolor="black")
    ax.bar(xx - 0.5*w, np.array(h1_v)*100, w, label="H1", color="#4CAF50", edgecolor="black")
    ax.bar(xx + 0.5*w, np.array(h2_v)*100, w, label="H2", color="#2196F3", edgecolor="black")
    ax.bar(xx + 1.5*w, np.array(h1b_v)*100, w, label="H1+RF-bal", color="#FF9800", edgecolor="black")
    ax.set_xticks(xx); ax.set_xticklabels(metrics)
    ax.set_ylabel("% (mean over seeds)"); ax.set_ylim(40, 100)
    ax.set_title("Imbalance-aware Metrics")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "stability_ecg5000.png", dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[已保存] {RESULTS_DIR/'stability_ecg5000.png'}")

    # 结论判定
    print("\n" + "=" * 72)
    print("结论判定")
    print("=" * 72)
    if p_w1 < 0.05 and d_h1.mean() > 0:
        print("  ✅ 「反超」成立且统计显著（Wilcoxon p<0.05）")
    elif d_h1.mean() > 0 and wins1 >= len(SEEDS) - 1:
        print(f"  △ 「反超」方向一致（{wins1}/{len(SEEDS)} 种子胜出）但未达统计显著（p={p_w1:.3f}）"
              f"→ 建议表述为「追平/略优」")
    else:
        print(f"  ✗ 「反超」不稳定（{wins1}/{len(SEEDS)} 种子胜出，均值差 {d_h1.mean()*100:+.2f}pp）"
              f"→ 应表述为「追平」")


if __name__ == "__main__":
    main()
