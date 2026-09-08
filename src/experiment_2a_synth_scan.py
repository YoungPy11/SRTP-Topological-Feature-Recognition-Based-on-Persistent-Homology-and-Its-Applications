#!/usr/bin/env python3
"""
阶段 2A：合成动力学验证——延迟嵌入 + PH 批量参数扫描
对 6 类合成时序（周期/拟周期/混沌×2/噪声×2）跑 (m, tau) 参数网格，
统计 H1 持久环数及其稳定性，验证拓扑特征能否区分动力学类型。
"""
import sys
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
from experiment_2a_time_series import GENERATORS, EXPECTED
from experiment_2a_embedding import time_delay_embedding
from ph_pipeline import compute_persistence_diagrams

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "exp2a_synth_scan"
MS = [3, 4, 5, 6]
TAUS = [1, 2, 4, 8]
N_REPEAT = 3


def count_h1_loops(diagrams, min_pers_ratio=0.3):
    """统计 H1 强持久环数（寿命超过最大寿命*min_pers_ratio 的环）"""
    h1 = diagrams[1]
    h1_f = h1[~np.isinf(h1[:, 1])]
    if len(h1_f) == 0:
        return 0, 0.0
    pers = h1_f[:, 1] - h1_f[:, 0]
    if len(pers) == 0:
        return 0, 0.0
    max_pers = pers.max()
    if max_pers <= 0:
        return 0, 0.0
    n_strong = int(np.sum(pers > min_pers_ratio * max_pers))
    return n_strong, max_pers


def main():
    sys.stdout.reconfigure(line_buffering=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("阶段 2A：合成动力学延迟嵌入 + PH 参数扫描")
    print("=" * 70)

    # 每类生成多份带噪声的样本，统计平均环数
    rows = []
    summary = {}  # (sig_type, m, tau) -> (avg_loops, std_loops)

    for sig_type, gen_fn in GENERATORS.items():
        print(f"\n[{sig_type}] 预期: {EXPECTED[sig_type]}")
        base_sig = gen_fn()
        # 归一化到 [0,1]
        base_sig = (base_sig - base_sig.min()) / (base_sig.max() - base_sig.min() + 1e-9)

        type_loops = []
        for m in MS:
            for tau in TAUS:
                loops_list = []
                for rep in range(N_REPEAT):
                    # 加少量噪声（种子可复现）
                    rng = np.random.RandomState(rep)
                    sig = base_sig + 0.05 * rng.randn(len(base_sig))
                    try:
                        pts = time_delay_embedding(sig, m=m, tau=tau)
                        dgms = compute_persistence_diagrams(pts, maxdim=1)
                        n_loops, max_pers = count_h1_loops(dgms)
                        loops_list.append(n_loops)
                    except Exception:
                        loops_list.append(np.nan)
                avg_loops = np.nanmean(loops_list) if any(not np.isnan(x) for x in loops_list) else np.nan
                summary[(sig_type, m, tau)] = avg_loops
                type_loops.append(avg_loops)
                rows.append({"signal": sig_type, "m": m, "tau": tau,
                             "avg_H1_loops": f"{avg_loops:.2f}" if not np.isnan(avg_loops) else "NA"})

        # 打印该类的最佳区分参数
        print(f"  平均环数范围: {min(x for x in type_loops if not np.isnan(x)):.1f} - "
              f"{max(x for x in type_loops if not np.isnan(x)):.1f}")

    # 汇总表
    import pandas as pd
    df = pd.DataFrame(rows)
    csv_path = RESULTS_DIR / "synth_scan_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[已保存] {csv_path}")

    # 可视化：每类在不同 (m,tau) 下的 H1 环数热图
    print("\n[可视化] 生成参数扫描热图...")
    sig_types = list(GENERATORS.keys())
    n_sig = len(sig_types)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, sig_type in enumerate(sig_types):
        ax = axes[i]
        mat = np.zeros((len(MS), len(TAUS)))
        for mi, m in enumerate(MS):
            for ti, tau in enumerate(TAUS):
                v = summary.get((sig_type, m, tau), np.nan)
                mat[mi, ti] = v if not np.isnan(v) else -1
        im = ax.imshow(mat, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(TAUS)))
        ax.set_xticklabels(TAUS)
        ax.set_yticks(range(len(MS)))
        ax.set_yticklabels(MS)
        ax.set_xlabel("tau")
        ax.set_ylabel("m")
        ax.set_title(f"{sig_type}: H1 loops")
        for mi in range(len(MS)):
            for ti in range(len(TAUS)):
                v = mat[mi, ti]
                ax.text(ti, mi, f"{v:.0f}" if v >= 0 else "NA",
                        ha="center", va="center", fontsize=8, color="w")
        plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    fig_path = RESULTS_DIR / "synth_param_scan_heatmaps.png"
    plt.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[已保存] {fig_path}")

    # 核心结论：各类在"最优参数"下的环数区分度
    print("\n" + "=" * 70)
    print("核心结论：各类在最优嵌入参数下的 H1 环数区分度")
    print("=" * 70)
    best_per_type = {}
    for sig_type in sig_types:
        # 找环数最稳定（标准差小）且非 NA 的参数
        vals = [(m, tau, summary[(sig_type, m, tau)]) for m in MS for tau in TAUS
                if not np.isnan(summary.get((sig_type, m, tau), np.nan))]
        if vals:
            # 用中位数附近的参数
            med = np.median([v[2] for v in vals])
            best = min(vals, key=lambda x: abs(x[2] - med))
            best_per_type[sig_type] = best
            print(f"  {sig_type:<14}: m={best[0]}, tau={best[1]}, H1环数={best[2]:.1f}  (预期 {EXPECTED[sig_type]})")

    # 能否区分：周期 vs 噪声 的环数差异
    print("\n[区分度检查]")
    if "periodic" in best_per_type and "noise_white" in best_per_type:
        p = best_per_type["periodic"][2]
        n_ = best_per_type["noise_white"][2]
        print(f"  周期({p:.1f}环) vs 白噪声({n_:.1f}环): "
              f"{'✓ 可区分' if p <= 1.5 and n_ > p + 1 else '⚠ 需改进判据'}")
    print(f"\n✅ 阶段 2A 合成验证完成")


if __name__ == "__main__":
    main()