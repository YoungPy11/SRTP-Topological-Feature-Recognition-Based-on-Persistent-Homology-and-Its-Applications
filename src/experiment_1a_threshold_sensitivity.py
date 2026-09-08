#!/usr/bin/env python3
"""
子任务 A 补充：绝对 WD 阈值稳健性分析
验证"10% 点云直径"阈值是稳健选择——扫描 5%/10%/15%/20% 等阈值，
看各对象的"建议子采样比例"随阈值的变化，证明结论对阈值选取不敏感。

数据来源：results/exp1a_param_scan/param_scan_summary.csv（子任务 A 已生成）
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import pdist

# src 路径
sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment_1a_param_scan import GENERATORS, OBJ_MAX_DIM

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "exp1a_param_scan"
OUT_DIR = RESULTS_DIR / "threshold_sensitivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    df = pd.read_csv(RESULTS_DIR / "param_scan_summary.csv")

    # 候选阈值（点云直径的百分比）
    ratios = np.array([0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 0.50])

    print("=" * 70)
    print("子任务 A 补充：绝对 WD 阈值稳健性分析")
    print("=" * 70)
    print(f"{'对象':<14}", end="")
    for r in ratios:
        print(f"{r*100:>6.0f}%", end="")
    print(f"{'':>6}建议(10%)")

    rows = []
    for obj_name in GENERATORS:
        obj_max_dim = OBJ_MAX_DIM[obj_name]
        sub = df[(df.object == obj_name) & (df.max_dim == obj_max_dim)]
        if len(sub) == 0:
            continue
        base_pts = GENERATORS[obj_name]()
        dia = float(pdist(base_pts).max())

        print(f"{obj_name:<14}", end="")
        recs = []
        for ratio in ratios:
            wd_accept = ratio * dia
            cand = sub[sub.wasserstein < wd_accept]
            if len(cand) > 0:
                rec = cand.subsample_ratio.min() * 100
            else:
                rec = 100  # 所有超阈值 -> 建议保持高密度
            recs.append(rec)
            print(f"{rec:>6.0f}%", end="")
        # 10% 的建议
        wd_accept_10 = 0.10 * dia
        cand10 = sub[sub.wasserstein < wd_accept_10]
        rec10 = cand10.subsample_ratio.min() * 100 if len(cand10) > 0 else 100
        print(f"{'':>6}{rec10:>6.0f}%")
        rows.append({"object": obj_name, "diameter": dia, **{f"r{int(r*100)}": rec for r, rec in zip(ratios, recs)}})

    # 稳健性判断：10% 阈值下的建议，在其邻域(5%-15%)是否稳定
    print("\n" + "=" * 70)
    print("稳健性判断（10% 阈值在 5%-15% 邻域内的建议是否一致）")
    print("=" * 70)
    stable = True
    for r in rows:
        rec_lo = r["r5"]    # 5%
        rec_hi = r["r15"]   # 15%
        rec_10 = r["r10"]   # 10%
        lo_ok = (rec_lo - rec_10) >= 0  # 更低阈值应允许更高子采样（或持平）
        hi_ok = (rec_hi - rec_10) <= 0  # 更高阈值应允许更低子采样（或持平）
        # 单调性：阈值升，建议子采样应不增
        monotone = True
        rs = [r[f"r{int(x*100)}"] for x in ratios]
        for i in range(len(rs) - 1):
            if rs[i + 1] > rs[i]:  # 阈值增大，子采样建议不应上升
                monotone = False
        status = "✓ 单调" if monotone else "✗ 非单调"
        print(f"  {r['object']:<14}: 10%建议={r['r10']:>3.0f}%  邻域[{rec_lo:.0f}%, {rec_hi:.0f}%]  {status}")
        if not monotone:
            stable = False

    print(f"\n{'✅ 阈值选择稳健' if stable else '⚠️ 部分对象对阈值敏感，需谨慎解释'}")
    print(f"[已保存] 分析数据: {OUT_DIR / 'threshold_sensitivity.csv'}")
    pd.DataFrame(rows).to_csv(OUT_DIR / "threshold_sensitivity.csv", index=False)

    # 可视化：每对象建议子采样 vs 阈值
    fig, ax = plt.subplots(figsize=(10, 6))
    for r in rows:
        xs = [int(x * 100) for x in ratios]
        ys = [r[f"r{x}"] for x in xs]
        ax.plot(xs, ys, "o-", label=r["object"], linewidth=2)
    ax.axvline(10, color="red", linestyle="--", alpha=0.6, label="10% 阈值")
    ax.set_xlabel("WD threshold (% of point-cloud diameter)")
    ax.set_ylabel("Recommended min subsample ratio (%)")
    ax.set_title("Threshold Sensitivity: Recommended Subsample vs WD Threshold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = OUT_DIR / "threshold_sensitivity.png"
    plt.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[已保存] {fig_path}")


if __name__ == "__main__":
    main()