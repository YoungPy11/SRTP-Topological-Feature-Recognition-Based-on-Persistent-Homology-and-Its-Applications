"""
====================================================================
experiment_1a_param_scan.py — 阶段1子任务A：复形参数与计算复杂度扫描
====================================================================
目标：对同一三维点云，在不同子采样比例与复形参数（max_dimension）下，
      比较持续同调主特征是否稳定，并记录计算时间，形成参数选择建议。

对应申报书拟解决问题①「复形构建的计算复杂度优化」。

对象：circle / disk / torus / sphere / porous_block（多孔块体，含 H2 空腔）
因素：子采样比例 ∈ {100%, 50%, 25%, 10%}
       max_dimension ∈ {1, 2}   (H0/H1/H2，不做 H3)
判据：Bottleneck 距离 + Wasserstein 距离（对基准 PD）+ 持久点统计 + 耗时

用法: python experiment_1a_param_scan.py
====================================================================
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(SRC_DIR))
from ph_pipeline import (
    generate_circle, generate_disk, generate_torus, generate_sphere,
    generate_porous_block, compute_persistence_diagrams,
    wasserstein_distance, bottleneck_distance,
)

RESULTS_DIR = Path(__file__).parent.parent / "results" / "exp1a_param_scan"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_POINTS = 800           # 基准采样点数（调小以在 CPU 上快速跑完；GPU 对 ripser 无加速）
SUB_SAMPLES = [1.0, 0.5, 0.25, 0.1]   # 子采样比例
MAX_DIMS = [1, 2]        # 复形维度上限（H0/H1/H2）
N_TRIALS = 2             # 每组合重复次数（消减随机性，取均值）

# 每个对象需要验证的同调维度上限（circle/disk 无高维空腔，算 H1 即可，避免浪费 CPU）
# torus 有 H1 环；sphere/porous_block 有 H2 空腔
OBJ_MAX_DIM = {
    "circle": 1,
    "disk": 1,
    "torus": 2,
    "sphere": 2,
    "porous_block": 2,
}
GENERATORS = {
    "circle":       lambda: generate_circle(n_points=N_POINTS, noise=0.05),
    "disk":         lambda: generate_disk(n_points=N_POINTS, noise=0.05),
    "torus":        lambda: generate_torus(n_points=N_POINTS, noise=0.05),
    "sphere":       lambda: generate_sphere(n_points=N_POINTS, noise=0.05),
    "porous_block": lambda: generate_porous_block(n_points=N_POINTS, n_cavities=4, noise=0.02),
}


def finite_persistence_stats(dgm):
    """返回 (有限点数, 最大持久性, 平均持久性) 三个统计量。"""
    d = dgm[~np.isinf(dgm[:, 1])]
    if len(d) == 0:
        return (0, 0.0, 0.0)
    pers = d[:, 1] - d[:, 0]
    return (len(d), float(pers.max()), float(pers.mean()))


def main():
    print("=" * 60)
    print("阶段1子任务A：复形参数与计算复杂度扫描")
    print("=" * 60)

    rows = []  # 收集统计行

    for obj_name, gen in GENERATORS.items():
        print(f"\n--- 对象: {obj_name} ---")
        base_pts = gen()
        obj_max_dim = OBJ_MAX_DIM[obj_name]
        dims_to_run = list(range(1, obj_max_dim + 1))  # 该对象要验证的维度（从 H1 起）
        print(f"  基准点云: {base_pts.shape[0]} 点, 验证维度 H1..H{obj_max_dim}")

        # 基准：100% 采样, 该对象的最大维度
        t0 = time.perf_counter()
        base_dgms = compute_persistence_diagrams(base_pts, maxdim=obj_max_dim)
        base_time = time.perf_counter() - t0
        print(f"  基准 PD (max_dim={obj_max_dim}): {base_time:.3f}s "
              f"[H1 {len(base_dgms[1])}点, "
              + (f"H2 {len(base_dgms[2])}点" if obj_max_dim >= 2 else "") + "]")

        # 为每对象单独布局一张图
        fig, axes = plt.subplots(2, len(dims_to_run), figsize=(11, 9))
        axes = np.atleast_2d(axes)

        for mi, md in enumerate(dims_to_run):
            # 记录该 max_dim 下的数据（横轴：子采样比例）
            wd_arr, bd_arr, npts_arr, maxpers_arr, t_arr = [], [], [], [], []

            for sub in SUB_SAMPLES:
                n_sub = max(30, int(N_POINTS * sub))
                # 多次重复取均值
                wd_sum = bd_sum = t_sum = 0.0
                npts_all, maxpers_all = [], []
                for _ in range(N_TRIALS):
                    idx = np.random.RandomState(42).choice(N_POINTS, n_sub, replace=False)
                    pts_sub = base_pts[idx]
                    t0 = time.perf_counter()
                    sub_dgms = compute_persistence_diagrams(pts_sub, maxdim=md)
                    t_sum += time.perf_counter() - t0
                    # 稳定性：与基准同维 PD 的 B/W 距离
                    wd_sum += wasserstein_distance(sub_dgms[md], base_dgms[md])
                    bd_sum += bottleneck_distance(sub_dgms[md], base_dgms[md])
                    npts_all.append(finite_persistence_stats(sub_dgms[md])[0])
                    maxpers_all.append(finite_persistence_stats(sub_dgms[md])[1])

                wd = wd_sum / N_TRIALS
                bd = bd_sum / N_TRIALS
                t_avg = t_sum / N_TRIALS
                npts = np.mean(npts_all)
                maxpers = np.mean(maxpers_all)

                wd_arr.append(wd)
                bd_arr.append(bd)
                npts_arr.append(npts)
                maxpers_arr.append(maxpers)
                t_arr.append(t_avg)

                # 记录到总表
                rows.append({
                    "object": obj_name, "max_dim": md,
                    "subsample_ratio": sub, "n_points": n_sub,
                    "bottleneck": bd, "wasserstein": wd,
                    "n_pers_points": npts, "max_persistence": maxpers,
                    "time_s": t_avg,
                })
                print(f"  max_dim={md} 子采样{sub*100:5.0f}%: "
                      f"BD={bd:.4f} WD={wd:.4f} H{md}点数={npts:.0f} 时间={t_avg:.3f}s")

            # 绘图：横轴 = 子采样比例
            ax = axes[mi, 0]
            x = [s * 100 for s in SUB_SAMPLES]
            ax2 = ax.twinx()
            ax.plot(x, wd_arr, "o-", color="tab:blue", label="Wasserstein")
            ax.plot(x, bd_arr, "s--", color="tab:red", label="Bottleneck")
            ax2.bar(x, t_arr, alpha=0.3, color="gray", label="time")
            ax.set_title(f"{obj_name} | max_dim={md}")
            ax.set_xlabel("子采样比例 (%)")
            ax.set_ylabel("距离")
            ax2.set_ylabel("时间 (s)")
            ax.legend(loc="upper left")

            # 持久点统计
            ax = axes[mi, 1]
            ax.plot(x, npts_arr, "o-", color="tab:green", label="H%d点数" % md)
            ax.plot(x, maxpers_arr, "s--", color="tab:purple", label="最大持久性")
            ax.set_title(f"{obj_name} | max_dim={md} 持久点统计")
            ax.set_xlabel("子采样比例 (%)")
            ax.legend(loc="best")

        fig.tight_layout()
        fname = RESULTS_DIR / f"{obj_name}_param_scan.png"
        fig.savefig(fname, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  [已保存] {fname}")

    # 汇总表
    df = pd.DataFrame(rows)
    csv_path = RESULTS_DIR / "param_scan_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[已保存] 汇总表: {csv_path}")

    # 参数选择建议：每个对象，找到"稳定性下降可接受"的最小子采样
    print("\n" + "=" * 60)
    print("参数选择建议（基于稳定性与耗时权衡）")
    print("=" * 60)
    for obj_name in GENERATORS:
        obj_max_dim = OBJ_MAX_DIM[obj_name]
        sub = df[(df.object == obj_name) & (df.max_dim == obj_max_dim)]
        if len(sub) == 0:
            print(f"  {obj_name:12s}: （无数据）")
            continue
        # 基准（100% 采样）的 WD
        base_row = sub[sub.subsample_ratio >= 0.999]  # 避免浮点比较误差
        if len(base_row) == 0:
            base_wd = sub.wasserstein.iloc[0]  # 兜底
        else:
            base_wd = base_row.wasserstein.values[0]
        rel = sub.wasserstein / max(base_wd, 1e-9)
        cand = sub[rel < 1.5]
        if len(cand) > 0:
            rec = cand.subsample_ratio.min() * 100
            note = f"建议子采样 ≥ {rec:.0f}%（WD 波动 < 50%）"
        else:
            note = "各子采样下 WD 波动较大，建议保持高密度"
        print(f"  {obj_name:12s}: {note}")

    print("\n✅ 子任务A 参数扫描完成")


if __name__ == "__main__":
    main()