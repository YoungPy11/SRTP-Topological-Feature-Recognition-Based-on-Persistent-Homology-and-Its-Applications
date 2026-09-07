"""
run_subtask_a.py — 阶段1子任务A：复形参数与计算复杂度分析
============================================================
在同一三维点云上，比较不同子采样比例和复形参数下 PH 主特征的
稳定性，并记录计算时间，形成参数选择建议。

对象：圆、圆盘、环面、球面、多空腔块（5种）
扫描：子采样比例 [1.0, 0.5, 0.25, 0.1] x 复形 maxdim [1, 2]
判据：Bottleneck/Wasserstein 距离 + 持久点统计（数量、最大持久性）
输出：结果图 + 参数建议 -> results/subtask_a/

用法: .venv/Scripts/python.exe run_subtask_a.py
"""
import sys, time, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent / "src"))
from ph_pipeline import (
    generate_circle, generate_disk, generate_torus, generate_sphere,
    generate_porous_block, compute_persistence_diagrams,
    wasserstein_distance, bottleneck_distance,
)

RESULTS = Path(__file__).parent / "results" / "subtask_a"
RESULTS.mkdir(parents=True, exist_ok=True)

# 各对象的生成器与基础点数（控制 CPU 算力）
SHAPES = {
    "circle":  lambda n: generate_circle(n_points=n, radius=1.0, noise=0.05, dim=3),
    "disk":    lambda n: generate_disk(n_points=n, radius=1.0, noise=0.05, dim=3),
    "torus":   lambda n: generate_torus(n_points=n, major_radius=2.0, minor_radius=1.0, noise=0.05),
    "sphere":  lambda n: generate_sphere(n_points=n, radius=1.0, noise=0.05),
    "porous":  lambda n: generate_porous_block(n_points=n, n_cavities=4, box_size=2.0,
                                               cavity_radius=0.35, noise=0.02),
}
BASE_N = 1200          # 全量点数（CPU 可承受的上限附近）
SUBSAMPLES = [1.0, 0.5, 0.25, 0.1]   # 子采样比例
MAXDIMS = [1, 2]                     # 复形最大维数


def subsample(pc, ratio, seed=0):
    """按比例无放回子采样点云。"""
    rng = np.random.default_rng(seed)
    n = max(int(len(pc) * ratio), 10)
    idx = rng.choice(len(pc), size=n, replace=False)
    return pc[idx]


def summarize_pd(dgm):
    """持久图统计：有限持久点数量 + 最大持久性。"""
    finite = np.array([d for d in dgm if d[1] != np.inf], dtype=float)
    if len(finite) == 0:
        return 0, 0.0
    pers = finite[:, 1] - finite[:, 0]
    return len(finite), float(pers.max())


def run():
    rows = []   # 每个配置一行
    for shape_name, gen in SHAPES.items():
        print(f"\n===== 对象: {shape_name} =====")
        # 全量基准 PD
        pc_full = gen(BASE_N)
        dgms_full = compute_persistence_diagrams(pc_full, maxdim=2)
        for ratio in SUBSAMPLES:
            pc_sub = subsample(pc_full, ratio)
            for maxdim in MAXDIMS:
                t0 = time.time()
                dgms = compute_persistence_diagrams(pc_sub, maxdim=maxdim)
                dt = time.time() - t0
                # 用全量 H1/H2 作为参考，比较子采样后的稳定性
                for dim in [0, 1, 2]:
                    if dim > maxdim:
                        continue
                    ref = dgms_full[dim]
                    got = dgms[dim]
                    bd = bottleneck_distance(ref, got)
                    wd = wasserstein_distance(ref, got)
                    n_ref, mx_ref = summarize_pd(ref)
                    n_got, mx_got = summarize_pd(got)
                    rows.append({
                        "shape": shape_name, "ratio": ratio, "maxdim": maxdim,
                        "dim": dim, "n_pts": len(pc_sub),
                        "time_s": round(dt, 4),
                        "bottleneck": round(bd, 6), "wasserstein": round(wd, 6),
                        "n_ref": n_ref, "n_got": n_got,
                        "maxpers_ref": round(mx_ref, 4), "maxpers_got": round(mx_got, 4),
                    })
                print(f"  ratio={ratio:.2f} maxdim={maxdim} 用时{dt:.2f}s")
    return rows


def plot(rows):
    """可视化：按维度画 距离-子采样比例 曲线 + 耗时。"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    shapes = list(SHAPES.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(shapes)))

    for dim, ax in zip([0, 1, 2], axes[0]):
        for si, sh in enumerate(shapes):
            sub = [r for r in rows if r["shape"] == sh and r["dim"] == dim and r["maxdim"] == 2]
            sub = sorted(sub, key=lambda r: r["ratio"])
            if sub:
                ax.plot([r["ratio"] for r in sub], [r["bottleneck"] for r in sub],
                        "o-", color=colors[si], label=sh)
        ax.set_title(f"Bottleneck distance (H{dim})")
        ax.set_xlabel("subsample ratio"); ax.set_ylabel("bottleneck")
        ax.set_xscale("log"); ax.legend(fontsize=8)

    for dim, ax in zip([0, 1, 2], axes[1]):
        for si, sh in enumerate(shapes):
            sub = [r for r in rows if r["shape"] == sh and r["dim"] == dim and r["maxdim"] == 2]
            sub = sorted(sub, key=lambda r: r["ratio"])
            if sub:
                ax.plot([r["ratio"] for r in sub], [r["time_s"] for r in sub],
                        "s-", color=colors[si], label=sh)
        ax.set_title(f"Compute time (H{dim})")
        ax.set_xlabel("subsample ratio"); ax.set_ylabel("seconds")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.legend(fontsize=8)

    plt.tight_layout()
    f = RESULTS / "subtask_a_distances.png"
    plt.savefig(f, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"已保存: {f}")


def advice(rows):
    """生成参数选择建议：对每个对象/维度，指出稳定且快的配置。"""
    print("\n===== 参数选择建议 =====")
    for shape in SHAPES:
        for dim in [0, 1, 2]:
            sub = [r for r in rows if r["shape"] == shape and r["dim"] == dim and r["maxdim"] == 2]
            if not sub:
                continue
            sub = sorted(sub, key=lambda r: r["ratio"])
            best = min(sub, key=lambda r: r["bottleneck"] + 0.5 * r["time_s"])
            print(f"  {shape} H{dim}: 推荐 ratio={best['ratio']:.2f} "
                  f"(bd={best['bottleneck']:.4f}, {best['time_s']:.2f}s, "
                  f"n={best['n_got']}/{best['n_ref']})")


if __name__ == "__main__":
    rows = run()
    with open(RESULTS / "subtask_a_data.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    plot(rows)
    advice(rows)
    print(f"\n全部完成，数据与图已存到 {RESULTS}/")