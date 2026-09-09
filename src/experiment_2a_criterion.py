#!/usr/bin/env python3
"""
阶段 2A：持久环判据优化
对 6 类动力学在统一嵌入参数 (m=4, tau=4) 下计算 PH，保存持久图，
离线比较多种判据的特征（环数/top-k持久度/分位数），选区分度最好的。

关键设计：PH 计算与判据分析解耦——先存持久图，判据可反复离线测试。
"""
import sys
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment_2a_time_series import GENERATORS
from experiment_2a_embedding import time_delay_embedding
from ph_pipeline import compute_persistence_diagrams

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "exp2a_criterion"
M_FIXED = 4
TAU_FIXED = 4
N_SAMPLES = 10
NOISE_LEVEL = 0.05


def main():
    sys.stdout.reconfigure(line_buffering=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"阶段 2A 判据优化：统一 (m={M_FIXED}, tau={TAU_FIXED})，{N_SAMPLES} 样本/类")
    print("=" * 70)

    all_diagrams = {}  # sig_type -> list of H1 (n,2) arrays
    for sig_type, gen_fn in GENERATORS.items():
        base = gen_fn()
        base = (base - base.min()) / (base.max() - base.min() + 1e-9)
        h1_dgms = []
        for rep in range(N_SAMPLES):
            rng = np.random.RandomState(rep)
            sig = base + NOISE_LEVEL * rng.randn(len(base))
            pts = time_delay_embedding(sig, m=M_FIXED, tau=TAU_FIXED)
            dgms = compute_persistence_diagrams(pts, maxdim=1)
            h1 = dgms[1]
            h1_f = h1[~np.isinf(h1[:, 1])]
            h1_dgms.append(h1_f)
        all_diagrams[sig_type] = h1_dgms
        print(f"  [{sig_type}] 完成 {N_SAMPLES} 个 H1 持久图")

    # 保存持久图（供离线复用）
    np.savez(RESULTS_DIR / "h1_diagrams.npz",
             **{st: np.concatenate([d for d in dd]) for st, dd in all_diagrams.items()})
    print(f"[已保存] {RESULTS_DIR / 'h1_diagrams.npz'}")

    # ============ 判据测试 ============
    print("\n" + "=" * 70)
    print("候选判据在各动力学类型下的分布（均值±std）")
    print("=" * 70)

    # 判据定义（每个返回标量特征）
    def crit_count_ratio(dgm, ratio=0.3):
        """当前判据：寿命>max*ratio 的环数"""
        if len(dgm) == 0:
            return 0
        pers = dgm[:, 1] - dgm[:, 0]
        mx = pers.max()
        if mx <= 0:
            return 0
        return int(np.sum(pers > ratio * mx))

    def crit_top1_pers(dgm):
        """top-1 环持久度（绝对）"""
        if len(dgm) == 0:
            return 0.0
        return float((dgm[:, 1] - dgm[:, 0]).max())

    def crit_top2_pers(dgm):
        """top-1 + top-2 环持久度之和"""
        if len(dgm) == 0:
            return 0.0
        pers = np.sort(dgm[:, 1] - dgm[:, 0])[::-1]
        return float(pers[:2].sum())

    def crit_top1_over_mean(dgm):
        """top-1持久度 / 平均持久度（比值，归一化尺度）"""
        if len(dgm) == 0:
            return 0.0
        pers = dgm[:, 1] - dgm[:, 0]
        if pers.mean() <= 0:
            return 0.0
        return float(pers.max() / pers.mean())

    def crit_count_abs(dgm, scale=0.2):
        """持久度 > scale × 相空间直径 的环数（绝对阈值）"""
        if len(dgm) == 0:
            return 0
        pers = dgm[:, 1] - dgm[:, 0]
        # 相空间直径近似：嵌入点云各维范围的最大跨度
        return int(np.sum(pers > scale))

    criteria = {
        "环数(比例0.3)": crit_count_ratio,
        "top1持久度": crit_top1_pers,
        "top1+2持久度": crit_top2_pers,
        "top1/均值": crit_top1_over_mean,
        "环数(绝对0.2)": lambda d: crit_count_abs(d, 0.2),
        "环数(绝对0.1)": lambda d: crit_count_abs(d, 0.1),
    }

    # 计算各判据分布
    dist = {}
    for cname, cfn in criteria.items():
        dist[cname] = {}
        for sig_type, dd in all_diagrams.items():
            vals = [cfn(d) for d in dd]
            dist[cname][sig_type] = (np.mean(vals), np.std(vals))
        # 打印该判据下所有类的分布
        line = "  ".join(f"{st[:6]}:{dist[cname][st][0]:6.2f}±{dist[cname][st][1]:5.2f}"
                         for st in GENERATORS)  # noqa
        print(f"  {cname:<18}: {line}")

    # ============ 区分度评估 ============
    print("\n" + "=" * 70)
    print("判据区分度评估（周期 vs 噪声 的类间/类内比）")
    print("=" * 70)
    best = None
    best_ratio = 0
    for cname, cdict in dist.items():
        # 周期 vs 白噪声：类间距离 / 类内平均std
        mu_p, sd_p = cdict["periodic"]
        mu_n, sd_n = cdict["noise_white"]
        denom = (sd_p + sd_n) / 2 + 1e-9
        ratio = abs(mu_n - mu_p) / denom
        sep = "✓ 区分好" if ratio > 3 else ("△ 可区分" if ratio > 1.5 else "✗ 区分弱")
        print(f"  {cname:<18}: (噪声-周期)/std = {ratio:6.2f}  {sep}")
        if ratio > best_ratio:
            best_ratio = ratio
            best = cname
    print(f"\n  最佳判据: {best} (区分度 {best_ratio:.2f})")

    # 混沌 vs 噪声（更难的任务）
    print("\n[难任务] 混沌(rossler) vs 白噪声 区分度：")
    for cname, cdict in dist.items():
        mu_c, sd_c = cdict["chaos_rossler"]
        mu_n, sd_n = cdict["noise_white"]
        denom = (sd_c + sd_n) / 2 + 1e-9
        ratio = abs(mu_n - mu_c) / denom
        sep = "✓" if ratio > 3 else ("△" if ratio > 1.5 else "✗")
        print(f"  {cname:<18}: {ratio:6.2f}  {sep}")


if __name__ == "__main__":
    main()