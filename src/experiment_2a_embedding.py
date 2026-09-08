#!/usr/bin/env python3
"""
阶段 2A 工具：延迟嵌入（Takens 嵌入）+ PH 管道
时序 -> 相空间点云 -> 持续同调 -> 持续图/特征

嵌入参数选取逻辑：
- 嵌入维数 m：根据 Taken 定理，需 m > 2*dim(吸引子)+1。常用 m 试探范围 2-10。
- 延迟 tau：用互信息第一极小 或 自相关首过零点（提供启发式）。
- 最终取一组 (m, tau) 扫描，比较 PH 的稳定性。
"""
import numpy as np
from pathlib import Path
import sys
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ph_pipeline import compute_persistence_diagrams, compute_top_k_persistences


def time_delay_embedding(signal: np.ndarray, m: int = 3, tau: int = 5) -> np.ndarray:
    """Takens 延迟嵌入：signal(n,) -> 相空间点云 (n-(m-1)*tau, m)
    x_t = [s_t, s_{t+tau}, s_{t+2*tau}, ..., s_{t+(m-1)*tau}]
    """
    signal = np.asarray(signal).reshape(-1)
    n = len(signal)
    n_pts = n - (m - 1) * tau
    if n_pts <= 0:
        raise ValueError(f"嵌入参数过大: m={m}, tau={tau}, n={n}")
    # 用 stride 构造嵌入矩阵
    idx = np.arange(n_pts)[:, None] + np.arange(m)[None, :] * tau
    return signal[idx]


def suggest_embedding_params(signal: np.ndarray, max_m: int = 10) -> dict:
    """启发式选择嵌入参数：
    - tau: 互信息第一极小（用分箱近似）或自相关首过零点
    - m: 取 tau 后，用假近邻比/FNN 简化版
    """
    signal = np.asarray(signal).reshape(-1)
    n = len(signal)

    # tau via 自相关首过零点（简单可靠）
    ac = np.correlate(signal - signal.mean(), signal - signal.mean(), mode="full")[n - 1:]
    ac /= ac[0]
    tau_ac = 1
    for t in range(1, n // 3):
        if ac[t] <= 0:
            tau_ac = t
            break

    # tau via 互信息第一极小（分箱近似）
    tau_mi = 1
    try:
        nbins = 20
        hist2d_s, _, _ = np.histogram2d(signal[:-1], signal[1:], bins=nbins)
        p_joint = hist2d_s / hist2d_s.sum()
        p_x = p_joint.sum(axis=1)
        p_y = p_joint.sum(axis=0)
        mi_1 = 0.0
        for i in range(nbins):
            for j in range(nbins):
                if p_joint[i, j] > 0:
                    mi_1 += p_joint[i, j] * np.log2(p_joint[i, j] / (p_x[i] * p_y[j] + 1e-12))
        # 扫描后续 tau 找第一极小
        for tau in range(1, min(30, n // 3)):
            sig2 = np.hstack([signal[tau:], signal[:tau]])  # 环形移位近似
            h2, _, _ = np.histogram2d(signal, sig2, bins=nbins)
            pj = h2 / h2.sum()
            px = pj.sum(axis=1)
            py = pj.sum(axis=0)
            mi_t = 0.0
            for i in range(nbins):
                for j in range(nbins):
                    if pj[i, j] > 0:
                        mi_t += pj[i, j] * np.log2(pj[i, j] / (px[i] * py[j] + 1e-12))
            if mi_t < mi_1:  # 第一极小
                tau_mi = tau
                break
    except Exception:
        pass

    # m: 简单启发式（相空间维数 ~ 信号频率成分数 + 2）
    m_sug = max(3, min(max_m, 4))
    return {"tau_ac": tau_ac, "tau_mi": tau_mi, "m": m_sug}


def time_series_ph(signal: np.ndarray, m: int, tau: int, maxdim: int = 1) -> np.ndarray:
    """时序 -> 嵌入 -> PH，返回持续图"""
    pts = time_delay_embedding(signal, m=m, tau=tau)
    return compute_persistence_diagrams(pts, maxdim=maxdim)


if __name__ == "__main__":
    from experiment_2a_time_series import gen_periodic, gen_chaos_rossler, gen_noise
    print("延迟嵌入测试：")
    for name, sig in [("periodic", gen_periodic()), ("chaos", gen_chaos_rossler()), ("noise", gen_noise())]:
        params = suggest_embedding_params(sig)
        pts = time_delay_embedding(sig, m=params["m"], tau=params["tau_mi"])
        dgms = compute_persistence_diagrams(pts, maxdim=1)
        # 统计 H1 持久点
        h1 = dgms[1]
        h1_finite = h1[~np.isinf(h1[:, 1])]
        pers = h1_finite[:, 1] - h1_finite[:, 0]
        n_strong = int(np.sum(pers > 0.5 * pers.max())) if len(pers) > 0 else 0
        print(f"  {name:<10} 参数(m={params['m']},tau={params['tau_mi']}), "
              f"相空间点数={len(pts)}, H1强持久环数={n_strong}")