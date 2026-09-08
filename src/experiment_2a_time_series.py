#!/usr/bin/env python3
"""
阶段 2A 工具：合成动力学时间序列生成器
生成 4 类可控动力学：周期 / 拟周期 / 混沌 / 纯噪声
用于验证"延迟嵌入 → PH"能否区分布不同动力学类型。
"""
import numpy as np


def gen_periodic(n=2000, freq=2.0, amp=1.0, phase=0.0):
    """周期信号：sin 单频"""
    t = np.linspace(0, 20, n, endpoint=False)
    return amp * np.sin(2 * np.pi * freq * t + phase)


def gen_quasiperiodic(n=2000, f1=2.0, f2=3.5, amp=1.0):
    """拟周期信号：两不可公度频率叠加（环面拓扑）"""
    t = np.linspace(0, 20, n, endpoint=False)
    return amp * (np.sin(2 * np.pi * f1 * t) + 0.6 * np.sin(2 * np.pi * f2 * t))


def gen_chaos_rossler(n=2000, dt=0.05, a=0.2, b=0.2, c=5.7):
    """混沌：Rössler 系统（x 分量），单频不可预测"""
    x, y, z = 0.0, 0.1, 0.0
    xs = []
    for _ in range(n):
        dx = -y - z
        dy = x + a * y
        dz = b + z * (x - c)
        x += dx * dt
        y += dy * dt
        z += dz * dt
        xs.append(x)
    return np.array(xs)


def gen_chaos_lorenz(n=2000, dt=0.01, sigma=10.0, rho=28.0, beta=8 / 3):
    """混沌：Lorenz 系统（x 分量）"""
    x, y, z = 1.0, 1.0, 1.0
    xs = []
    for _ in range(n):
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        x += dx * dt
        y += dy * dt
        z += dz * dt
        xs.append(x)
    return np.array(xs)


def gen_noise(n=2000, sigma=1.0):
    """纯高斯噪声"""
    return sigma * np.random.randn(n)


def gen_voltage_noise(n=2000, sigma=1.0):
    """自回归噪声 AR(1)：红噪声，有短暂记忆但无确定性"""
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.8 * x[i - 1] + sigma * np.random.randn()
    return x


GENERATORS = {
    "periodic": gen_periodic,
    "quasiperiodic": gen_quasiperiodic,
    "chaos_rossler": gen_chaos_rossler,
    "chaos_lorenz": gen_chaos_lorenz,
    "noise_white": gen_noise,
    "noise_ar1": gen_voltage_noise,
}

# 期望的拓扑/动力学特征（用于结果解释）
EXPECTED = {
    "periodic": "H1 x 1 (单环，延迟嵌入为圆)",
    "quasiperiodic": "H1 x 2 (两环，环面)",
    "chaos_rossler": "H1 x 1 (吸引子，单环但有细结构)",
    "chaos_lorenz": "H1 x 1 (蝴蝶吸引子，单环)",
    "noise_white": "无显著持久环（噪声占主导）",
    "noise_ar1": "弱持久环（红噪声可能伪影）",
}


if __name__ == "__main__":
    print("合成时序生成器测试：")
    for name, fn in GENERATORS.items():
        sig = fn()
        print(f"  {name:<14} 长度 {len(sig)}, 范围 [{sig.min():.2f}, {sig.max():.2f}]")
        print(f"                预期: {EXPECTED[name]}")