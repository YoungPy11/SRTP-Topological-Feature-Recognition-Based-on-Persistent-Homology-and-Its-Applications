# 基于持续同调的拓扑特征识别及其应用

> 浙江大学大学生创新训练项目（SRTP）· 2026.03 – 2027.05

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 项目简介

计算拓扑是21世纪兴起的、基于代数拓扑的应用数学分支，融合微分几何学和计算机科学，展现出深刻的理论价值与广阔的应用前景。它突破了传统方法对坐标与度量的依赖，为理解复杂数据的内在拓扑结构提供了全新视角。

本项目以欧氏空间中的**持续同调（Persistent Homology）**为起点，系统研究其向更复杂环境的推广方法，力图弥补传统算法在坐标系依赖性及海量特征提取能力上的不足。通过理论推导与实验验证，为数据科学、几何设计与智能分析提供新思路。

### 核心工具

持续同调能够从点云、图像、网络等数据中提取多尺度的拓扑不变量（如连通分量、环、空洞等），且具有**稳定性**——当数据受噪声或采样扰动时，所得到的持续图在瓶颈距离下变化很小。

## 项目目标

1. **基础理论深度构建**：研究度量空间向单纯复形的离散映射机制，掌握同调群的代数构造，深入理解持续同调算法中过滤与持续图生成的数学本质。
2. **三维空间拓扑识别实践**：通过构建复形提取三维物体的零维（连通分支）、一维（环路）及二维（空腔）拓扑特征，验证 TDA 在处理三维物体旋转不变性、噪声鲁棒性方面的优势。
3. **高维抽象数据的拓扑拓展**：将技术栈拓展至高维参数空间，探索时间序列等抽象数据的拓扑特征提取策略，研究延迟嵌入与持续景观等向量化方法。

## 快速开始

### 环境安装

```bash
git clone https://github.com/YoungPy11/SRTP-Topological-Feature-Recognition-Based-on-Persistent-Homology-and-Its-Applications.git
cd "基于持续同调的拓扑特征识别及其应用"
pip install -r requirements.txt
```

### 目录结构

```
基于持续同调的拓扑特征识别及其应用/
├── README.md              # 项目说明
├── PROGRESS.md            # 全程进度时间线（实验记录，逐 commit 对应）
├── STAGE1_SUMMARY.md      # 阶段 1 总结（三维拓扑识别）
├── STAGE2AB_SUMMARY.md    # 阶段 2A/2B 总结（时序表征 + 向量化研究）
├── STAGE2C_SUMMARY.md     # 阶段 2C 总结（可学习向量化，三数据集）
├── requirements.txt       # Python 依赖
├── run_all_pipelines.py   # 批量运行参考论文 pipeline 复现（--paper 1~5 / all）
├── .gitignore             # Git 忽略规则（大数据集 data/ 不入库）
├── LICENSE                # MIT 许可证
├── docs/                  # 理论笔记 + 实验设计文档（2A 嵌入参数 / 2C 提纲 / 导师讨论材料 / 结题大纲）
├── src/                   # 核心代码（见下方"实验体系"）
├── notebooks/             # Jupyter Notebook 实验
├── data/                  # 数据集（gitignore：modelnet10 2.2G / ecg200 / forda 125M / ecg5000）
└── results/               # 实验结果（图表、CSV、JSON），按 expXX_ 前缀分目录
```

### 核心代码（`src/`）

**基础模块**：
- **`ph_pipeline.py`**：持续同调工具模块，含点云生成（圆/圆盘/环面/球面/多空腔）、PH 计算（ripser）、四种向量化方法（top-k 持续特征 / 持续图像 / 持续景观 / Betti 曲线）、显著特征选择（TopoGAT 软掩码等）、加权 VR 复形、B/W 距离评估、批量特征提取与数据集划分。
- **`perslay_lite.py`**：PersLay-lite 可微向量化层（阶段 2C）——可学习权重 sigmoid(a·pers+b) + 可学习带宽高斯核，PD→网格特征，支持二分类/多分类。
- **`pipeline_paper1~5_*.py`**：五篇参考论文的 pipeline 复现（DNA 结构分析、喷注标记、TopoGAT、3DPHDL、过滤学习）。其中 paper2/3/4/5 的 GNN 部分需要 `torch`。

**实验脚本**（按阶段编号，均可独立重跑，随机种子固定）：

| 阶段 | 脚本 | 实验 |
|------|------|------|
| 1A | `experiment_1a_param_scan.py` + `experiment_1a_threshold_sensitivity.py` | 复形参数扫描 + WD 阈值稳健性 |
| 1B | `experiment_1b_topo_vs_geo.py` | 拓扑 vs 几何（SO(3) 旋转公平协议） |
| 1C | `experiment_1c_modelnet10.py` | ModelNet10 真实三维分类（GroupKFold，`--no-cache`） |
| 2A | `experiment_2a_time_series.py` / `experiment_2a_embedding.py` / `experiment_2a_synth_scan.py` / `experiment_2a_criterion.py` / `experiment_2a_ecg200.py` / `experiment_2a_ecg200_boost.py` | 合成时序生成 → 延迟嵌入 → 参数扫描 → 判据优化 → ECG200 实证 + 85% 冲刺 |
| 2B | `experiment_2b_vectorization_stability.py` / `experiment_2b_vectorization_rotation.py` / `experiment_2b_vectorization_ecg200.py` | 四种向量化：噪声稳定性 / 旋转连续性 / 分类效果 |
| 2C | `experiment_2c_ecg200_perslay.py` / `experiment_2c_perslay_refine.py` / `experiment_2c_forda.py` / `experiment_2c_ecg5000.py` / `experiment_2c_stability_ecg5000.py` | PersLay-lite：ECG200 首轮 → 定向修复 → FordA 数据量检验 → ECG5000 任务相关性检验 → 多种子稳定性确认 |

### 实验体系与核心结果

| 结论 | 证据 |
|------|------|
| 拓扑特征旋转不变优势 | 合成 5 类 SO(3) 旋转下：拓扑 91.1% vs 几何 71.4% |
| 拓扑特征的补充价值 | ModelNet10 融合 +6.7pp（76.8% vs 70.1%）；ECG200 融合 +5~7pp |
| 申报书 85% 目标 | **ECG200 融合 85.0% 达标**（严格官方划分） |
| 时序拓扑表征可行 | 周期 1 环 vs 噪声 531 环；比例判据区分度 9.12 |
| 稳定性 ≠ 判别力 | PI 抗噪最好但分类不如 Top-K；PL 最连续但 test 弱 |
| 可学习向量化双重门槛 | 数据量（100→3601 差距 −7pp→−1.6pp）× 任务拓扑相关性（ECG5000 纯拓扑 87-90%，混合 92.0% 略优固定 91.4%，5/5 种子） |
| 小样本稳健范式 | "可学习表征 + RF" 三数据集一律优于端到端 MLP |

详细数字与图表索引见三份 STAGE 总结文档与 `docs/导师讨论材料_阶段2成果与创新点.md`。

## 理论笔记

项目组的理论学习笔记位于 `docs/` 目录，目前包含：

- **CTDA阅读笔记理论部分整合**：《Computational Topology for Data Analysis》中单纯复形、Čech/Rips 复形、同调群、滤过、持续同调、持续图、持续模、PL-函数等核心理论的整理。这部分基本思路和框架由人类提供，细节补全由deepseek完成，人类大致检查了deepseek的工作，质量尚可。建议最好还是配合教材原著食用。
- **CTDA第三章算法部分**：主要是第三章的五个算法，完全由deepseek整理。质量欠佳，建议阅读教材。
- **CTDA理论部分重点梳理（手写）**：《CTDA阅读笔记理论部分整合》的进一步精简和总结，建议学习完后再酌情参考。
- **CTDA算法部分重点梳理（手写）**：抄书并稍作梳理，同样，酌情参考。
- **各个时间点的总结与规划**:参考txt文件。

## 进度安排

| 阶段 | 计划时间 | 内容 | 实际状态（2026.09.10） |
|------|------|------|------|
| 理论夯实与环境搭建 | 2026.03 – 2026.07 | 研读核心文献，配置 Python 拓扑计算环境 | ✅ 完成（5 篇论文 pipeline 复现，WSL+CUDA 环境） |
| 三维场景拓扑识别 | 2026.07 – 2026.11 | 实现三维点云持续同调计算，完成初步模型训练 | ✅ **提前完成**（A/B/C 三子任务，见 STAGE1_SUMMARY） |
| 高维拓展与向量化 | 2026.11 – 2027.02 | 时间序列拓扑表征，持续图向量化，深度学习融合 | ✅ **提前完成**（2A/2B/2C，ECG200 达 85%，见 STAGE2AB/2C_SUMMARY） |
| 成果总结与结题 | 2027.02 – 2027.05 | 整理实验数据，撰写结题报告，准备答辩 | 🔄 进行中（结题大纲 v1 已出：docs/结题报告大纲_v1.md） |

## 许可证

本项目采用 [MIT License](LICENSE) 开源。
