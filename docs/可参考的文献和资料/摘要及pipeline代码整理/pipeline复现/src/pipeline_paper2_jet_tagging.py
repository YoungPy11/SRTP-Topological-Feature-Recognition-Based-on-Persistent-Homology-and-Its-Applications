"""
============================================================
pipeline_paper2_jet_tagging.py — Paper 2: 喷注标记
============================================================
基于 2601.01450v1:
  Topology-Informed Jet Tagging using Persistent Homology

Pipeline:
  1. 构造喷注点云 (模拟: 夸克喷注 vs 胶子喷注)
  2. Vietoris-Rips 过滤计算持久同调
  3. 将 H0/H1 持久图转化为持久图像 (Persistence Image)
  4. 用 CNN 进行分类训练
  5. 消融实验: H0 only, H1 only, H0+H1

核心创新: H1 持久图像具有与 H0 相当的判别能力
============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path
import sys
from typing import Tuple, List

rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

sys.path.insert(0, str(Path(__file__).parent))
from ph_pipeline import (
    compute_persistence_diagrams,
    compute_persistence_image,
    compute_top_k_persistences,
    batch_compute_ph_features,
    train_test_split_ph,
)

RESULTS_DIR = Path(__file__).parent.parent / "results"


# ============================================================
# 1. 模拟喷注点云生成
# ============================================================

def generate_quark_jet(
    n_particles: int = 50,
    noise: float = 0.02
) -> np.ndarray:
    """生成模拟夸克喷注点云。

    夸克喷注: 粒子更集中，在特征空间中形成较紧凑的分布。
    特征: (p_T_rel, η_rot, φ_rot) 三维。
    """
    # 夸克喷注粒子在接近中心的位置更集中
    p_t = np.random.exponential(0.3, n_particles)  # 指数分布
    eta = np.random.normal(0, 0.3, n_particles)     # 窄分布
    phi = np.random.normal(0, 0.3, n_particles)
    return np.column_stack((p_t, eta, phi))


def generate_gluon_jet(
    n_particles: int = 50,
    noise: float = 0.02
) -> np.ndarray:
    """生成模拟胶子喷注点云。

    胶子喷注: 粒子更分散，在特征空间中分布更广。
    """
    # 胶子喷注粒子分布更广
    p_t = np.random.exponential(0.5, n_particles)
    eta = np.random.normal(0, 0.6, n_particles)     # 宽分布
    phi = np.random.normal(0, 0.6, n_particles)
    return np.column_stack((p_t, eta, phi))


# ============================================================
# 2. CNN 分类器
# ============================================================

class SimpleCNN:
    """简单的 CNN 分类器，用于处理持久图像。

    对应 Paper 2 的 CNN 架构:
    - 3 个卷积块 + BN + ReLU
    - 全连接层
    - Adam 优化器
    """
    def __init__(self, input_channels: int, input_size: int, num_classes: int = 2):
        self.input_channels = input_channels
        self.input_size = input_size
        self.num_classes = num_classes

    def _build_model(self):
        import torch
        import torch.nn as nn

        class CNN(nn.Module):
            def __init__(self, in_channels, in_size, num_classes):
                super().__init__()
                self.conv1 = nn.Conv2d(in_channels, 16, 3, padding=1)
                self.bn1 = nn.BatchNorm2d(16)
                self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
                self.bn2 = nn.BatchNorm2d(32)
                self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
                self.bn3 = nn.BatchNorm2d(64)
                self.pool = nn.AdaptiveAvgPool2d(8)
                self.fc1 = nn.Linear(64 * 8 * 8, 128)
                self.fc2 = nn.Linear(128, num_classes)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.3)

            def forward(self, x):
                x = self.relu(self.bn1(self.conv1(x)))
                x = self.relu(self.bn2(self.conv2(x)))
                x = self.relu(self.bn3(self.conv3(x)))
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.fc2(x)
                return x

        return CNN(self.input_channels, self.input_size, self.num_classes)

    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 30,
        lr: float = 0.001,
        batch_size: int = 32
    ):
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = self._build_model().to(device)

        # 根据 channels 数决定是否需要 unsqueeze
        if self.input_channels == 1:
            X_train_t = torch.FloatTensor(X_train).unsqueeze(1)  # (N, H, W) -> (N, 1, H, W)
            X_val_t = torch.FloatTensor(X_val).unsqueeze(1)
        else:
            X_train_t = torch.FloatTensor(X_train)  # already (N, C, H, W)
            X_val_t = torch.FloatTensor(X_val)
        y_train_t = torch.LongTensor(y_train)
        y_val_t = torch.LongTensor(y_val)

        train_ds = TensorDataset(X_train_t, y_train_t)
        val_ds = TensorDataset(X_val_t, y_val_t)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=batch_size)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        history = {'train_loss': [], 'val_acc': []}
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for xb, yb in train_dl:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb, yb = xb.to(device), yb.to(device)
                    preds = model(xb).argmax(1)
                    correct += (preds == yb).sum().item()
                    total += yb.size(0)
            val_acc = correct / total
            history['train_loss'].append(epoch_loss / len(train_dl))
            history['val_acc'].append(val_acc)

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: loss={epoch_loss:.4f}, val_acc={val_acc:.3f}")

        self.model = model
        return history


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("Paper 2: 喷注标记 (Jet Tagging) Pipeline")
    print("=" * 60)

    np.random.seed(42)

    # 1. 生成模拟喷注数据集
    n_jets = 200
    n_particles = 50
    print(f"生成 {n_jets} 个喷注点云 (夸克/胶子 各半)...")

    quark_jets = [generate_quark_jet(n_particles) for _ in range(n_jets // 2)]
    gluon_jets = [generate_gluon_jet(n_particles) for _ in range(n_jets // 2)]
    point_clouds = quark_jets + gluon_jets
    labels = np.array([1] * len(quark_jets) + [0] * len(gluon_jets))

    # 2. 计算持久图像 (H0 和 H1)
    PI_RES = 40
    print(f"计算持久图像 (H0 和 H1, {PI_RES}x{PI_RES})...")

    # 先计算全局 birth/death 范围，确保所有 PI 尺寸一致
    all_births, all_deaths = [], []
    all_diagrams = []
    for pts in point_clouds:
        diagrams = compute_persistence_diagrams(pts, maxdim=1)
        all_diagrams.append(diagrams)
        for dim in [0, 1]:
            dgm = diagrams[dim]
            finite = dgm[~np.isinf(dgm[:, 1])]
            if len(finite) > 0:
                all_births.extend(finite[:, 0].tolist())
                all_deaths.extend(finite[:, 1].tolist())

    def _make_range(vals):
        if not vals:
            return (0.0, 1.0)
        vmin, vmax = np.min(vals), np.max(vals)
        pad = max((vmax - vmin) * 0.1, 0.01)
        return (vmin - pad, vmax + pad)

    shared_birth_range = _make_range(all_births)
    shared_death_range = _make_range(all_deaths)

    # 使用 top_k 特征做快速比较
    X_topk, y = batch_compute_ph_features(
        point_clouds, labels, method='top_k', dim=1, top_k=10
    )

    X_pi_h0 = []
    X_pi_h1 = []
    for diagrams in all_diagrams:
        pi_h0 = compute_persistence_image(diagrams, dim=0, resolution=PI_RES,
                                           birth_range=shared_birth_range,
                                           death_range=shared_death_range)
        pi_h1 = compute_persistence_image(diagrams, dim=1, resolution=PI_RES,
                                           birth_range=shared_birth_range,
                                           death_range=shared_death_range)
        X_pi_h0.append(pi_h0)
        X_pi_h1.append(pi_h1)

    X_pi_h0 = np.array(X_pi_h0)
    X_pi_h1 = np.array(X_pi_h1)
    X_pi_combined = np.array([np.stack([h0, h1], axis=0)
                               for h0, h1 in zip(X_pi_h0, X_pi_h1)])

    # 3. 划分训练/测试集
    from sklearn.model_selection import train_test_split

    def split(*arrays):
        return train_test_split(*arrays, test_size=0.3, random_state=42)

    X_h0_train, X_h0_test, y_train, y_test = split(X_pi_h0, labels)
    X_h1_train, X_h1_test, _, _ = split(X_pi_h1, labels)
    X_comb_train, X_comb_test, _, _ = split(X_pi_combined, labels)

    # 4. 消融实验: H0 only, H1 only, H0+H1
    print("\n" + "-" * 40)
    print("消融实验: 比较不同输入组合的性能")
    print("-" * 40)

    results = {}
    for name, X_tr, X_te, channels in [
        ("H0 only", X_h0_train, X_h0_test, 1),
        ("H1 only", X_h1_train, X_h1_test, 1),
        ("H0+H1", X_comb_train, X_comb_test, 2),
    ]:
        print(f"\n训练: {name} (channels={channels})")
        cnn = SimpleCNN(input_channels=channels, input_size=PI_RES)
        history = cnn.train_model(
            X_tr, y_train, X_te, y_test,
            epochs=20, lr=0.001, batch_size=16
        )
        final_acc = history['val_acc'][-1]
        results[name] = final_acc
        print(f"  {name} 最终验证准确率: {final_acc:.3f}")

    # 5. 可视化对比
    sample_idx = 0
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for row, (jet_type, idx) in enumerate([("Quark", 0), ("Gluon", n_jets // 2)]):
        axes[row, 0].imshow(X_pi_h0[idx], origin='lower', aspect='equal')
        axes[row, 0].set_title(f"{jet_type} H0 PI")
        axes[row, 1].imshow(X_pi_h1[idx], origin='lower', aspect='equal')
        axes[row, 1].set_title(f"{jet_type} H1 PI")
        axes[row, 2].bar(['H0 only', 'H1 only', 'H0+H1'],
                         [results.get('H0 only', 0),
                          results.get('H1 only', 0),
                          results.get('H0+H1', 0)],
                         color=['red', 'blue', 'purple'])
        axes[row, 2].set_title("Ablation Results")
        axes[row, 2].set_ylim(0, 1)

    plt.tight_layout()
    fname = RESULTS_DIR / "paper2_jet_tagging.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[Paper 2] 已保存: {fname}")

    print("\n" + "=" * 40)
    print("Paper 2 消融实验总结:")
    for name, acc in results.items():
        print(f"  {name}: Accuracy = {acc:.3f}")
    print("✅ Paper 2 Pipeline 完成")


if __name__ == "__main__":
    main()