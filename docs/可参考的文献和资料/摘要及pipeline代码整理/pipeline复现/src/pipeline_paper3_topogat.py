"""
============================================================
pipeline_paper3_topogat.py — Paper 3: TopoGAT 显著特征选择
============================================================
基于 2602.14228v1:
  Learning Significant Persistent Homology Features for
  3D Shape Understanding

Pipeline:
  1. 加载/生成点云数据, 计算 H1/H2 持久图
  2. 实现不同程度的显著特征选择:
     a) 硬阈值 (Top-K 风格)
     b) 软掩码 (TopoGAT 风格)
     c) 统计置信带方法 (Fasy et al. 方法)
  3. 比较不同选择方法的效果
     - Wasserstein 距离
     - Bottleneck 距离
     - 持久熵差异
  4. 使用 GNN 进行分类 (GCN/GAT/GIN 简化版)

核心创新: 可学习的显著特征选择替代固定 Top-K
============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path
import sys
from typing import List, Tuple, Optional

rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

sys.path.insert(0, str(Path(__file__).parent))
from ph_pipeline import (
    compute_persistence_diagrams,
    select_significant_features,
    wasserstein_distance,
    bottleneck_distance,
    compute_top_k_persistences,
    generate_circle, generate_disk, generate_sphere, generate_torus,
)

RESULTS_DIR = Path(__file__).parent.parent / "results"


# ============================================================
# 1. 持久熵计算
# ============================================================

def persistent_entropy(diagram: np.ndarray) -> float:
    """计算持久熵。

    对应 Paper 3 公式所需:
    E(D) = -sum(p_i * log(p_i)), where p_i = (d_i - b_i) / sum(d_j - b_j)

    Args:
        diagram: 持久图 (n, 2)

    Returns:
        持久熵值
    """
    finite = diagram[~np.isinf(diagram[:, 1])]
    if len(finite) == 0:
        return 0.0
    pers = finite[:, 1] - finite[:, 0]
    pers = pers[pers > 0]
    if len(pers) == 0:
        return 0.0
    total = pers.sum()
    probs = pers / total
    # 过滤掉接近 0 的概率
    probs = probs[probs > 1e-10]
    return -np.sum(probs * np.log(probs))


# ============================================================
# 2. 特征选择方法对比
# ============================================================

def compare_selection_methods(
    diagrams: List[np.ndarray],
    dim: int = 1,
    thresholds: List[float] = None
) -> dict:
    """比较不同特征选择方法的效果。

    对应 Paper 3 Tables 1-2 的对比实验。
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.5, 10)

    dgm = diagrams[dim]
    finite = dgm[~np.isinf(dgm[:, 1])]
    if len(finite) == 0:
        return {'method': [], 'wd': [], 'bd': [], 'pe_diff': []}

    results = {'method': [], 'threshold': [], 'wd': [], 'bd': [], 'pe_diff': [], 'n_selected': []}

    for method in ['hard', 'topo', 'stat']:
        for thresh in thresholds:
            selected, mask = select_significant_features(
                dgm, threshold=thresh, method=method, eta=10.0
            )

            if len(selected) == 0:
                continue

            # 构建选后持久图
            selected_dgm = selected
            wd = wasserstein_distance(finite, selected_dgm, p=2)
            bd = bottleneck_distance(finite, selected_dgm)
            pe_orig = persistent_entropy(finite)
            pe_sel = persistent_entropy(selected_dgm)

            results['method'].append(method)
            results['threshold'].append(thresh)
            results['wd'].append(wd)
            results['bd'].append(bd)
            results['pe_diff'].append(abs(pe_orig - pe_sel))
            results['n_selected'].append(len(selected))

    return results


# ============================================================
# 3. 简化 GAT 实现
# ============================================================

class SimpleGAT:
    """简化版图注意力网络，用于点云分类。

    对应 Paper 3 的 GAT 分类器实现。
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_classes: int = 40,
        num_heads: int = 4
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_heads = num_heads

    def _build_model(self):
        import torch
        import torch.nn as nn

        class GATLayer(nn.Module):
            def __init__(self, in_dim, out_dim, num_heads):
                super().__init__()
                self.num_heads = num_heads
                self.attn = nn.Linear(2 * out_dim, 1)
                self.W = nn.Linear(in_dim, out_dim * num_heads, bias=False)
                self.leaky_relu = nn.LeakyReLU(0.2)

            def forward(self, x, adj):
                # x: (N, in_dim), adj: (N, N)
                N = x.size(0)
                h = self.W(x)  # (N, out_dim * num_heads)
                h = h.view(N, self.num_heads, -1)  # (N, heads, out_dim)

                # 简化: 使用所有边
                out = torch.relu(h.mean(dim=1))  # 平均聚合
                return out

        class GATClassifier(nn.Module):
            def __init__(self, in_dim, hidden_dim, num_classes, num_heads):
                super().__init__()
                self.gat1 = GATLayer(in_dim, hidden_dim, num_heads)
                self.gat2 = GATLayer(hidden_dim, hidden_dim, num_heads)
                self.fc = nn.Linear(hidden_dim, num_classes)

            def forward(self, x, adj):
                x = self.gat1(x, adj)
                x = self.gat2(x, adj)
                x = x.mean(dim=0)  # 全局池化
                return self.fc(x)

        return GATClassifier(self.input_dim, self.hidden_dim, self.num_classes, self.num_heads)

    def train_model(
        self,
        X_train: List[Tuple[np.ndarray, np.ndarray]],
        y_train: np.ndarray,
        X_val: List[Tuple[np.ndarray, np.ndarray]],
        y_val: np.ndarray,
        epochs: int = 50,
        lr: float = 0.001
    ):
        """使用简化版 GAT 训练分类器。

        Args:
            X_train: 每个元素为 (几何特征, 拓扑特征) 的元组
            y_train: 标签
            X_val: 验证集
            y_val: 验证标签
            epochs: 训练轮数
            lr: 学习率
        """
        import torch
        import torch.nn as nn

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = self._build_model().to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        history = {'train_loss': [], 'val_acc': []}
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            n_batches = 0

            # 逐个样本训练 (简化)
            for (geo_feat, topo_feat), label in zip(X_train, y_train):
                # 构建特征矩阵
                combined = np.concatenate([geo_feat, topo_feat], axis=-1)
                x_t = torch.FloatTensor(combined).to(device)
                y_t = torch.LongTensor([label]).to(device)

                # 用 kNN 构建邻接矩阵
                from sklearn.neighbors import NearestNeighbors
                nn_model = NearestNeighbors(n_neighbors=min(10, len(combined)))
                nn_model.fit(combined)
                adj = torch.zeros((len(combined), len(combined)), device=device)
                dists, indices = nn_model.kneighbors(combined)
                for i, neigh in enumerate(indices):
                    adj[i, neigh] = 1.0

                optimizer.zero_grad()
                output = model(x_t, adj).unsqueeze(0)
                loss = criterion(output, y_t)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

                if n_batches >= 50:  # 限制 batch 数
                    break

            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for (geo_feat, topo_feat), label in zip(X_val, y_val):
                    if total >= 30:
                        break
                    combined = np.concatenate([geo_feat, topo_feat], axis=-1)
                    x_t = torch.FloatTensor(combined).to(device)
                    y_t = torch.LongTensor([label]).to(device)

                    from sklearn.neighbors import NearestNeighbors
                    nn_model = NearestNeighbors(n_neighbors=min(10, len(combined)))
                    nn_model.fit(combined)
                    adj = torch.zeros((len(combined), len(combined)), device=device)
                    dists, indices = nn_model.kneighbors(combined)
                    for i, neigh in enumerate(indices):
                        adj[i, neigh] = 1.0

                    output = model(x_t, adj).unsqueeze(0)
                    pred = output.argmax(1)
                    correct += (pred == y_t).sum().item()
                    total += 1

            val_acc = correct / max(total, 1)
            history['train_loss'].append(epoch_loss / max(n_batches, 1))
            history['val_acc'].append(val_acc)
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: loss={history['train_loss'][-1]:.4f}, acc={val_acc:.3f}")

        self.model = model
        return history


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("Paper 3: TopoGAT 显著特征选择 Pipeline")
    print("=" * 60)

    np.random.seed(42)

    # 1. 生成样本点云用于演示
    print("生成演示点云...")
    shapes = {
        'circle': generate_circle(200, noise=0.05),
        'disk': generate_disk(200, noise=0.05),
        'sphere': generate_sphere(200, noise=0.05),
        'torus': generate_torus(200, noise=0.05),
    }

    # 2. 计算持久图并比较选择方法
    print("\n" + "-" * 40)
    print("特征选择方法对比 (H1 持久图)")
    print("-" * 40)

    all_results = {}
    for shape_name, points in shapes.items():
        diagrams = compute_persistence_diagrams(points, maxdim=2)
        print(f"\n{shape_name}:")
        for dim in [1, 2]:
            if dim == 2 and shape_name == 'circle':
                continue
            dgm = diagrams[dim]
            finite = dgm[~np.isinf(dgm[:, 1])]
            if len(finite) == 0:
                continue
            print(f"  H{dim}: {len(finite)} 个有限持久点")

            # 比较不同方法和阈值
            for method in ['hard', 'topo', 'stat']:
                for thresh in [0.05, 0.1, 0.2]:
                    selected, mask = select_significant_features(
                        dgm, threshold=thresh, method=method
                    )
                    if len(selected) == 0:
                        continue
                    wd = wasserstein_distance(finite, selected, p=2)
                    bd = bottleneck_distance(finite, selected)
                    print(f"    method={method}, λ={thresh:.2f}: "
                          f"n={len(selected)}, WD={wd:.4f}, BD={bd:.4f}")

    # 3. 可视化持久图和显著特征
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for idx, (shape_name, points) in enumerate(shapes.items()):
        ax = axes[idx // 2, idx % 2]
        diagrams = compute_persistence_diagrams(points, maxdim=2)
        for dim, color in [(0, 'red'), (1, 'blue'), (2, 'green')]:
            dgm = diagrams[dim]
            finite = dgm[~np.isinf(dgm[:, 1])]
            if len(finite) > 0:
                ax.scatter(finite[:, 0], finite[:, 1],
                          c=color, s=5, alpha=0.5, label=f'H{dim}')

        # 标记显著特征 (H1, threshold=0.1)
        selected, _ = select_significant_features(
            diagrams[1], threshold=0.1, method='topo'
        )
        if len(selected) > 0:
            ax.scatter(selected[:, 0], selected[:, 1],
                      facecolors='none', edgecolors='magenta',
                      s=80, linewidths=2, label='Significant (TopoGAT)')

        lim = max(ax.get_xlim()[1], ax.get_ylim()[1], 1.0)
        ax.plot([0, lim], [0, lim], 'k--', alpha=0.3)
        ax.set_title(f"{shape_name} - PD with Significant Features")
        ax.set_xlabel("Birth"); ax.set_ylabel("Death")
        ax.legend(fontsize=7, loc='lower right')
        ax.set_aspect('equal')

    plt.tight_layout()
    fname = RESULTS_DIR / "paper3_topogat_selection.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[Paper 3] 已保存: {fname}")

    print("\n✅ Paper 3 Pipeline 完成")


if __name__ == "__main__":
    main()