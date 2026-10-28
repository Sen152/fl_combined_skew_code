"""Non-IID 数据分区：Dirichlet + 特征偏斜"""

import numpy as np
from typing import List, Tuple


def dirichlet_partition(labels: np.ndarray, num_clients: int, alpha: float,
                        seed: int = 42) -> List[np.ndarray]:
    """
    Dirichlet标签分区：每个类别的样本按Dir(alpha)分配给各客户端

    Args:
        labels: 全部样本的标签数组
        num_clients: 客户端数量
        alpha: Dirichlet浓度参数，越小越偏斜
        seed: 随机种子

    Returns:
        每个客户端的样本索引列表
    """
    rng = np.random.default_rng(seed)
    num_classes = int(labels.max() + 1)

    # 每个类别的样本索引
    class_indices = [np.where(labels == c)[0] for c in range(num_classes)]

    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        # 对每个类别，用Dirichlet分配比例
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        # 按比例分割该类别的样本
        indices = class_indices[c].copy()
        rng.shuffle(indices)
        splits = (proportions * len(indices)).astype(int)
        splits[-1] = len(indices) - splits[:-1].sum()  # 修正舍入误差

        start = 0
        for i, split_size in enumerate(splits):
            client_indices[i].extend(indices[start:start + split_size])
            start += split_size

    return [np.array(idx, dtype=int) for idx in client_indices]


def combined_partition(labels: np.ndarray, cluster_ids: np.ndarray,
                       num_clients: int, alpha_label: float, alpha_feature: float,
                       seed: int = 42) -> List[np.ndarray]:
    """
    组合偏斜分区：同时控制标签偏斜和特征偏斜

    核心思路：
    1. 先做Dirichlet标签分区（alpha_label控制标签偏斜程度）
    2. 在每个客户端已分配的样本中，对特征簇做二次Dirichlet重分配
       - alpha_feature很大(如1000) → 特征簇均匀分布（对照组）
       - alpha_feature很小(如0.1) → 特征簇严重偏斜（实验组）

    Args:
        labels: 全部样本的标签数组
        cluster_ids: 每个样本的特征簇ID（由k-means得到）
        num_clients: 客户端数量
        alpha_label: 标签Dirichlet参数
        alpha_feature: 特征簇Dirichlet参数
        seed: 随机种子

    Returns:
        每个客户端的样本索引列表
    """
    rng = np.random.default_rng(seed)
    num_classes = int(labels.max() + 1)
    num_clusters = int(cluster_ids.max() + 1)

    # 第一步：标签维度Dirichlet分区
    label_client_indices = dirichlet_partition(labels, num_clients, alpha_label, seed)

    # 第二步：在每个客户端内，按特征簇做二次分配
    # 目的：控制P(c|y)的偏斜程度
    final_indices = [[] for _ in range(num_clients)]

    for client_id in range(num_clients):
        client_samples = label_client_indices[client_id]
        client_labels = labels[client_samples]
        client_clusters = cluster_ids[client_samples]

        # 对每个标签内的样本，按特征簇做Dirichlet重分配
        # 这里简化处理：直接在特征簇维度做Dirichlet
        # 将该客户端的样本按(标签, 特征簇)分组
        keep_mask = np.ones(len(client_samples), dtype=bool)

        if alpha_feature < 100:  # 需要特征偏斜
            # 对每个标签内的样本，按特征簇做筛选
            for y in range(num_classes):
                label_mask = client_labels == y
                label_indices = np.where(label_mask)[0]
                if len(label_indices) == 0:
                    continue

                # 该标签下各特征簇的样本
                label_clusters = client_clusters[label_indices]
                unique_clusters = np.unique(label_clusters)

                # Dirichlet决定每个特征簇的保留比例
                proportions = rng.dirichlet(np.repeat(alpha_feature, len(unique_clusters)))
                # 只保留比例最高的几个簇（模拟特征偏斜）
                # 按比例随机采样
                num_keep = max(1, int(len(label_indices) * 0.7))  # 保留70%
                cluster_probs = np.zeros(len(label_indices))
                for i, c in enumerate(unique_clusters):
                    cluster_mask = label_clusters == c
                    cluster_probs[cluster_mask] = proportions[i]

                # 确保概率非零且归一化
                cluster_probs = np.maximum(cluster_probs, 1e-10)
                cluster_probs /= cluster_probs.sum()
                keep_idx = rng.choice(len(label_indices), size=num_keep, replace=False, p=cluster_probs)
                drop_idx = np.setdiff1d(np.arange(len(label_indices)), keep_idx)
                keep_mask[label_indices[drop_idx]] = False

        final_indices[client_id] = client_samples[keep_mask]

    return [np.array(idx, dtype=int) for idx in final_indices]


def iid_partition(num_samples: int, num_clients: int, seed: int = 42) -> List[np.ndarray]:
    """IID分区：随机均匀分配"""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_samples)
    split_size = num_samples // num_clients
    return [indices[i * split_size:(i + 1) * split_size] for i in range(num_clients)]
