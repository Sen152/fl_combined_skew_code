"""联合分布覆盖度度量"""

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy, wasserstein_distance


def compute_joint_distribution(labels: np.ndarray, cluster_ids: np.ndarray,
                               num_classes: int, num_clusters: int,
                               smoothing: float = 1.0) -> np.ndarray:
    """
    计算单个客户端的标签×特征簇联合分布 P(y,c)

    Args:
        labels: 该客户端的样本标签数组
        cluster_ids: 对应的特征簇ID数组
        num_classes: 类别数
        num_clusters: 特征簇数
        smoothing: Laplace平滑参数（0=无平滑）

    Returns:
        (num_classes, num_clusters) 的联合分布矩阵
    """
    joint = np.zeros((num_classes, num_clusters))
    for y, c in zip(labels, cluster_ids):
        joint[y, c] += 1
    # Laplace平滑
    joint = (joint + smoothing) / (joint.sum() + smoothing * num_classes * num_clusters)
    return joint


def compute_marginal_label(joint_dist: np.ndarray) -> np.ndarray:
    """从联合分布计算标签边缘分布 P(y)"""
    return joint_dist.sum(axis=1)


def compute_marginal_feature(joint_dist: np.ndarray) -> np.ndarray:
    """从联合分布计算特征边缘分布 P(c)"""
    return joint_dist.sum(axis=0)


def compute_coverage_basic(P_i: np.ndarray, threshold: float = None) -> float:
    """
    基础覆盖度：被覆盖的grid cell比例

    Args:
        P_i: 客户端联合分布 (C, K)
        threshold: 低于此值视为空（默认=均匀分布概率）
    """
    if threshold is None:
        threshold = 1.0 / (P_i.shape[0] * P_i.shape[1])
    return (P_i > threshold).sum() / P_i.size


def compute_coverage_weighted(P_i: np.ndarray, P_global: np.ndarray) -> float:
    """
    加权覆盖度：按全局分布加权的覆盖度

    高概率区域被覆盖的权重更大
    """
    return np.sum(P_global * np.minimum(P_i / (P_global + 1e-10), 1.0))


def compute_js_divergence(P: np.ndarray, Q: np.ndarray) -> float:
    """计算JS散度（对称版KL散度）"""
    return jensenshannon(P.flatten(), Q.flatten())


def compute_kl_divergence(P: np.ndarray, Q: np.ndarray) -> float:
    """计算KL散度 P || Q"""
    p = P.flatten()
    q = Q.flatten()
    # 确保没有零值
    mask = (p > 0) & (q > 0)
    return entropy(p[mask], q[mask])


def compute_all_metrics(P_i: np.ndarray, P_global: np.ndarray,
                        threshold: float = None) -> dict:
    """
    计算所有覆盖度相关度量

    Args:
        P_i: 客户端联合分布
        P_global: 全局联合分布
        threshold: 基础覆盖度的阈值

    Returns:
        包含所有度量的字典
    """
    # 联合分布层面
    js_joint = compute_js_divergence(P_i, P_global)

    # 边缘分布层面
    P_i_label = compute_marginal_label(P_i)
    P_global_label = compute_marginal_label(P_global)
    js_label = jensenshannon(P_i_label, P_global_label)

    P_i_feature = compute_marginal_feature(P_i)
    P_global_feature = compute_marginal_feature(P_global)
    js_feature = jensenshannon(P_i_feature, P_global_feature)

    # EMD (Earth Mover's Distance / Wasserstein-1)
    emd_label = wasserstein_distance(
        np.arange(len(P_i_label)), np.arange(len(P_global_label)),
        P_i_label, P_global_label
    )
    emd_feature = wasserstein_distance(
        np.arange(len(P_i_feature)), np.arange(len(P_global_feature)),
        P_i_feature, P_global_feature
    )

    # 覆盖度
    cov_basic = compute_coverage_basic(P_i, threshold)
    cov_weighted = compute_coverage_weighted(P_i, P_global)

    # 条件分布层面：JS(P(c|y) || P_global(c|y)) 按y加权平均
    js_conditional = 0.0
    for y in range(P_i.shape[0]):
        if P_global_label[y] > 1e-10 and P_i[y].sum() > 1e-10:
            P_i_cond = P_i[y] / (P_i[y].sum() + 1e-10)
            P_global_cond = P_global[y] / (P_global[y].sum() + 1e-10)
            js_val = jensenshannon(P_i_cond, P_global_cond)
            if not np.isnan(js_val):
                js_conditional += P_global_label[y] * js_val

    return {
        'JS_joint': float(js_joint),
        'JS_label': float(js_label),
        'JS_feature': float(js_feature),
        'JS_conditional': float(js_conditional),
        'EMD_label': float(emd_label),
        'EMD_feature': float(emd_feature),
        'Cov_basic': float(cov_basic),
        'Cov_weighted': float(cov_weighted),
    }
