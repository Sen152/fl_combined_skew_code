"""度量排名对比实验：在同一批实验数据上计算所有候选度量，输出R²排名表

目标：证明JS_joint在所有可计算度量中是唯一最优的FL性能预测指标

数据来源：
1. k_selection实验（12组组合偏斜）——用k=10的度量
2. metric_comparison实验（8组纯标签/纯特征偏斜）
3. intervention实验（5组干预实验）
4. blindspot/domain_blindspot/style_blindspot实验

新增度量（需要重新计算）：
- MMD：特征空间分布距离
- 标签分布方差：简单baseline
- 标签熵 per client：信息论
- 客户端数据量方差：控制变量
- Dirichlet α值：文献常用proxy
"""

import json
import os
import sys
import numpy as np
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import jensenshannon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.partition import dirichlet_partition
from src.metrics.coverage import compute_joint_distribution, compute_all_metrics
from torchvision import datasets, transforms


def compute_mmd(X, Y, gamma=None):
    """计算两个样本集的MMD (Maximum Mean Discrepancy)"""
    from sklearn.metrics.pairwise import rbf_kernel
    if gamma is None:
        # Median heuristic
        combined = np.vstack([X[:500], Y[:500]])
        dists = np.sqrt(((combined[:, None] - combined[None, :]) ** 2).sum(axis=2))
        gamma = 1.0 / (2 * np.median(dists[dists > 0]) ** 2 + 1e-10)

    K_xx = rbf_kernel(X[:500], X[:500], gamma=gamma)
    K_yy = rbf_kernel(Y[:500], Y[:500], gamma=gamma)
    K_xy = rbf_kernel(X[:500], Y[:500], gamma=gamma)

    mmd = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
    return float(max(mmd, 0))


def compute_label_variance(client_labels_list, num_classes=10):
    """计算客户端间标签分布方差"""
    dists = []
    for labels in client_labels_list:
        counts = np.bincount(labels, minlength=num_classes).astype(float)
        dists.append(counts / counts.sum())

    dists = np.array(dists)
    mean_dist = dists.mean(axis=0)
    variance = np.mean(np.sum((dists - mean_dist) ** 2, axis=1))
    return float(variance)


def compute_label_entropy(client_labels_list, num_classes=10):
    """计算客户端间标签熵的加权平均"""
    entropies = []
    weights = []
    for labels in client_labels_list:
        counts = np.bincount(labels, minlength=num_classes).astype(float)
        probs = counts / counts.sum()
        # Shannon entropy
        mask = probs > 0
        ent = -np.sum(probs[mask] * np.log2(probs[mask]))
        entropies.append(ent)
        weights.append(len(labels))

    weights = np.array(weights, dtype=float)
    weights /= weights.sum()
    return float(np.average(entropies, weights=weights))


def compute_data_size_variance(client_indices_list):
    """计算客户端数据量方差"""
    sizes = [len(idx) for idx in client_indices_list]
    return float(np.var(sizes))


def main():
    print("=" * 80)
    print("METRIC RANKING EXPERIMENT")
    print("=" * 80)

    # 收集所有实验数据
    all_data = []

    # 1. k_selection实验（组合偏斜，用k=10）
    k_dir = "results/k_selection"
    if os.path.exists(k_dir):
        for fname in sorted(os.listdir(k_dir)):
            if fname.endswith(".json") and fname != "summary.json":
                with open(os.path.join(k_dir, fname)) as f:
                    d = json.load(f)
                if 'metrics_by_k' in d and 'k10' in d['metrics_by_k']:
                    m = d['metrics_by_k']['k10']
                    all_data.append({
                        'source': 'k_selection',
                        'alpha_label': d['alpha_label'],
                        'alpha_feature': d['alpha_feature'],
                        'accuracy': d['final_accuracy'],
                        'JS_joint': m.get('JS_joint', float('nan')),
                        'JS_label': m.get('JS_label', float('nan')),
                        'JS_feature': m.get('JS_feature', float('nan')),
                        'EMD_label': m.get('EMD_label', float('nan')),
                        'EMD_feature': m.get('EMD_feature', float('nan')),
                        'Cov_weighted': m.get('Cov_weighted', float('nan')),
                    })

    # 2. metric_comparison实验
    mc_dir = "results/metric_comparison"
    if os.path.exists(mc_dir):
        for fname in sorted(os.listdir(mc_dir)):
            if fname.endswith(".json"):
                with open(os.path.join(mc_dir, fname)) as f:
                    d = json.load(f)
                m = d.get('metrics', {})
                all_data.append({
                    'source': 'metric_comparison',
                    'alpha_label': d.get('alpha_label', float('nan')),
                    'alpha_feature': d.get('alpha_feature', float('nan')),
                    'accuracy': d['final_accuracy'],
                    'JS_joint': m.get('JS_joint', float('nan')),
                    'JS_label': m.get('JS_label', float('nan')),
                    'JS_feature': m.get('JS_feature', float('nan')),
                    'EMD_label': m.get('EMD_label', float('nan')),
                    'EMD_feature': m.get('EMD_feature', float('nan')),
                    'Cov_weighted': m.get('Cov_weighted', float('nan')),
                })

    # 3. intervention实验
    int_dir = "results/intervention"
    if os.path.exists(int_dir):
        for fname in sorted(os.listdir(int_dir)):
            if fname.endswith(".json"):
                with open(os.path.join(int_dir, fname)) as f:
                    d = json.load(f)
                m = d.get('metrics', {})
                all_data.append({
                    'source': 'intervention',
                    'alpha_label': d.get('alpha_label', float('nan')),
                    'alpha_feature': float('nan'),
                    'invert_ratio': d.get('invert_ratio', float('nan')),
                    'accuracy': d['final_accuracy'],
                    'JS_joint': m.get('JS_joint', float('nan')),
                    'JS_label': m.get('JS_label', float('nan')),
                    'JS_feature': m.get('JS_feature', float('nan')),
                    'EMD_label': m.get('EMD_label', float('nan')),
                    'EMD_feature': m.get('EMD_feature', float('nan')),
                    'Cov_weighted': m.get('Cov_weighted', float('nan')),
                })

    # 4. domain_blindspot实验（round_log格式）
    db_dir = "results/domain_blindspot"
    if os.path.exists(db_dir):
        for fname in sorted(os.listdir(db_dir)):
            if fname.endswith(".json"):
                with open(os.path.join(db_dir, fname)) as f:
                    d = json.load(f)
                # domain_blindspot是round_log列表，取最后一轮
                if isinstance(d, list) and len(d) > 0:
                    last_round = d[-1]
                    mode = fname.replace('domain_', '').replace('_alpha0.1_seed42.json', '')
                    all_data.append({
                        'source': 'domain_blindspot',
                        'alpha_label': 0.1,
                        'alpha_feature': float('nan'),
                        'mode': mode,
                        'accuracy': last_round.get('accuracy', float('nan')),
                        'JS_joint': float('nan'),
                        'JS_label': float('nan'),
                        'JS_feature': float('nan'),
                        'EMD_label': float('nan'),
                        'EMD_feature': float('nan'),
                        'Cov_weighted': float('nan'),
                    })

    print(f"\nTotal experiments collected: {len(all_data)}")
    for src in set(d['source'] for d in all_data):
        count = sum(1 for d in all_data if d['source'] == src)
        print(f"  {src}: {count}")

    # 计算新增度量（需要数据分区信息）
    # 先尝试从已有实验中推断，或者重新计算
    print("\n--- Computing additional metrics ---")

    # 加载特征聚类
    cluster_file = "results/feature_clusters_cifar10_k10.npz"
    if os.path.exists(cluster_file):
        cluster_data = np.load(cluster_file)
        cluster_ids = cluster_data['cluster_ids']
        num_clusters = int(cluster_ids.max() + 1)
        print(f"Loaded cluster file: {num_clusters} clusters")
    else:
        print("WARNING: cluster file not found, skipping MMD computation")
        cluster_ids = None

    # 加载CIFAR-10数据集
    data_root = "./data"
    try:
        train_dataset = datasets.CIFAR10(root=data_root, train=True, download=False,
                                          transform=transforms.ToTensor())
        labels = np.array(train_dataset.targets)
        num_classes = 10
        print(f"Loaded CIFAR-10: {len(labels)} samples")
    except:
        print("WARNING: CIFAR-10 not found")
        train_dataset = None
        labels = None

    # 对每个实验，补充计算新增度量
    if labels is not None and cluster_ids is not None:
        for d in all_data:
            aL = d.get('alpha_label', None)
            aF = d.get('alpha_feature', None)

            if aL is None or np.isnan(aL):
                continue

            # 重新生成分区
            try:
                from src.data.partition import dirichlet_partition, combined_partition
                seed = 42

                if aF is not None and not np.isnan(aF) and aF < 100:
                    client_indices = combined_partition(
                        labels, cluster_ids, 10, aL, aF, seed)
                else:
                    client_indices = dirichlet_partition(labels, 10, aL, seed)

                # 标签分布方差
                client_labels = [labels[idx] for idx in client_indices]
                d['label_variance'] = compute_label_variance(client_labels, num_classes)

                # 标签熵
                d['label_entropy'] = compute_label_entropy(client_labels, num_classes)

                # 数据量方差
                d['data_size_variance'] = compute_data_size_variance(client_indices)

                # Dirichlet alpha（用aL作为proxy）
                d['alpha_label_value'] = float(aL)
                if aF is not None and not np.isnan(aF):
                    d['alpha_feature_value'] = float(aF)

            except Exception as e:
                print(f"  Error computing metrics for aL={aL}, aF={aF}: {e}")

    # 对intervention实验，额外计算invert_ratio作为proxy
    for d in all_data:
        if d['source'] == 'intervention':
            d['alpha_label_value'] = d.get('alpha_label', 0.3)
            d['invert_ratio_value'] = d.get('invert_ratio', 0.0)

    # === R²排名表 ===
    print(f"\n{'=' * 80}")
    print("R² RANKING TABLE")
    print("=" * 80)

    accuracies = np.array([d['accuracy'] for d in all_data])

    # 所有候选度量
    candidate_metrics = [
        ('JS_joint', '联合分布（我们提出的）'),
        ('JS_feature', '特征分布'),
        ('EMD_feature', '特征分布'),
        ('Cov_weighted', '联合覆盖度'),
        ('JS_label', '标签分布'),
        ('EMD_label', '标签分布'),
        ('label_variance', '标签分布'),
        ('label_entropy', '标签分布'),
        ('data_size_variance', '数据量'),
        ('alpha_label_value', '间接指标'),
        ('alpha_feature_value', '间接指标'),
    ]

    ranking = []
    for metric_key, category in candidate_metrics:
        vals = []
        accs = []
        for d in all_data:
            v = d.get(metric_key, float('nan'))
            if not np.isnan(v) if isinstance(v, float) else True:
                vals.append(v)
                accs.append(d['accuracy'])

        if len(vals) < 5:
            continue

        vals = np.array(vals)
        accs = np.array(accs)

        if np.std(vals) < 1e-10:
            continue

        r, p = pearsonr(vals, accs)
        rho, p_spearman = spearmanr(vals, accs)

        ranking.append({
            'metric': metric_key,
            'category': category,
            'r': r,
            'r_squared': r ** 2,
            'p_value': p,
            'spearman_rho': rho,
            'n_points': len(vals),
        })

    # 按R²绝对值排序
    ranking.sort(key=lambda x: abs(x['r_squared']), reverse=True)

    print(f"\n{'Rank':>4} {'Metric':>22} {'Category':>20} {'R²':>8} {'r':>8} {'p-value':>10} {'Spearman ρ':>10} {'N':>4}")
    print("-" * 100)
    for i, item in enumerate(ranking, 1):
        marker = " ★" if item['metric'] == 'JS_joint' else ""
        print(f"{i:4d} {item['metric']:>22} {item['category']:>20} "
              f"{item['r_squared']:8.4f} {item['r']:+8.4f} {item['p_value']:10.4f} "
              f"{item['spearman_rho']:+10.4f} {item['n_points']:4d}{marker}")

    # === 分场景分析 ===
    print(f"\n{'=' * 80}")
    print("SCENARIO-SPECIFIC ANALYSIS")
    print("=" * 80)

    scenarios = {
        '组合偏斜': [d for d in all_data if d['source'] == 'k_selection'],
        '纯标签偏斜': [d for d in all_data if d['source'] == 'metric_comparison'
                     and d.get('alpha_feature', 0) > 100],
        '纯特征偏斜': [d for d in all_data if d['source'] == 'metric_comparison'
                     and d.get('alpha_feature', 0) <= 100],
        '干预实验': [d for d in all_data if d['source'] == 'intervention'],
        '域偏斜': [d for d in all_data if d['source'] == 'domain_blindspot'],
    }

    for scenario_name, scenario_data in scenarios.items():
        if len(scenario_data) < 3:
            print(f"\n{scenario_name}: too few data points ({len(scenario_data)})")
            continue

        print(f"\n--- {scenario_name} ({len(scenario_data)} experiments) ---")
        accs = np.array([d['accuracy'] for d in scenario_data])

        for metric_key in ['JS_joint', 'JS_label', 'JS_feature']:
            vals = np.array([d.get(metric_key, float('nan')) for d in scenario_data])
            valid = ~np.isnan(vals)
            if valid.sum() >= 3 and np.std(vals[valid]) > 1e-10:
                r, p = pearsonr(vals[valid], accs[valid])
                print(f"  {metric_key:20s}: R²={r**2:.4f}, r={r:+.4f}, p={p:.4f}")

    # === 保存结果 ===
    output = {
        'ranking': ranking,
        'total_experiments': len(all_data),
        'by_source': {src: sum(1 for d in all_data if d['source'] == src)
                      for src in set(d['source'] for d in all_data)},
    }

    output_path = "results/metric_ranking.json"
    os.makedirs("results", exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
