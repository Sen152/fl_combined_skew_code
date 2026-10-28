"""k值选择实验：跑多组alpha的FL实验，计算不同k值下的覆盖度-性能R²

策略：
1. 用已有的特征聚类（k=4,10,20,50）
2. 跑 alpha_label ∈ {0.05, 0.1, 0.3, 0.5, 1.0} × alpha_feature ∈ {0.05, 0.1, 0.3, 1.0, 1000}
   = 25组实验（但1000是对照组，只跑1次）
   实际：5×4 + 1 = 21组
3. 每组记录最终accuracy + 各k值下的覆盖度度量
4. 分析：哪个k的JS_joint与accuracy的R²最高
"""

import argparse
import sys
import os
import json
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.data.datasets import get_dataset, get_test_dataloader
from src.data.partition import dirichlet_partition
from src.models.cnn import get_model, get_model_params, set_model_params
from src.metrics.coverage import compute_joint_distribution, compute_all_metrics
from src.utils.logging import ExperimentLogger


def train_client(model, trainloader, device, local_epochs, lr):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    for _ in range(local_epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()


def evaluate(model, testloader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss_sum += criterion(outputs, labels).item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return loss_sum / total, correct / total


def fedavg_aggregate(global_params, client_params_list, num_samples_list):
    total_samples = sum(num_samples_list)
    aggregated = []
    for i in range(len(global_params)):
        weighted_sum = sum(
            params[i] * (n / total_samples)
            for params, n in zip(client_params_list, num_samples_list)
        )
        aggregated.append(weighted_sum)
    return aggregated


def feature_skew_partition(labels, cluster_ids, num_clients, alpha_label, alpha_feature, seed=42):
    rng = np.random.default_rng(seed)
    num_classes = int(labels.max() + 1)

    label_client_indices = dirichlet_partition(labels, num_clients, alpha_label, seed)

    final_indices = [[] for _ in range(num_clients)]

    for client_id in range(num_clients):
        client_samples = label_client_indices[client_id]
        client_labels = labels[client_samples]
        client_clusters = cluster_ids[client_samples]

        if alpha_feature >= 100:
            final_indices[client_id] = client_samples
            continue

        new_client_samples = []
        for y in range(num_classes):
            label_mask = client_labels == y
            label_indices = np.where(label_mask)[0]
            if len(label_indices) == 0:
                continue

            label_clusters = client_clusters[label_indices]
            unique_clusters, cluster_counts = np.unique(label_clusters, return_counts=True)

            proportions = rng.dirichlet(np.repeat(alpha_feature, len(unique_clusters)))

            total = len(label_indices)
            target_counts = (proportions * total).astype(int)
            target_counts[-1] = total - target_counts[:-1].sum()

            for i, c in enumerate(unique_clusters):
                cluster_mask = label_clusters == c
                cluster_sample_indices = label_indices[cluster_mask]
                n_select = min(target_counts[i], len(cluster_sample_indices))
                if n_select > 0:
                    selected = rng.choice(cluster_sample_indices, size=n_select, replace=False)
                    new_client_samples.extend(selected)

        final_indices[client_id] = np.array(new_client_samples, dtype=int)

    return [np.array(idx, dtype=int) for idx in final_indices]


def compute_metrics_for_k(client_indices, labels, cluster_data_k, num_classes):
    """对给定的k值聚类，计算所有客户端的覆盖度度量"""
    cluster_ids = cluster_data_k['cluster_ids']
    num_clusters = int(cluster_ids.max() + 1)

    P_global = compute_joint_distribution(labels, cluster_ids, num_classes, num_clusters)

    client_metrics = []
    for idx in client_indices:
        client_labels = labels[idx]
        client_clusters = cluster_ids[idx]
        P_i = compute_joint_distribution(client_labels, client_clusters, num_classes, num_clusters)
        metrics = compute_all_metrics(P_i, P_global)
        client_metrics.append(metrics)

    # 聚合为系统级度量（加权平均）
    n_samples = [len(idx) for idx in client_indices]
    total = sum(n_samples)
    weights = [n / total for n in n_samples]

    agg = {}
    for key in client_metrics[0]:
        agg[key] = sum(w * m[key] for w, m in zip(weights, client_metrics))

    return agg


def main():
    parser = argparse.ArgumentParser(description="K-value selection experiment")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--num_rounds", type=int, default=200)
    parser.add_argument("--local_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="./results/k_selection")
    parser.add_argument("--cluster_dir", type=str, default="./results")
    parser.add_argument("--k_values", type=str, default="4,10,20,50",
                        help="逗号分隔的k值列表")
    # 可选：只跑指定的alpha组合
    parser.add_argument("--alpha_label", type=float, default=None,
                        help="只跑指定alpha_label（不指定则跑全部）")
    parser.add_argument("--alpha_feature", type=float, default=None,
                        help="只跑指定alpha_feature（不指定则跑全部）")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    k_values = [int(k) for k in args.k_values.split(',')]

    # 加载不同k值的特征聚类
    cluster_data = {}
    for k in k_values:
        path = os.path.join(args.cluster_dir, f"feature_clusters_{args.dataset}_k{k}.npz")
        if os.path.exists(path):
            cluster_data[k] = np.load(path)
            print(f"Loaded k={k}: {cluster_data[k]['features'].shape[0]} samples, "
                  f"{int(cluster_data[k]['cluster_ids'].max()+1)} clusters")
        else:
            print(f"WARNING: k={k} cluster file not found at {path}")

    if not cluster_data:
        print("ERROR: No cluster files found!")
        return

    # 加载数据集
    print(f"Loading {args.dataset}...")
    train_dataset = get_dataset(args.dataset, train=True, data_root=args.data_root)
    testloader = get_test_dataloader(args.dataset, batch_size=256, data_root=args.data_root)
    labels = np.array([train_dataset[i][1] for i in range(len(train_dataset))])
    num_classes = int(labels.max() + 1)

    # 定义alpha组合
    alpha_labels = [0.05, 0.1, 0.3, 0.5, 1.0]
    alpha_features = [0.05, 0.1, 0.3, 1.0, 1000.0]

    if args.alpha_label is not None:
        alpha_labels = [args.alpha_label]
    if args.alpha_feature is not None:
        alpha_features = [args.alpha_feature]

    # 用第一个k的聚类做分区（所有k用相同的分区，只换聚类）
    primary_k = min(cluster_data.keys())
    primary_cluster_ids = cluster_data[primary_k]['cluster_ids']

    all_results = []

    for aL, aF in product(alpha_labels, alpha_features):
        exp_name = f"aL{aL}_aF{aF}_seed{args.seed}"
        result_path = os.path.join(args.save_dir, f"{exp_name}.json")

        # 检查是否已有结果
        if os.path.exists(result_path):
            print(f"\n[SKIP] {exp_name} already exists")
            with open(result_path) as f:
                result = json.load(f)
            all_results.append(result)
            continue

        print(f"\n{'='*60}")
        print(f"Running: alpha_label={aL}, alpha_feature={aF}")

        # 分区
        client_indices = feature_skew_partition(
            labels, primary_cluster_ids, args.num_clients,
            aL, aF, seed=args.seed
        )

        # 计算各k值下的覆盖度度量
        metrics_by_k = {}
        for k, cdata in cluster_data.items():
            metrics_by_k[f"k{k}"] = compute_metrics_for_k(
                client_indices, labels, cdata, num_classes
            )
            m = metrics_by_k[f"k{k}"]
            print(f"  k={k}: JS_joint={m['JS_joint']:.4f}, JS_label={m['JS_label']:.4f}, "
                  f"JS_feature={m['JS_feature']:.4f}, Cov_basic={m['Cov_basic']:.4f}")

        # 创建DataLoader
        client_loaders = []
        for idx in client_indices:
            subset = Subset(train_dataset, idx)
            loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True, num_workers=0)
            client_loaders.append(loader)

        # FL训练
        global_model = get_model(args.dataset, device)

        for round_num in tqdm(range(1, args.num_rounds + 1), desc=f"aL={aL},aF={aF}"):
            client_params_list = []
            num_samples_list = []
            for cid in range(args.num_clients):
                local_model = get_model(args.dataset, device)
                set_model_params(local_model, get_model_params(global_model))
                train_client(local_model, client_loaders[cid], device, args.local_epochs, args.lr)
                client_params_list.append(get_model_params(local_model))
                num_samples_list.append(len(client_indices[cid]))

            new_params = fedavg_aggregate(get_model_params(global_model), client_params_list, num_samples_list)
            set_model_params(global_model, new_params)

            loss, accuracy = evaluate(global_model, testloader, device)

            if round_num % 50 == 0 or round_num <= 3:
                print(f"  Round {round_num}: acc={accuracy:.4f}")

        # 保存结果
        result = {
            'alpha_label': aL,
            'alpha_feature': aF,
            'final_accuracy': float(accuracy),
            'final_loss': float(loss),
            'metrics_by_k': metrics_by_k,
        }

        os.makedirs(args.save_dir, exist_ok=True)
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)

        all_results.append(result)
        print(f"  Final accuracy: {accuracy:.4f}")

    # 汇总分析
    print(f"\n{'='*60}")
    print("SUMMARY: R² analysis for each k value")
    print(f"{'='*60}")

    from scipy.stats import pearsonr

    accuracies = np.array([r['final_accuracy'] for r in all_results])

    for k in cluster_data.keys():
        key = f"k{k}"
        for metric_name in ['JS_joint', 'JS_label', 'JS_feature', 'Cov_basic', 'Cov_weighted']:
            values = np.array([r['metrics_by_k'][key][metric_name] for r in all_results])
            if len(values) > 2 and np.std(values) > 0:
                r_val, p_val = pearsonr(values, accuracies)
                print(f"  k={k:3d}, {metric_name:15s}: r={r_val:+.4f}, r²={r_val**2:.4f}, p={p_val:.4f}")
            else:
                print(f"  k={k:3d}, {metric_name:15s}: no variance")

    # 保存汇总
    summary_path = os.path.join(args.save_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
