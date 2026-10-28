"""度量对比基准实验：证明联合分布度量优于单维度度量

实验设计：
1. 纯标签偏斜：aF=1000（无特征偏斜），aL ∈ {0.05, 0.1, 0.3, 0.5, 1.0}
2. 纯特征偏斜：aL=100（近似均匀标签），aF ∈ {0.05, 0.1, 0.3, 1.0}
3. 组合偏斜：已有12组k_selection实验

关键论证：
- 纯标签偏斜：JS_label能预测，JS_feature不能 → JS_joint ≈ JS_label
- 纯特征偏斜：JS_feature能预测，JS_label不能 → JS_joint ≈ JS_feature
- 组合偏斜：JS_joint > max(JS_label, JS_feature) → 联合度量严格优于单维度
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


def compute_system_metrics(client_indices, labels, cluster_ids, num_classes, num_clusters):
    """计算系统级度量（加权平均）"""
    P_global = compute_joint_distribution(labels, cluster_ids, num_classes, num_clusters)

    client_metrics = []
    for idx in client_indices:
        client_labels = labels[idx]
        client_clusters = cluster_ids[idx]
        P_i = compute_joint_distribution(client_labels, client_clusters, num_classes, num_clusters)
        metrics = compute_all_metrics(P_i, P_global)
        client_metrics.append(metrics)

    n_samples = [len(idx) for idx in client_indices]
    total = sum(n_samples)
    weights = [n / total for n in n_samples]

    agg = {}
    for key in client_metrics[0]:
        agg[key] = sum(w * m[key] for w, m in zip(weights, client_metrics))

    return agg


def main():
    parser = argparse.ArgumentParser(description="Metric comparison experiment")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--num_rounds", type=int, default=200)
    parser.add_argument("--local_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="./results/metric_comparison")
    parser.add_argument("--cluster_file", type=str, default="./results/feature_clusters_cifar10_k10.npz")
    parser.add_argument("--skew_type", type=str, required=True,
                        choices=["label_only", "feature_only", "combined"],
                        help="偏斜类型")
    parser.add_argument("--alpha_label", type=float, default=None)
    parser.add_argument("--alpha_feature", type=float, default=None)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}, Skew type: {args.skew_type}")

    # 加载特征聚类
    cluster_data = np.load(args.cluster_file)
    cluster_ids = cluster_data['cluster_ids']
    num_clusters = int(cluster_ids.max() + 1)

    # 加载数据集
    train_dataset = get_dataset(args.dataset, train=True, data_root=args.data_root)
    testloader = get_test_dataloader(args.dataset, batch_size=256, data_root=args.data_root)
    labels = np.array([train_dataset[i][1] for i in range(len(train_dataset))])
    num_classes = int(labels.max() + 1)

    aL = args.alpha_label
    aF = args.alpha_feature

    exp_name = f"{args.skew_type}_aL{aL}_aF{aF}_seed{args.seed}"
    result_path = os.path.join(args.save_dir, f"{exp_name}.json")

    if os.path.exists(result_path):
        print(f"[SKIP] {exp_name} already exists")
        with open(result_path) as f:
            result = json.load(f)
        print(f"Accuracy: {result['final_accuracy']:.4f}")
        return

    print(f"\nRunning: {exp_name}")

    # 分区
    client_indices = feature_skew_partition(
        labels, cluster_ids, args.num_clients, aL, aF, seed=args.seed
    )

    # 计算度量
    metrics = compute_system_metrics(client_indices, labels, cluster_ids, num_classes, num_clusters)
    print(f"Metrics: JS_joint={metrics['JS_joint']:.4f}, JS_label={metrics['JS_label']:.4f}, "
          f"JS_feature={metrics['JS_feature']:.4f}, JS_cond={metrics['JS_conditional']:.4f}, "
          f"EMD_label={metrics['EMD_label']:.4f}, EMD_feature={metrics['EMD_feature']:.4f}")

    # 创建DataLoader
    client_loaders = []
    for idx in client_indices:
        subset = Subset(train_dataset, idx)
        loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        client_loaders.append(loader)

    # FL训练
    global_model = get_model(args.dataset, device)

    for round_num in tqdm(range(1, args.num_rounds + 1), desc=exp_name):
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
        'skew_type': args.skew_type,
        'alpha_label': aL,
        'alpha_feature': aF,
        'final_accuracy': float(accuracy),
        'final_loss': float(loss),
        'metrics': metrics,
    }

    os.makedirs(args.save_dir, exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nFinal accuracy: {accuracy:.4f}")
    print(f"Saved to {result_path}")


if __name__ == "__main__":
    main()
