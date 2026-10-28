"""盲区实验：验证同P(y)不同P(c|y)是否导致性能显著差异"""

import argparse
import sys
import os

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
    """
    组合偏斜分区：标签偏斜 + 特征偏斜

    alpha_feature=1000 → 特征均匀（对照组）
    alpha_feature=0.1 → 特征严重偏斜（实验组）

    实现方式：对每个客户端的每个标签，用Dirichlet决定每个特征簇的保留比例，
    然后按比例采样。alpha_feature越小，少数簇占比越高，特征偏斜越严重。
    """
    rng = np.random.default_rng(seed)
    num_classes = int(labels.max() + 1)
    num_clusters = int(cluster_ids.max() + 1)

    # 第一步：标签Dirichlet分区
    label_client_indices = dirichlet_partition(labels, num_clients, alpha_label, seed)

    # 第二步：在每个客户端内，按特征簇做二次Dirichlet
    final_indices = [[] for _ in range(num_clients)]

    for client_id in range(num_clients):
        client_samples = label_client_indices[client_id]
        client_labels = labels[client_samples]
        client_clusters = cluster_ids[client_samples]

        if alpha_feature >= 100:
            # 不做特征偏斜，直接返回
            final_indices[client_id] = client_samples
            continue

        # 对每个标签内的样本，按特征簇做Dirichlet重采样
        new_client_samples = []
        for y in range(num_classes):
            label_mask = client_labels == y
            label_indices = np.where(label_mask)[0]
            if len(label_indices) == 0:
                continue

            label_clusters = client_clusters[label_indices]
            unique_clusters, cluster_counts = np.unique(label_clusters, return_counts=True)

            # Dirichlet决定每个特征簇的目标比例
            proportions = rng.dirichlet(np.repeat(alpha_feature, len(unique_clusters)))

            # 按比例分配样本数
            total = len(label_indices)
            target_counts = (proportions * total).astype(int)
            target_counts[-1] = total - target_counts[:-1].sum()  # 修正舍入

            # 从每个簇中按目标数量采样
            for i, c in enumerate(unique_clusters):
                cluster_mask = label_clusters == c
                cluster_sample_indices = label_indices[cluster_mask]
                n_select = min(target_counts[i], len(cluster_sample_indices))
                if n_select > 0:
                    selected = rng.choice(cluster_sample_indices, size=n_select, replace=False)
                    new_client_samples.extend(selected)

        final_indices[client_id] = np.array(new_client_samples, dtype=int)

    return [np.array(idx, dtype=int) for idx in final_indices]


def main():
    parser = argparse.ArgumentParser(description="Blind spot experiment")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--alpha_label", type=float, default=0.1, help="标签Dirichlet参数")
    parser.add_argument("--alpha_feature", type=float, default=0.3, help="特征Dirichlet参数(1000=均匀)")
    parser.add_argument("--num_rounds", type=int, default=200)
    parser.add_argument("--local_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--cluster_file", type=str, default="./results/feature_clusters_cifar10_k10.npz")
    parser.add_argument("--save_dir", type=str, default="./results/blindspot")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 加载特征簇
    print("Loading feature clusters...")
    cluster_data = np.load(args.cluster_file)
    all_cluster_ids = cluster_data['cluster_ids']
    num_clusters = int(all_cluster_ids.max() + 1)

    # 加载数据集
    print(f"Loading {args.dataset}...")
    train_dataset = get_dataset(args.dataset, train=True, data_root=args.data_root)
    testloader = get_test_dataloader(args.dataset, batch_size=256, data_root=args.data_root)
    labels = np.array([train_dataset[i][1] for i in range(len(train_dataset))])

    # 组合偏斜分区
    print(f"\nPartitioning: alpha_label={args.alpha_label}, alpha_feature={args.alpha_feature}")
    client_indices = feature_skew_partition(
        labels, all_cluster_ids, args.num_clients,
        args.alpha_label, args.alpha_feature, seed=args.seed
    )

    # 打印每个客户端的分布信息
    num_classes = int(labels.max() + 1)
    P_global = compute_joint_distribution(labels, all_cluster_ids, num_classes, num_clusters)

    for i, idx in enumerate(client_indices):
        client_labels = labels[idx]
        client_clusters = all_cluster_ids[idx]
        P_i = compute_joint_distribution(client_labels, client_clusters, num_classes, num_clusters)
        metrics = compute_all_metrics(P_i, P_global)
        print(f"  Client {i}: {len(idx)} samples, JS_joint={metrics['JS_joint']:.4f}, "
              f"JS_label={metrics['JS_label']:.4f}, JS_feature={metrics['JS_feature']:.4f}, "
              f"Cov_basic={metrics['Cov_basic']:.4f}")

    # 创建DataLoader
    client_loaders = []
    for idx in client_indices:
        subset = Subset(train_dataset, idx)
        loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        client_loaders.append(loader)

    # FL训练
    global_model = get_model(args.dataset, device)
    exp_name = f"blindspot_aL{args.alpha_label}_aF{args.alpha_feature}_seed{args.seed}"
    logger = ExperimentLogger(args.save_dir, exp_name)

    print(f"\nStarting FL: {args.num_rounds} rounds, alpha_label={args.alpha_label}, alpha_feature={args.alpha_feature}")

    for round_num in tqdm(range(1, args.num_rounds + 1), desc="FL Training"):
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
        logger.log_round(round_num, {"loss": float(loss), "accuracy": float(accuracy)})

        if round_num % 20 == 0 or round_num <= 5:
            print(f"Round {round_num}: loss={loss:.4f}, accuracy={accuracy:.4f}")

    save_path = logger.save()
    print(f"\nResults saved to {save_path}")
    print(f"Final accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
