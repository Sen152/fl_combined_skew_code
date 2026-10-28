"""覆盖度加权FedAvg (CW-FedAvg)：用JS_joint指导聚合权重

核心思想：JS_joint高的客户端（分布偏离全局）给低权重，JS_joint低的给高权重
权重公式：w_k = 1/(JS_joint_k + eps)，归一化后替代数据量加权

实验设计：
- 在12组组合偏斜配置上跑CW-FedAvg
- 对比标准FedAvg（已有结果）
- 额外对比：随机加权（消融实验）
"""

import argparse
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from tqdm import tqdm
from sklearn.cluster import KMeans

from src.data.partition import dirichlet_partition, combined_partition
from src.models.cnn import get_model, get_model_params, set_model_params
from src.metrics.coverage import compute_joint_distribution, compute_all_metrics


def train_client(model, trainloader, device, local_epochs, lr, grad_clip=0.0):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    for _ in range(local_epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
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


def fedavg_aggregate(global_params, client_params_list, weights):
    """通用加权聚合，weights已归一化"""
    aggregated = []
    for i in range(len(global_params)):
        weighted_sum = sum(
            params[i] * w
            for params, w in zip(client_params_list, weights)
        )
        aggregated.append(weighted_sum)
    return aggregated


def median_aggregate(client_params_list):
    """坐标级中位数聚合（鲁棒聚合baseline）"""
    aggregated = []
    for i in range(len(client_params_list[0])):
        stacked = np.stack([params[i] for params in client_params_list])  # (K, ...)
        aggregated.append(np.median(stacked, axis=0))
    return aggregated


def main():
    parser = argparse.ArgumentParser(description="CW-FedAvg experiment")
    parser.add_argument("--alpha_label", type=float, required=True)
    parser.add_argument("--alpha_feature", type=float, required=True)
    parser.add_argument("--weight_mode", type=str, default="cw",
                       choices=["data", "cw", "cw_exp", "random", "median"],
                       help="data=标准FedAvg, cw=1/(JS+eps), cw_exp=exp(-beta*JS), random=随机权重, median=中位数聚合")
    parser.add_argument("--grad_clip", type=float, default=0.0,
                       help="梯度裁剪max_norm（0=关闭，>0启用）")
    parser.add_argument("--beta", type=float, default=10.0,
                       help="温度系数，用于cw_exp模式")
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--num_rounds", type=int, default=200)
    parser.add_argument("--local_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k_clusters", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="./results/cw_fedavg")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}, aL={args.alpha_label}, aF={args.alpha_feature}, mode={args.weight_mode}")

    # 加载CIFAR-10
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    train_dataset = datasets.CIFAR10(root="./data", train=True, download=False, transform=train_transform)
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=False, transform=test_transform)
    testloader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    labels = np.array(train_dataset.targets)
    num_classes = 10

    # 加载特征聚类
    cluster_file = "results/feature_clusters_cifar10_k10.npz"
    cluster_data = np.load(cluster_file)
    cluster_ids = cluster_data['cluster_ids']
    num_clusters = int(cluster_ids.max() + 1)

    # 数据分区
    if args.alpha_feature < 100:
        client_indices = combined_partition(
            labels, cluster_ids, args.num_clients,
            args.alpha_label, args.alpha_feature, args.seed)
    else:
        client_indices = dirichlet_partition(
            labels, args.num_clients, args.alpha_label, args.seed)

    # 过滤空客户端
    active_clients = [cid for cid in range(args.num_clients) if len(client_indices[cid]) > 0]
    print(f"Active clients: {len(active_clients)}/{args.num_clients}")

    # 计算每个客户端的JS_joint
    P_global = compute_joint_distribution(labels, cluster_ids, num_classes, num_clusters)

    client_js_joints = []
    active_client_indices = []
    for cid in active_clients:
        idx = client_indices[cid]
        active_client_indices.append(idx)
        client_labels = labels[idx]
        client_clusters = cluster_ids[idx]
        P_i = compute_joint_distribution(client_labels, client_clusters, num_classes, num_clusters)
        metrics = compute_all_metrics(P_i, P_global)
        client_js_joints.append(metrics['JS_joint'])

    client_js_joints = np.array(client_js_joints)
    print(f"Client JS_joint: mean={client_js_joints.mean():.4f}, std={client_js_joints.std():.4f}")
    print(f"  per client: {[f'{v:.4f}' for v in client_js_joints]}")

    # 计算聚合权重
    if args.weight_mode == "data":
        # 标准FedAvg：数据量加权
        sizes = np.array([len(idx) for idx in active_client_indices], dtype=float)
        weights = sizes / sizes.sum()
    elif args.weight_mode == "cw":
        # CW-FedAvg：1/(JS_joint + eps)
        eps = 1e-6
        raw_weights = 1.0 / (client_js_joints + eps)
        weights = raw_weights / raw_weights.sum()
    elif args.weight_mode == "cw_exp":
        # CW-FedAvg (exp)：exp(-beta * JS_joint)
        raw_weights = np.exp(-args.beta * client_js_joints)
        weights = raw_weights / raw_weights.sum()
    elif args.weight_mode == "random":
        # 随机权重（消融实验）
        rng = np.random.default_rng(args.seed)
        raw_weights = rng.random(len(active_clients))
        weights = raw_weights / raw_weights.sum()
    elif args.weight_mode == "median":
        # 中位数聚合：均匀权重（仅记录用，实际走median_aggregate）
        weights = np.ones(len(active_clients)) / len(active_clients)

    print(f"Aggregation weights: {[f'{w:.4f}' for w in weights]}")

    # 创建客户端数据加载器
    client_loaders = []
    for idx in active_client_indices:
        subset = Subset(train_dataset, idx)
        loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        client_loaders.append(loader)

    # FL训练
    exp_name = f"cw_{args.weight_mode}_aL{args.alpha_label}_aF{args.alpha_feature}_seed{args.seed}"
    if args.weight_mode == "cw_exp":
        exp_name += f"_b{args.beta}"
    if args.grad_clip > 0:
        exp_name += f"_clip{args.grad_clip}"
    result_path = os.path.join(args.save_dir, f"{exp_name}.json")

    if os.path.exists(result_path):
        try:
            with open(result_path) as f:
                existing = json.load(f)
            if 'final_accuracy' in existing:
                print(f"[SKIP] {exp_name} already exists, acc={existing['final_accuracy']:.4f}")
                return
        except:
            print(f"[OVERWRITE] corrupted file, re-running")

    global_model = get_model("cifar10", device)

    round_log = []
    for round_num in tqdm(range(1, args.num_rounds + 1), desc=exp_name):
        client_params_list = []
        for i in range(len(active_clients)):
            local_model = get_model("cifar10", device)
            set_model_params(local_model, get_model_params(global_model))
            train_client(local_model, client_loaders[i], device,
                        args.local_epochs, args.lr, grad_clip=args.grad_clip)
            client_params_list.append(get_model_params(local_model))

        if args.weight_mode == "median":
            new_params = median_aggregate(client_params_list)
        else:
            new_params = fedavg_aggregate(get_model_params(global_model),
                                           client_params_list, weights)
        set_model_params(global_model, new_params)

        loss, accuracy = evaluate(global_model, testloader, device)
        round_log.append({'round': round_num, 'loss': float(loss), 'accuracy': float(accuracy)})

        if round_num % 50 == 0 or round_num <= 3:
            print(f"  Round {round_num}: acc={accuracy:.4f}")

    # 保存结果
    result = {
        'dataset': 'cifar10',
        'weight_mode': args.weight_mode,
        'alpha_label': float(args.alpha_label),
        'alpha_feature': float(args.alpha_feature),
        'beta': float(args.beta) if args.weight_mode == "cw_exp" else None,
        'grad_clip': float(args.grad_clip),
        'final_accuracy': float(accuracy),
        'final_loss': float(loss),
        'client_js_joints': [float(v) for v in client_js_joints],
        'aggregation_weights': [float(w) for w in weights],
        'round_log': round_log,
    }

    os.makedirs(args.save_dir, exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nFinal accuracy: {accuracy:.4f}")
    print(f"Saved to {result_path}")


if __name__ == "__main__":
    main()
