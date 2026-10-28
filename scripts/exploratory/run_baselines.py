"""FedProx和SCAFFOLD baseline实验

FedProx: 本地训练加proximal term, loss + μ/2 * ||w - w_global||²
SCAFFOLD: 方差缩减，维护server和client control variates

实验配置与CW-FedAvg相同：
- aL0.1_aF0.1, aL0.3_aF0.1, aL0.3_aF0.3
- 200轮, 10客户端, seed=42
"""

import argparse
import sys
import os
import json
import copy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from tqdm import tqdm

from src.data.partition import dirichlet_partition, combined_partition
from src.models.cnn import get_model, get_model_params, set_model_params
from src.metrics.coverage import compute_joint_distribution, compute_all_metrics
from sklearn.cluster import KMeans


def train_client_fedavg(model, trainloader, device, local_epochs, lr):
    """标准FedAvg本地训练"""
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


def train_client_fedprox(model, trainloader, device, local_epochs, lr, global_params, mu):
    """FedProx本地训练：loss + μ/2 * ||w - w_global||²"""
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # 将global_params转为tensor list用于proximal term
    global_tensors = [p.clone().detach() for p in global_params]

    for _ in range(local_epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)

            # Proximal term: μ/2 * Σ ||w_i - w_global_i||²
            proximal_term = 0.0
            for local_p, global_p in zip(model.parameters(), global_tensors):
                proximal_term += ((local_p - global_p.to(device)) ** 2).sum()
            loss = loss + (mu / 2) * proximal_term

            loss.backward()
            optimizer.step()


def train_client_scaffold(model, trainloader, device, local_epochs, lr,
                          global_params, c_global, c_local):
    """SCAFFOLD本地训练：方差缩减

    核心更新规则：
    v_i = v_i - c_i + c  (修正梯度方向)
    其中 c_i 是客户端control variate, c 是server control variate
    """
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # 保存训练前参数（移到cpu，与后续计算一致）
    params_before = [p.clone().detach().cpu() for p in model.parameters()]

    for _ in range(local_epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()

            # SCAFFOLD修正：减去client control + 加上server control
            with torch.no_grad():
                for p, c_g, c_l in zip(model.parameters(), c_global, c_local):
                    if c_g is not None and c_l is not None:
                        p.grad = p.grad - c_l.to(device) + c_g.to(device)

            optimizer.step()

    # 更新client control variate: c_i = c_i - c + (w_before - w_after) / (lr * K)
    # 简化版: c_i_new = c_i - c_global + (1/total_steps) * sum(grad - grad_corrected)
    # 实际实现用参数差近似
    total_steps = local_epochs * len(trainloader)
    new_c_local = []
    with torch.no_grad():
        for p_after, p_before, c_g, c_l in zip(model.parameters(), params_before, c_global, c_local):
            # Δw = w_after - w_before
            delta_w = p_after.detach().cpu() - p_before
            # c_local_new = -delta_w / (lr * total_steps) + c_global
            if c_g is not None:
                c_new = -delta_w / (lr * total_steps) + c_g.cpu() - (c_l.cpu() - c_g.cpu()) if c_l is not None else -delta_w / (lr * total_steps) + c_g.cpu()
            else:
                c_new = -delta_w / (lr * total_steps)
            new_c_local.append(c_new)

    return new_c_local


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
    """标准数据量加权聚合"""
    total_samples = sum(num_samples_list)
    aggregated = []
    for i in range(len(global_params)):
        weighted_sum = sum(
            params[i] * (n / total_samples)
            for params, n in zip(client_params_list, num_samples_list)
        )
        aggregated.append(weighted_sum)
    return aggregated


def main():
    parser = argparse.ArgumentParser(description="FedProx/SCAFFOLD baseline experiment")
    parser.add_argument("--method", type=str, required=True, choices=["fedprox", "scaffold"])
    parser.add_argument("--alpha_label", type=float, required=True)
    parser.add_argument("--alpha_feature", type=float, required=True)
    parser.add_argument("--mu", type=float, default=0.01, help="FedProx proximal term coefficient")
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--num_rounds", type=int, default=200)
    parser.add_argument("--local_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k_clusters", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="./results/baselines")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{args.method}] Device: {device}, aL={args.alpha_label}, aF={args.alpha_feature}, mu={args.mu}")

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

    # 计算覆盖度度量（用于结果记录）
    P_global = compute_joint_distribution(labels, cluster_ids, num_classes, num_clusters)
    client_metrics = []
    n_samples = []
    for cid in active_clients:
        idx = client_indices[cid]
        client_labels = labels[idx]
        client_clusters = cluster_ids[idx]
        P_i = compute_joint_distribution(client_labels, client_clusters, num_classes, num_clusters)
        metrics = compute_all_metrics(P_i, P_global)
        client_metrics.append(metrics)
        n_samples.append(len(idx))

    total = sum(n_samples)
    weights = [n / total for n in n_samples]
    agg_metrics = {}
    for key in client_metrics[0]:
        val = sum(w * m[key] for w, m in zip(weights, client_metrics))
        agg_metrics[key] = float(val) if not np.isnan(val) else 0.0

    # 创建客户端数据加载器
    client_loaders = []
    active_client_indices = []
    for cid in active_clients:
        idx = client_indices[cid]
        active_client_indices.append(idx)
        subset = Subset(train_dataset, idx)
        loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        client_loaders.append(loader)

    # FL训练
    if args.method == "fedprox":
        exp_name = f"fedprox_mu{args.mu}_aL{args.alpha_label}_aF{args.alpha_feature}_seed{args.seed}"
    else:
        exp_name = f"scaffold_aL{args.alpha_label}_aF{args.alpha_feature}_seed{args.seed}"

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

    # SCAFFOLD: 初始化control variates
    if args.method == "scaffold":
        # c_global: server control variate (与模型参数同形状)
        c_global = [torch.zeros_like(p).cpu() for p in global_model.parameters()]
        # c_local: 每个客户端的control variate
        c_locals = [[torch.zeros_like(p).cpu() for p in global_model.parameters()]
                    for _ in range(len(active_clients))]

    round_log = []
    for round_num in tqdm(range(1, args.num_rounds + 1), desc=exp_name):
        client_params_list = []

        if args.method == "fedprox":
            # FedProx: 每个客户端本地训练加proximal term
            global_params = list(global_model.parameters())
            for i in range(len(active_clients)):
                local_model = get_model("cifar10", device)
                set_model_params(local_model, get_model_params(global_model))
                train_client_fedprox(local_model, client_loaders[i], device,
                                     args.local_epochs, args.lr, global_params, args.mu)
                client_params_list.append(get_model_params(local_model))

        elif args.method == "scaffold":
            # SCAFFOLD: 方差缩减
            for i in range(len(active_clients)):
                local_model = get_model("cifar10", device)
                set_model_params(local_model, get_model_params(global_model))
                new_c_local = train_client_scaffold(
                    local_model, client_loaders[i], device,
                    args.local_epochs, args.lr,
                    list(global_model.parameters()), c_global, c_locals[i])
                client_params_list.append(get_model_params(local_model))
                c_locals[i] = new_c_local

            # 更新server control variate
            # c_global = c_global + 1/N * Σ(c_local_i - c_global)
            n_active = len(active_clients)
            with torch.no_grad():
                for j in range(len(c_global)):
                    delta = sum(c_locals[i][j] - c_global[j] for i in range(n_active)) / n_active
                    c_global[j] = c_global[j] + delta

        # 聚合
        new_params = fedavg_aggregate(get_model_params(global_model),
                                       client_params_list, n_samples)
        set_model_params(global_model, new_params)

        loss, accuracy = evaluate(global_model, testloader, device)
        round_log.append({'round': round_num, 'loss': float(loss), 'accuracy': float(accuracy)})

        if round_num % 50 == 0 or round_num <= 3:
            print(f"  Round {round_num}: acc={accuracy:.4f}")

    # 保存结果
    result = {
        'dataset': 'cifar10',
        'method': args.method,
        'alpha_label': float(args.alpha_label),
        'alpha_feature': float(args.alpha_feature),
        'mu': float(args.mu) if args.method == "fedprox" else None,
        'final_accuracy': float(accuracy),
        'final_loss': float(loss),
        'metrics': {k: float(v) for k, v in agg_metrics.items()},
        'round_log': round_log,
    }

    os.makedirs(args.save_dir, exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nFinal accuracy: {accuracy:.4f}")
    print(f"Saved to {result_path}")


if __name__ == "__main__":
    main()
