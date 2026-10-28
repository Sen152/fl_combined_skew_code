"""干预实验：固定标签偏斜，操纵特征偏斜强度，验证因果性

核心论证：JS_joint下降 → accuracy下降（剂量-反应关系）

实验设计：
- 固定alpha_label=0.3（中等标签偏斜）
- 用invert变换的混合比例控制特征偏斜强度
- invert_ratio ∈ {0.0, 0.25, 0.5, 0.75, 1.0}
  - 0.0: 无特征偏斜（对照组）
  - 0.25: 25%客户端收到invert图像
  - 0.5: 50%客户端收到invert图像
  - 0.75: 75%客户端收到invert图像
  - 1.0: 所有客户端收到invert图像（无特征偏斜，因为分布一致）

关键：invert_ratio=1.0时所有客户端看到同样的invert图像，
JS_feature应该接近0（无特征偏斜），这形成一个U型曲线验证。
真正的特征偏斜在invert_ratio=0.5时最大（一半原始一半invert）。
"""

import argparse
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, datasets
from PIL import ImageOps
from tqdm import tqdm

from src.data.partition import dirichlet_partition
from src.models.cnn import get_model, get_model_params, set_model_params
from src.metrics.coverage import compute_joint_distribution, compute_all_metrics
from src.utils.logging import ExperimentLogger


class MixedTransformDataset(Dataset):
    """混合变换数据集：部分样本原始，部分样本invert"""

    def __init__(self, base_dataset, indices, should_invert, train_transform=None):
        """
        Args:
            base_dataset: 原始CIFAR-10数据集
            indices: 样本索引
            should_invert: bool数组，标记哪些样本需要invert
            train_transform: 训练时增强
        """
        self.base_dataset = base_dataset
        self.indices = indices
        self.should_invert = should_invert
        self.train_transform = train_transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img, label = self.base_dataset[real_idx]

        if self.should_invert[idx]:
            # tensor → PIL → invert → tensor
            pil_img = transforms.ToPILImage()(img)
            inverted = ImageOps.invert(pil_img)
            img = transforms.ToTensor()(inverted)

        if self.train_transform:
            img = self.train_transform(img)
        return img, label


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


def main():
    parser = argparse.ArgumentParser(description="Intervention experiment")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--num_rounds", type=int, default=200)
    parser.add_argument("--local_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--alpha_label", type=float, default=0.3,
                        help="固定标签偏斜程度")
    parser.add_argument("--invert_ratio", type=float, required=True,
                        help="收到invert变换的客户端比例 (0.0~1.0)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="./results/intervention")
    parser.add_argument("--cluster_file", type=str,
                        default="./results/feature_clusters_cifar10_k10.npz")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}, alpha_label={args.alpha_label}, invert_ratio={args.invert_ratio}")

    # 加载特征聚类
    cluster_data = np.load(args.cluster_file)
    cluster_ids = cluster_data['cluster_ids']
    num_clusters = int(cluster_ids.max() + 1)

    # 加载数据集
    base_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.CIFAR10(root=args.data_root, train=True, download=True,
                                      transform=base_transform)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    test_dataset = datasets.CIFAR10(root=args.data_root, train=False, download=True,
                                     transform=test_transform)
    testloader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    labels = np.array([train_dataset.targets[i] for i in range(len(train_dataset))])
    num_classes = 10

    # Dirichlet分区（固定标签偏斜）
    client_indices = dirichlet_partition(labels, args.num_clients, args.alpha_label, args.seed)

    # 决定哪些客户端收到invert变换
    rng = np.random.default_rng(args.seed)
    n_invert_clients = int(args.num_clients * args.invert_ratio)
    invert_clients = set(rng.choice(args.num_clients, size=n_invert_clients, replace=False))

    print(f"\nInvert clients: {sorted(invert_clients)} ({len(invert_clients)}/{args.num_clients})")

    # 创建DataLoader
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    client_loaders = []
    for cid in range(args.num_clients):
        idx = client_indices[cid]
        should_invert = np.array([cid in invert_clients] * len(idx))
        dataset = MixedTransformDataset(train_dataset, idx, should_invert, train_transform)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        client_loaders.append(loader)

    # 计算覆盖度度量
    # 需要重新计算每个客户端的cluster_ids（因为invert改变了特征）
    # 简化方案：用原始cluster_ids + invert标记计算度量
    # invert客户端的cluster_ids偏移num_clusters（创建虚拟"域"）
    P_global = compute_joint_distribution(labels, cluster_ids, num_classes, num_clusters)

    client_metrics = []
    for cid in range(args.num_clients):
        idx = client_indices[cid]
        client_labels = labels[idx]
        client_clusters = cluster_ids[idx]

        if cid in invert_clients:
            # invert客户端的特征簇偏移，模拟域差异
            client_clusters_shifted = client_clusters + num_clusters
            P_i = compute_joint_distribution(client_labels, client_clusters_shifted,
                                              num_classes, num_clusters * 2)
            P_global_expanded = np.zeros((num_classes, num_clusters * 2))
            P_global_expanded[:, :num_clusters] = P_global
            metrics = compute_all_metrics(P_i, P_global_expanded)
        else:
            P_i = compute_joint_distribution(client_labels, client_clusters,
                                              num_classes, num_clusters)
            metrics = compute_all_metrics(P_i, P_global)

        client_metrics.append(metrics)

    # 加权平均度量
    n_samples = [len(idx) for idx in client_indices]
    total = sum(n_samples)
    weights = [n / total for n in n_samples]

    agg_metrics = {}
    for key in client_metrics[0]:
        agg_metrics[key] = sum(w * m[key] for w, m in zip(weights, client_metrics))

    print(f"\nMetrics (weighted avg):")
    for k, v in agg_metrics.items():
        print(f"  {k}: {v:.4f}")

    # FL训练
    exp_name = f"intervention_aL{args.alpha_label}_inv{args.invert_ratio}_seed{args.seed}"
    result_path = os.path.join(args.save_dir, f"{exp_name}.json")

    if os.path.exists(result_path):
        # 检查文件是否完整
        try:
            with open(result_path) as f:
                existing = json.load(f)
            if 'final_accuracy' in existing:
                print(f"[SKIP] {exp_name} already exists and is valid")
                print(f"Accuracy: {existing['final_accuracy']:.4f}")
                return
        except:
            print(f"[OVERWRITE] {exp_name} exists but is corrupted, re-running")

    global_model = get_model(args.dataset, device)

    round_log = []
    for round_num in tqdm(range(1, args.num_rounds + 1), desc=exp_name):
        client_params_list = []
        num_samples_list = []
        for cid in range(args.num_clients):
            local_model = get_model(args.dataset, device)
            set_model_params(local_model, get_model_params(global_model))
            train_client(local_model, client_loaders[cid], device,
                        args.local_epochs, args.lr)
            client_params_list.append(get_model_params(local_model))
            num_samples_list.append(len(client_indices[cid]))

        new_params = fedavg_aggregate(get_model_params(global_model),
                                       client_params_list, num_samples_list)
        set_model_params(global_model, new_params)

        loss, accuracy = evaluate(global_model, testloader, device)
        round_log.append({'round': round_num, 'loss': float(loss), 'accuracy': float(accuracy)})

        if round_num % 50 == 0 or round_num <= 3:
            print(f"  Round {round_num}: acc={accuracy:.4f}")

    # 保存结果
    result = {
        'alpha_label': float(args.alpha_label),
        'invert_ratio': float(args.invert_ratio),
        'num_invert_clients': int(len(invert_clients)),
        'invert_clients': [int(x) for x in sorted(list(invert_clients))],
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
