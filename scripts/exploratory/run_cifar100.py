"""CIFAR-100跨数据集验证：在100类细粒度分类上验证JS_joint优越性

CIFAR-100特点：
- 100个细粒度类，联合分布网格=100×k=1000格（k=10）
- 比 CIFAR-10 (10×10=100格) 大10倍
- 能验证度量在更大网格、更稀疏分布下的稳定性

实验设计（8组）：
- 纯标签偏斜：aL∈{0.05, 0.3}，aF=1000（2组）
- 纯特征偏斜：aL=100，aF∈{0.05, 0.3}（2组）
- 组合偏斜：aL∈{0.05, 0.3} × aF∈{0.05, 0.3}（4组）
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


def extract_features_cifar100(dataset, device, batch_size=256):
    """用预训练ResNet-18提取CIFAR-100特征"""
    import torchvision.models as models_
    resnet = models_.resnet18(weights=models_.ResNet18_Weights.DEFAULT)
    resnet = resnet.to(device)
    resnet.eval()

    # 去掉最后的FC层
    feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    features_list = []

    # 需要调整transform
    feat_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    with torch.no_grad():
        for images, _ in loader:
            images = feat_transform(images).to(device)
            feat = feature_extractor(images)
            feat = feat.view(feat.size(0), -1)
            features_list.append(feat.cpu().numpy())

    return np.vstack(features_list)


def main():
    parser = argparse.ArgumentParser(description="CIFAR-100 cross-dataset validation")
    parser.add_argument("--alpha_label", type=float, required=True)
    parser.add_argument("--alpha_feature", type=float, required=True)
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--num_rounds", type=int, default=200)
    parser.add_argument("--local_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="./results/cifar100")
    parser.add_argument("--k_clusters", type=int, default=10)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}, aL={args.alpha_label}, aF={args.alpha_feature}")

    # 加载CIFAR-100
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    train_dataset = datasets.CIFAR100(root=args.data_root, train=True, download=True,
                                       transform=train_transform)
    test_dataset = datasets.CIFAR100(root=args.data_root, train=False, download=True,
                                      transform=test_transform)
    testloader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # 获取原始标签（用transform前的dataset）
    raw_train = datasets.CIFAR100(root=args.data_root, train=True, download=False,
                                   transform=transforms.ToTensor())
    labels = np.array(raw_train.targets)
    num_classes = 100

    # 提取特征并聚类
    cluster_file = os.path.join(args.save_dir, f"cifar100_clusters_k{args.k_clusters}.npz")
    if os.path.exists(cluster_file):
        print(f"Loading existing clusters from {cluster_file}")
        cluster_data = np.load(cluster_file)
        cluster_ids = cluster_data['cluster_ids']
        num_clusters = int(cluster_ids.max() + 1)
    else:
        print("Extracting features and clustering...")
        features = extract_features_cifar100(raw_train, device)
        print(f"Features shape: {features.shape}")

        kmeans = KMeans(n_clusters=args.k_clusters, random_state=args.seed, n_init=10)
        cluster_ids = kmeans.fit_predict(features)
        num_clusters = args.k_clusters

        os.makedirs(args.save_dir, exist_ok=True)
        np.savez(cluster_file, cluster_ids=cluster_ids)
        print(f"Saved clusters to {cluster_file}")

    print(f"Num classes: {num_classes}, Num clusters: {num_clusters}")
    print(f"Joint distribution grid: {num_classes}×{num_clusters} = {num_classes * num_clusters} cells")

    # 数据分区
    if args.alpha_feature < 100:
        client_indices = combined_partition(
            labels, cluster_ids, args.num_clients,
            args.alpha_label, args.alpha_feature, args.seed)
    else:
        client_indices = dirichlet_partition(
            labels, args.num_clients, args.alpha_label, args.seed)

    # 计算覆盖度度量
    P_global = compute_joint_distribution(labels, cluster_ids, num_classes, num_clusters)

    client_metrics = []
    n_samples = []
    for cid in range(args.num_clients):
        idx = client_indices[cid]
        client_labels = labels[idx]
        client_clusters = cluster_ids[idx]
        P_i = compute_joint_distribution(client_labels, client_clusters,
                                          num_classes, num_clusters)
        metrics = compute_all_metrics(P_i, P_global)
        client_metrics.append(metrics)
        n_samples.append(len(idx))

    # 加权平均度量
    total = sum(n_samples)
    weights = [n / total for n in n_samples]
    agg_metrics = {}
    for key in client_metrics[0]:
        val = sum(w * m[key] for w, m in zip(weights, client_metrics))
        agg_metrics[key] = float(val) if not np.isnan(val) else 0.0

    print(f"\nMetrics (weighted avg):")
    for k, v in agg_metrics.items():
        print(f"  {k}: {v:.4f}")

    # FL训练
    exp_name = f"cifar100_aL{args.alpha_label}_aF{args.alpha_feature}_seed{args.seed}"
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

    # 过滤空客户端
    active_clients = [cid for cid in range(args.num_clients) if len(client_indices[cid]) > 0]
    print(f"Active clients: {len(active_clients)}/{args.num_clients}")

    client_loaders = []
    active_client_indices = []
    for cid in active_clients:
        idx = client_indices[cid]
        active_client_indices.append(idx)
        subset = Subset(train_dataset, idx)
        loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        client_loaders.append(loader)

    global_model = get_model("cifar100", device)

    round_log = []
    for round_num in tqdm(range(1, args.num_rounds + 1), desc=exp_name):
        client_params_list = []
        num_samples_list = []
        for i, cid in enumerate(active_clients):
            local_model = get_model("cifar100", device)
            set_model_params(local_model, get_model_params(global_model))
            train_client(local_model, client_loaders[i], device,
                        args.local_epochs, args.lr)
            client_params_list.append(get_model_params(local_model))
            num_samples_list.append(len(active_client_indices[i]))

        new_params = fedavg_aggregate(get_model_params(global_model),
                                       client_params_list, num_samples_list)
        set_model_params(global_model, new_params)

        loss, accuracy = evaluate(global_model, testloader, device)
        round_log.append({'round': round_num, 'loss': float(loss), 'accuracy': float(accuracy)})

        if round_num % 50 == 0 or round_num <= 3:
            print(f"  Round {round_num}: acc={accuracy:.4f}")

    # 保存结果
    result = {
        'dataset': 'cifar100',
        'alpha_label': float(args.alpha_label),
        'alpha_feature': float(args.alpha_feature),
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
