"""特征提取 + k-means聚类脚本"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score

from src.data.datasets import get_dataset
from src.models.feature_extractor import FeatureExtractor


def main():
    parser = argparse.ArgumentParser(description="Extract features and cluster")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--k_per_class", type=int, default=10, help="每个类别的特征簇数")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="./results")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 加载数据集（用无增强的transform）
    from torchvision import transforms
    if args.dataset == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ])

    from torchvision import datasets
    if args.dataset == "cifar10":
        dataset = datasets.CIFAR10(root=args.data_root, train=True, download=True, transform=transform)
        num_classes = 10
    else:
        dataset = datasets.FashionMNIST(root=args.data_root, train=True, download=True, transform=transform)
        num_classes = 10

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 提取特征
    print("Extracting features with ResNet-18...")
    extractor = FeatureExtractor(device=device)
    features, labels = extractor.extract_with_labels(loader)
    print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")

    # 按标签分组聚类
    total_clusters = num_classes * args.k_per_class
    cluster_ids = np.zeros(len(labels), dtype=int)

    print(f"\nClustering: {args.k_per_class} clusters per class, {total_clusters} total")
    for c in range(num_classes):
        mask = labels == c
        class_features = features[mask]
        print(f"  Class {c}: {mask.sum()} samples, clustering into {args.k_per_class} groups...")

        kmeans = MiniBatchKMeans(n_clusters=args.k_per_class, random_state=42, batch_size=1024)
        local_ids = kmeans.fit_predict(class_features)

        # 映射到全局cluster_id
        cluster_ids[mask] = local_ids + c * args.k_per_class

        # 轮廓系数
        if len(class_features) > args.k_per_class * 2:
            sample_idx = np.random.choice(len(class_features), min(5000, len(class_features)), replace=False)
            sil = silhouette_score(class_features[sample_idx], local_ids[sample_idx])
            print(f"    Silhouette score: {sil:.4f}")

    # 保存结果
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(
        args.save_dir,
        f"feature_clusters_{args.dataset}_k{args.k_per_class}.npz"
    )
    np.savez(save_path, features=features, labels=labels, cluster_ids=cluster_ids)
    print(f"\nSaved to {save_path}")

    # 统计
    unique_clusters = np.unique(cluster_ids)
    print(f"\nTotal unique clusters: {len(unique_clusters)}")
    print(f"Cluster ID range: {cluster_ids.min()} - {cluster_ids.max()}")

    # 每个簇的样本数
    _, counts = np.unique(cluster_ids, return_counts=True)
    print(f"Samples per cluster: min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f}")


if __name__ == "__main__":
    main()
