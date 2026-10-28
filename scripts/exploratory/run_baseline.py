"""运行FedAvg基线实验（纯PyTorch模拟，不依赖Flower/Ray）"""

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
from src.data.partition import dirichlet_partition, iid_partition
from src.models.cnn import get_model, get_model_params, set_model_params
from src.utils.logging import ExperimentLogger


def train_client(model, trainloader, device, local_epochs, lr):
    """客户端本地训练"""
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
    """评估模型"""
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
    """FedAvg聚合：按样本数加权平均"""
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
    parser = argparse.ArgumentParser(description="Run FL baseline experiment")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "fmnist"])
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.1, help="Dirichlet alpha (1000=iid)")
    parser.add_argument("--num_rounds", type=int, default=200)
    parser.add_argument("--local_epochs", type=int, default=2)
    parser.add_argument("--fraction_fit", type=float, default=1.0, help="每轮参与训练的客户端比例")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="./results/baselines")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 加载数据集
    print(f"Loading {args.dataset}...")
    train_dataset = get_dataset(args.dataset, train=True, data_root=args.data_root)
    testloader = get_test_dataloader(args.dataset, batch_size=256, data_root=args.data_root)

    # 数据分区
    labels = np.array([train_dataset[i][1] for i in range(len(train_dataset))])
    if args.alpha >= 100:
        print(f"IID partition (alpha={args.alpha})")
        client_indices = iid_partition(len(train_dataset), args.num_clients, seed=args.seed)
    else:
        print(f"Dirichlet partition (alpha={args.alpha})")
        client_indices = dirichlet_partition(labels, args.num_clients, args.alpha, seed=args.seed)

    # 打印每个客户端的数据分布
    for i, idx in enumerate(client_indices):
        client_labels = labels[idx]
        unique, counts = np.unique(client_labels, return_counts=True)
        print(f"  Client {i}: {len(idx)} samples, label dist: {dict(zip(unique, counts))}")

    # 创建每个客户端的DataLoader
    client_loaders = []
    for idx in client_indices:
        subset = Subset(train_dataset, idx)
        loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        client_loaders.append(loader)

    # 初始化全局模型
    global_model = get_model(args.dataset, device)
    num_fit_clients = max(1, int(args.num_clients * args.fraction_fit))

    # 实验日志
    exp_name = f"{args.dataset}_alpha{args.alpha}_clients{args.num_clients}_seed{args.seed}"
    logger = ExperimentLogger(args.save_dir, exp_name)

    print(f"\nStarting FL simulation: {args.num_rounds} rounds, {args.num_clients} clients")
    print(f"Alpha={args.alpha}, fraction_fit={args.fraction_fit}, local_epochs={args.local_epochs}, lr={args.lr}")

    # FL训练循环
    for round_num in tqdm(range(1, args.num_rounds + 1), desc="FL Training"):
        # 选择本轮参与的客户端
        if num_fit_clients < args.num_clients:
            selected = np.random.choice(args.num_clients, num_fit_clients, replace=False)
        else:
            selected = np.arange(args.num_clients)

        # 客户端本地训练
        client_params_list = []
        num_samples_list = []
        for cid in selected:
            local_model = get_model(args.dataset, device)
            set_model_params(local_model, get_model_params(global_model))
            train_client(local_model, client_loaders[cid], device, args.local_epochs, args.lr)
            client_params_list.append(get_model_params(local_model))
            num_samples_list.append(len(client_indices[cid]))

        # FedAvg聚合
        new_params = fedavg_aggregate(
            get_model_params(global_model), client_params_list, num_samples_list
        )
        set_model_params(global_model, new_params)

        # 评估
        loss, accuracy = evaluate(global_model, testloader, device)
        logger.log_round(round_num, {"loss": float(loss), "accuracy": float(accuracy)})

        if round_num % 10 == 0 or round_num <= 5:
            print(f"Round {round_num}: loss={loss:.4f}, accuracy={accuracy:.4f}")

    # 保存结果
    save_path = logger.save()
    print(f"\nResults saved to {save_path}")
    print(f"Final accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
