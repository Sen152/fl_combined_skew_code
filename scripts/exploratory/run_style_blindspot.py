"""风格偏斜盲区实验：用数据增强模拟真实的特征偏斜

核心思路：
- 对照组：所有客户端用标准CIFAR-10图像
- 实验组：不同客户端用不同视觉风格的图像（色彩偏移/灰度/模糊等）
- 标签分布完全相同（Dirichlet α=0.1）
- 如果实验组性能显著下降，说明"同类不同貌"才是真正的特征偏斜
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms, datasets
from tqdm import tqdm

from src.data.partition import dirichlet_partition
from src.models.cnn import get_model, get_model_params, set_model_params
from src.utils.logging import ExperimentLogger


# 定义不同客户端的视觉风格
STYLE_TRANSFORMS = {
    'normal': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ]),
    'color_jitter': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ]),
    'grayscale': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(num_output_channels=3),  # 保持3通道但变灰
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ]),
    'gaussian_blur': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.GaussianBlur(kernel_size=5, sigma=(1.0, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ]),
    'solarize': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomSolarize(threshold=128, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ]),
}


class StyleDataset(Dataset):
    """给子集应用特定风格的Dataset包装器"""

    def __init__(self, base_dataset, indices, style_transform):
        self.base_dataset = base_dataset
        self.indices = indices
        self.style_transform = style_transform
        # 缓存原始PIL图像和标签
        self._data = [(base_dataset[i][0], base_dataset[i][1]) for i in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self._data[idx]
        # img是已经经过base_transform的tensor，我们需要原始PIL
        # 所以直接从base_dataset重新获取
        pil_img = self.base_dataset.raw_data[self.indices[idx]]
        transformed = self.style_transform(pil_img)
        return transformed, label


class CIFAR10Raw(datasets.CIFAR10):
    """CIFAR10子类，保留原始PIL图像访问"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.data 是 numpy array, 需要转成PIL
        from PIL import Image
        self.raw_data = [Image.fromarray(img) for img in self.data]


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
    parser = argparse.ArgumentParser(description="Style-based blind spot experiment")
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.1, help="标签Dirichlet参数")
    parser.add_argument("--style_mode", type=str, default="mixed",
                        choices=["uniform", "mixed", "two_group", "extreme"],
                        help="""
                        uniform: 所有客户端用normal风格（对照组）
                        mixed: 每个客户端随机分配不同风格
                        two_group: 5个normal + 5个color_jitter
                        extreme: 5个normal + 5个grayscale
                        """)
    parser.add_argument("--num_rounds", type=int, default=200)
    parser.add_argument("--local_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="./results/style_blindspot")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 加载原始CIFAR-10（不应用transform，在StyleDataset中按客户端应用）
    print("Loading CIFAR-10 (raw)...")
    raw_train = CIFAR10Raw(root=args.data_root, train=True, download=True)

    # 标签分区
    labels = np.array(raw_train.targets)
    print(f"Dirichlet partition (alpha={args.alpha})")
    client_indices = dirichlet_partition(labels, args.num_clients, args.alpha, seed=args.seed)

    # 为每个客户端分配风格
    style_names = list(STYLE_TRANSFORMS.keys())
    if args.style_mode == "uniform":
        client_styles = ['normal'] * args.num_clients
    elif args.style_mode == "two_group":
        client_styles = ['normal'] * 5 + ['color_jitter'] * 5
    elif args.style_mode == "extreme":
        client_styles = ['normal'] * 5 + ['grayscale'] * 5
    elif args.style_mode == "mixed":
        rng = np.random.default_rng(args.seed)
        client_styles = [style_names[i % len(style_names)] for i in range(args.num_clients)]
        rng.shuffle(client_styles)

    print(f"\nStyle mode: {args.style_mode}")
    for i, (idx, style) in enumerate(zip(client_indices, client_styles)):
        client_labels = labels[idx]
        unique, counts = np.unique(client_labels, return_counts=True)
        print(f"  Client {i}: style={style}, {len(idx)} samples, "
              f"labels={dict(zip(unique.tolist(), counts.tolist()))}")

    # 创建DataLoader
    client_loaders = []
    for i, (idx, style) in enumerate(zip(client_indices, client_styles)):
        style_dataset = StyleDataset(raw_train, idx, STYLE_TRANSFORMS[style])
        loader = DataLoader(style_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        client_loaders.append(loader)

    # 测试集（标准transform）
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_dataset = datasets.CIFAR10(root=args.data_root, train=False, download=True,
                                     transform=test_transform)
    testloader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)

    # FL训练
    global_model = get_model("cifar10", device)
    exp_name = f"style_{args.style_mode}_alpha{args.alpha}_seed{args.seed}"
    logger = ExperimentLogger(args.save_dir, exp_name)

    print(f"\nStarting FL: {args.num_rounds} rounds, style_mode={args.style_mode}")

    for round_num in tqdm(range(1, args.num_rounds + 1), desc="FL Training"):
        client_params_list = []
        num_samples_list = []
        for cid in range(args.num_clients):
            local_model = get_model("cifar10", device)
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
