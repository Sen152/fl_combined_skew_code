"""预变换域偏斜实验：对CIFAR-10做不可逆的视觉变换，创建真正的"虚拟域"

与之前style_blindspot的区别：
- style_blindspot：训练时随机增强（每个epoch不同），模型看到的是多样化数据
- 本脚本：预先对图像做不可逆变换（固定变换），不同客户端看到的是完全不同的视觉域

这就是"同类不同貌"：同样是"猫"，客户端A看到的是正常猫，客户端B看到的是灰度猫
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from PIL import Image, ImageFilter, ImageOps
from tqdm import tqdm

from src.data.partition import dirichlet_partition
from src.models.cnn import get_model, get_model_params, set_model_params
from src.utils.logging import ExperimentLogger


class PretransformedDataset(Dataset):
    """预变换数据集：对图像做不可逆变换后存储"""

    def __init__(self, base_dataset, indices, pre_transform, train_transform):
        """
        Args:
            base_dataset: 原始CIFAR-10数据集（用ToTensor()加载）
            indices: 样本索引
            pre_transform: 预变换函数（PIL Image → PIL Image）
            train_transform: 训练时增强（Tensor → Tensor）
        """
        self.data = []
        self.labels = []
        self.train_transform = train_transform

        print(f"  Pre-transforming {len(indices)} images...")
        for idx in indices:
            img, label = base_dataset[idx]
            # img是tensor，转回PIL
            pil_img = transforms.ToPILImage()(img)
            # 应用预变换
            transformed = pre_transform(pil_img)
            # 存储为tensor
            self.data.append(transforms.ToTensor()(transformed))
            self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        if self.train_transform:
            img = self.train_transform(img)
        return img, label


# 定义预变换函数（不可逆）
def pretransform_normal(img):
    """无变换"""
    return img

def pretransform_grayscale(img):
    """灰度化（保持3通道）"""
    return img.convert('L').convert('RGB')

def pretransform_high_contrast(img):
    """高对比度"""
    return ImageOps.autocontrast(img, cutoff=10)

def pretransform_posterize(img):
    """色调分离（减少色彩层次）"""
    return ImageOps.posterize(img, bits=2)

def pretransform_invert_colors(img):
    """反色"""
    return ImageOps.invert(img)

def pretransform_edge_enhance(img):
    """边缘增强"""
    return img.filter(ImageFilter.EDGE_ENHANCE_MORE)


PRETRANSFORMS = {
    'normal': pretransform_normal,
    'grayscale': pretransform_grayscale,
    'high_contrast': pretransform_high_contrast,
    'posterize': pretransform_posterize,
    'invert': pretransform_invert_colors,
    'edge_enhance': pretransform_edge_enhance,
}


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
    parser = argparse.ArgumentParser(description="Pre-transformed domain skew experiment")
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--domain_mode", type=str, default="two_domain",
                        choices=["uniform", "two_domain", "three_domain", "extreme",
                                 # 统一口径的像素级梯度（一律 5 正常 + 5 变换，位置固定）
                                 "grad_high_contrast", "grad_posterize",
                                 "grad_edge_enhance", "grad_grayscale",
                                 "grad_invert"])
    parser.add_argument("--num_rounds", type=int, default=200)
    parser.add_argument("--local_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="./results/domain_blindspot")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 加载原始CIFAR-10（只用ToTensor，不做normalize）
    print("Loading CIFAR-10...")
    base_dataset = datasets.CIFAR10(
        root=args.data_root, train=True, download=True,
        transform=transforms.ToTensor()
    )

    # 标签分区
    labels = np.array(base_dataset.targets)
    print(f"Dirichlet partition (alpha={args.alpha})")
    client_indices = dirichlet_partition(labels, args.num_clients, args.alpha, seed=args.seed)

    # 为每个客户端分配预变换
    if args.domain_mode == "uniform":
        # 对照组：所有客户端用normal
        client_transforms = ['normal'] * args.num_clients
    elif args.domain_mode == "two_domain":
        # 5个normal + 5个grayscale
        client_transforms = ['normal'] * 5 + ['grayscale'] * 5
    elif args.domain_mode == "three_domain":
        # 4个normal + 3个grayscale + 3个posterize
        client_transforms = ['normal'] * 4 + ['grayscale'] * 3 + ['posterize'] * 3
    elif args.domain_mode == "extreme":
        # 5个normal + 5个invert
        client_transforms = ['normal'] * 5 + ['invert'] * 5
    elif args.domain_mode.startswith("grad_"):
        # 统一梯度口径：前5个客户端 normal，后5个施加同一变换
        # 位置、数量完全一致，只有变换类型不同 → 横向可直接比较
        tname = args.domain_mode[len("grad_"):]
        assert tname in PRETRANSFORMS, f"unknown transform {tname}"
        client_transforms = ['normal'] * 5 + [tname] * 5

    # 训练时增强（在预变换之后应用）
    train_aug = transforms.Compose([
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        transforms.RandomErasing(p=0.2),  # 在tensor上做随机擦除代替RandomCrop
    ])

    # 创建DataLoader
    print(f"\nDomain mode: {args.domain_mode}")
    client_loaders = []
    for i, (idx, tname) in enumerate(zip(client_indices, client_transforms)):
        pre_t = PRETRANSFORMS[tname]
        dataset = PretransformedDataset(base_dataset, idx, pre_t, train_aug)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        client_loaders.append(loader)
        client_labels = labels[idx]
        unique, counts = np.unique(client_labels, return_counts=True)
        print(f"  Client {i}: transform={tname}, {len(idx)} samples, "
              f"labels={dict(zip(unique.tolist(), counts.tolist()))}")

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
    exp_name = f"domain_{args.domain_mode}_alpha{args.alpha}_seed{args.seed}"
    logger = ExperimentLogger(args.save_dir, exp_name)

    print(f"\nStarting FL: {args.num_rounds} rounds, domain_mode={args.domain_mode}")

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
