"""CIFAR-10 / FMNIST 数据集加载"""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_transforms(dataset_name: str, train: bool = True) -> transforms.Compose:
    """获取数据增强和预处理"""
    if dataset_name == "cifar10":
        if train:
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ])
    elif dataset_name == "fmnist":
        if train:
            return transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,)),
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,)),
            ])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_dataset(dataset_name: str, train: bool = True, data_root: str = "./data"):
    """获取完整数据集"""
    if dataset_name == "cifar10":
        return datasets.CIFAR10(
            root=data_root, train=train, download=True,
            transform=get_transforms(dataset_name, train)
        )
    elif dataset_name == "fmnist":
        return datasets.FashionMNIST(
            root=data_root, train=train, download=True,
            transform=get_transforms(dataset_name, train)
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_client_dataloader(dataset_name: str, indices: list, batch_size: int = 32,
                          train: bool = True, data_root: str = "./data") -> DataLoader:
    """获取单个客户端的DataLoader"""
    full_dataset = get_dataset(dataset_name, train, data_root)
    subset = Subset(full_dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=train, num_workers=0)


def get_test_dataloader(dataset_name: str, batch_size: int = 256,
                        data_root: str = "./data") -> DataLoader:
    """获取测试集DataLoader"""
    test_dataset = get_dataset(dataset_name, train=False, data_root=data_root)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
