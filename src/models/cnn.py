"""CNN模型：Flower example标准实现"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNCifar10(nn.Module):
    """标准CNN for CIFAR-10 (32x32x3 input, 10 classes)"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 32x16x16
        x = self.pool(F.relu(self.conv2(x)))  # 64x8x8
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class CNNFMnist(nn.Module):
    """标准CNN for FMNIST (28x28x1 input, 10 classes)"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 32x14x14
        x = self.pool(F.relu(self.conv2(x)))  # 64x7x7
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class CNNCifar100(nn.Module):
    """标准CNN for CIFAR-100 (32x32x3 input, 100 classes)"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 100)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 32x16x16
        x = self.pool(F.relu(self.conv2(x)))  # 64x8x8
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def get_model(dataset_name: str, device: str = "cuda") -> nn.Module:
    """获取模型"""
    if dataset_name == "cifar10":
        model = CNNCifar10()
    elif dataset_name == "fmnist":
        model = CNNFMnist()
    elif dataset_name == "cifar100":
        model = CNNCifar100()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return model.to(device)


def get_model_params(model: nn.Module) -> list:
    """获取模型参数为numpy数组列表"""
    return [val.cpu().numpy() for val in model.state_dict().values()]


def set_model_params(model: nn.Module, params: list) -> nn.Module:
    """从numpy数组列表设置模型参数"""
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)
    return model
