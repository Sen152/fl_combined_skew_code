"""ResNet-18 特征提取器"""

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm


class FeatureExtractor:
    """用预训练ResNet-18提取特征"""

    def __init__(self, device: str = "cuda"):
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # 移除最后FC层，保留到avgpool输出（512维）
        self.model = nn.Sequential(*list(model.children())[:-1])  # 去掉fc
        self.model = self.model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def extract(self, dataloader: DataLoader) -> np.ndarray:
        """
        提取特征

        Args:
            dataloader: 数据加载器

        Returns:
            (N, 512) 的特征矩阵
        """
        all_features = []
        for images, _ in tqdm(dataloader, desc="Extracting features"):
            images = images.to(self.device)
            feat = self.model(images).squeeze(-1).squeeze(-1)  # (B, 512)
            all_features.append(feat.cpu().numpy())
        return np.concatenate(all_features, axis=0)

    @torch.no_grad()
    def extract_with_labels(self, dataloader: DataLoader) -> tuple:
        """
        提取特征和对应标签

        Returns:
            (features, labels): (N, 512) 特征矩阵, (N,) 标签数组
        """
        all_features = []
        all_labels = []
        for images, labels in tqdm(dataloader, desc="Extracting features"):
            images = images.to(self.device)
            feat = self.model(images).squeeze(-1).squeeze(-1)
            all_features.append(feat.cpu().numpy())
            all_labels.append(labels.numpy())
        return np.concatenate(all_features, axis=0), np.concatenate(all_labels, axis=0)
