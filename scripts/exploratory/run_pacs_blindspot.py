"""PACS数据集盲区实验：用自然域偏移验证特征偏斜的影响

PACS = Photo + Art_painting + Cartoon + Sketch
7个类 × 4个域 = 天然的"同P(y)不同P(c|y)"场景

实验设计：
- 对照组：所有客户端只用Photo域（特征均匀）
- 实验组1：不同客户端用不同域（如5个Photo + 5个Sketch）
- 实验组2：4个客户端各用1个域（最大域偏移）
- 标签分布用Dirichlet控制，确保P(y)相同
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms
from tqdm import tqdm

from src.data.partition import dirichlet_partition
from src.utils.logging import ExperimentLogger


# PACS的7个类别
PACS_CLASSES = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']
PACS_DOMAINS = ['art_painting', 'cartoon', 'photo', 'sketch']


class PACSDataset(Dataset):
    """PACS数据集（从HuggingFace加载）"""

    def __init__(self, domain: str = None, transform=None, split='train'):
        from datasets import load_dataset
        print(f"Loading PACS dataset from HuggingFace...")
        ds = load_dataset("flwrlabs/pacs", split="train", trust_remote_code=True)

        self.samples = []
        self.labels = []
        self.domains = []

        for item in ds:
            item_domain = item['domain']
            if domain is not None and item_domain != domain:
                continue
            self.samples.append(item['image'])
            self.labels.append(item['label'])
            self.domains.append(item_domain)

        self.labels = np.array(self.labels)
        self.transform = transform
        print(f"  Loaded {len(self.samples)} samples"
              f"{f' from domain={domain}' if domain else ''}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = self.samples[idx]
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label


class PACSFilteredDataset(Dataset):
    """从预加载的PACS数据中按索引过滤"""

    def __init__(self, pacs_dataset, indices, transform=None):
        self.pacs = pacs_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img = self.pacs.samples[real_idx]
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.pacs.labels[real_idx]


class CNNPACS(nn.Module):
    """CNN for PACS (227x227x3 input, 7 classes)"""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 128, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 7),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


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
    parser = argparse.ArgumentParser(description="PACS blind spot experiment")
    parser.add_argument("--num_clients", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.1, help="标签Dirichlet参数")
    parser.add_argument("--domain_mode", type=str, default="mixed",
                        choices=["uniform_photo", "two_domain", "four_domain", "all_mixed"],
                        help="""
                        uniform_photo: 所有客户端只用Photo域（对照组）
                        two_domain: 2个Photo + 2个Sketch（中等域偏移）
                        four_domain: 每个客户端1个域（最大域偏移）
                        all_mixed: 所有域混在一起再分区
                        """)
    parser.add_argument("--num_rounds", type=int, default=200)
    parser.add_argument("--local_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="./results/pacs_blindspot")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 数据增强
    train_transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.RandomCrop(227, padding=16),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # 加载PACS
    full_pacs = PACSDataset(transform=None)  # 不应用transform，后面按客户端应用

    # 按域分组索引
    domain_indices = {}
    for i, d in enumerate(full_pacs.domains):
        if d not in domain_indices:
            domain_indices[d] = []
        domain_indices[d].append(i)

    print(f"\nDomain distribution:")
    for d, idx_list in domain_indices.items():
        labels_d = full_pacs.labels[idx_list]
        unique, counts = np.unique(labels_d, return_counts=True)
        print(f"  {d}: {len(idx_list)} samples, class dist: {dict(zip(unique.tolist(), counts.tolist()))}")

    # 根据domain_mode分配客户端数据
    client_indices = []
    client_domain_labels = []

    if args.domain_mode == "uniform_photo":
        # 对照组：所有客户端只用Photo域
        photo_indices = np.array(domain_indices['photo'])
        photo_labels = full_pacs.labels[photo_indices]
        partitions = dirichlet_partition(photo_labels, args.num_clients, args.alpha, seed=args.seed)
        for p in partitions:
            client_indices.append(photo_indices[p])
            client_domain_labels.append('photo')

    elif args.domain_mode == "two_domain":
        # 2个Photo + 2个Sketch
        for domain_name in ['photo', 'photo', 'sketch', 'sketch']:
            d_indices = np.array(domain_indices[domain_name])
            d_labels = full_pacs.labels[d_indices]
            # 每个域1个客户端，用Dirichlet分区
            partitions = dirichlet_partition(d_labels, 1, args.alpha, seed=args.seed)
            client_indices.append(d_indices[partitions[0]])
            client_domain_labels.append(domain_name)

    elif args.domain_mode == "four_domain":
        # 每个客户端1个域（最大域偏移）
        for domain_name in PACS_DOMAINS:
            d_indices = np.array(domain_indices[domain_name])
            d_labels = full_pacs.labels[d_indices]
            partitions = dirichlet_partition(d_labels, 1, args.alpha, seed=args.seed)
            client_indices.append(d_indices[partitions[0]])
            client_domain_labels.append(domain_name)

    elif args.domain_mode == "all_mixed":
        # 所有域混在一起再分区
        all_indices = np.arange(len(full_pacs))
        all_labels = full_pacs.labels
        partitions = dirichlet_partition(all_labels, args.num_clients, args.alpha, seed=args.seed)
        for p in partitions:
            client_indices.append(all_indices[p])
            client_domain_labels.append('mixed')

    # 打印客户端信息
    print(f"\nDomain mode: {args.domain_mode}")
    for i, (idx, domain) in enumerate(zip(client_indices, client_domain_labels)):
        client_labels = full_pacs.labels[idx]
        unique, counts = np.unique(client_labels, return_counts=True)
        # 统计该客户端的域分布
        client_domains = [full_pacs.domains[j] for j in idx]
        domain_unique, domain_counts = np.unique(client_domains, return_counts=True)
        print(f"  Client {i}: domain={domain}, {len(idx)} samples, "
              f"labels={dict(zip(unique.tolist(), counts.tolist()))}, "
              f"domain_dist={dict(zip(domain_unique.tolist(), domain_counts.tolist()))}")

    # 创建DataLoader
    client_loaders = []
    for idx in client_indices:
        dataset = PACSFilteredDataset(full_pacs, idx, transform=train_transform)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        client_loaders.append(loader)

    # 测试集：用所有域
    test_dataset = PACSFilteredDataset(
        full_pacs, np.arange(len(full_pacs)), transform=test_transform
    )
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    # FL训练
    global_model = CNNPACS().to(device)
    exp_name = f"pacs_{args.domain_mode}_alpha{args.alpha}_clients{args.num_clients}_seed{args.seed}"
    logger = ExperimentLogger(args.save_dir, exp_name)

    print(f"\nStarting FL: {args.num_rounds} rounds, domain_mode={args.domain_mode}")

    for round_num in tqdm(range(1, args.num_rounds + 1), desc="FL Training"):
        client_params_list = []
        num_samples_list = []
        for cid in range(len(client_indices)):
            local_model = CNNPACS().to(device)
            # 复制全局参数
            local_model.load_state_dict(global_model.state_dict())
            train_client(local_model, client_loaders[cid], device, args.local_epochs, args.lr)
            client_params_list.append([val.cpu().numpy() for val in local_model.state_dict().values()])
            num_samples_list.append(len(client_indices[cid]))

        # 聚合
        global_params = [val.cpu().numpy() for val in global_model.state_dict().values()]
        new_params = fedavg_aggregate(global_params, client_params_list, num_samples_list)
        state_dict = {k: torch.tensor(v) for k, v in zip(global_model.state_dict().keys(), new_params)}
        global_model.load_state_dict(state_dict)

        loss, accuracy = evaluate(global_model, testloader, device)
        logger.log_round(round_num, {"loss": float(loss), "accuracy": float(accuracy)})

        if round_num % 20 == 0 or round_num <= 5:
            print(f"Round {round_num}: loss={loss:.4f}, accuracy={accuracy:.4f}")

    save_path = logger.save()
    print(f"\nResults saved to {save_path}")
    print(f"Final accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
