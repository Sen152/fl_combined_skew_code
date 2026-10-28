"""PACS盲区实验 - 从本地图片加载

如果PACS图片已下载到本地，直接加载
否则提示用户手动下载
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from src.utils.logging import ExperimentLogger


PACS_CLASSES = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']
PACS_DOMAINS = ['art_painting', 'cartoon', 'photo', 'sketch']


class PACSLocalDataset(Dataset):
    """从本地文件夹加载PACS数据集"""

    def __init__(self, root_dir, domains=None, transform=None):
        """
        Args:
            root_dir: PACS数据集根目录，应包含 art_painting/, cartoon/, photo/, sketch/ 子目录
            domains: 要加载的域列表，None表示加载全部
            transform: 图像变换
        """
        self.samples = []  # (image_path, label, domain)
        self.transform = transform

        if domains is None:
            domains = PACS_DOMAINS

        for domain in domains:
            domain_dir = os.path.join(root_dir, domain)
            if not os.path.exists(domain_dir):
                print(f"  Warning: {domain_dir} not found, skipping")
                continue

            for class_idx, class_name in enumerate(PACS_CLASSES):
                class_dir = os.path.join(domain_dir, class_name)
                if not os.path.exists(class_dir):
                    continue
                for fname in os.listdir(class_dir):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        self.samples.append((
                            os.path.join(class_dir, fname),
                            class_idx,
                            domain
                        ))

        print(f"  Loaded {len(self.samples)} samples from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, domain = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label, domain

    def get_domain_indices(self):
        """返回每个域的样本索引"""
        domain_indices = {}
        for i, (_, _, domain) in enumerate(self.samples):
            if domain not in domain_indices:
                domain_indices[domain] = []
            domain_indices[domain].append(i)
        return domain_indices

    def get_labels(self):
        """返回所有标签"""
        return np.array([s[1] for s in self.samples])


class PACSClientDataset(Dataset):
    """单个客户端的PACS数据集"""

    def __init__(self, base_dataset, indices, transform=None):
        self.base = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img, label, _ = self.base[real_idx]
        if self.transform:
            img = self.transform(img)
        return img, label


class CNNPACS(nn.Module):
    """CNN for PACS (227x227x3, 7 classes)"""

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
    parser = argparse.ArgumentParser(description="PACS blind spot experiment (local)")
    parser.add_argument("--pacs_root", type=str, default="./data/PACS",
                        help="PACS数据集根目录")
    parser.add_argument("--num_clients", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.5, help="标签Dirichlet参数")
    parser.add_argument("--domain_mode", type=str, default="four_domain",
                        choices=["uniform_photo", "two_domain", "four_domain", "all_mixed"])
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

    # 检查PACS数据是否存在
    if not os.path.exists(args.pacs_root):
        print(f"\n❌ PACS数据集未找到: {args.pacs_root}")
        print(f"请手动下载PACS数据集并解压到该目录")
        print(f"目录结构应为: {args.pacs_root}/photo/, {args.pacs_root}/art_painting/, etc.")
        print(f"\n下载方式：")
        print(f"1. 从原始论文作者页面下载: https://sketchx.eecs.qmul.ac.uk/downloads/")
        print(f"2. 或用HuggingFace镜像: HF_ENDPOINT=https://hf-mirror.com python -c \"from datasets import load_dataset; ds = load_dataset('flwrlabs/pacs', split='train')\"")
        print(f"3. 或从Google Drive下载（搜索PACS dataset）")
        return

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
    print("Loading PACS dataset...")
    full_pacs = PACSLocalDataset(args.pacs_root, transform=None)

    if len(full_pacs) == 0:
        print("❌ No samples loaded. Check your PACS directory structure.")
        return

    domain_indices = full_pacs.get_domain_indices()
    all_labels = full_pacs.get_labels()

    print(f"\nDomain distribution:")
    for d, idx_list in domain_indices.items():
        labels_d = all_labels[idx_list]
        unique, counts = np.unique(labels_d, return_counts=True)
        print(f"  {d}: {len(idx_list)} samples, class dist: {dict(zip(unique.tolist(), counts.tolist()))}")

    # 根据domain_mode分配客户端数据
    from src.data.partition import dirichlet_partition

    client_indices = []
    client_domain_labels = []

    if args.domain_mode == "uniform_photo":
        photo_indices = np.array(domain_indices['photo'])
        photo_labels = all_labels[photo_indices]
        partitions = dirichlet_partition(photo_labels, args.num_clients, args.alpha, seed=args.seed)
        for p in partitions:
            client_indices.append(photo_indices[p])
            client_domain_labels.append('photo')

    elif args.domain_mode == "two_domain":
        for domain_name in ['photo', 'photo', 'sketch', 'sketch']:
            d_indices = np.array(domain_indices[domain_name])
            d_labels = all_labels[d_indices]
            partitions = dirichlet_partition(d_labels, 1, args.alpha, seed=args.seed)
            client_indices.append(d_indices[partitions[0]])
            client_domain_labels.append(domain_name)

    elif args.domain_mode == "four_domain":
        for domain_name in PACS_DOMAINS:
            d_indices = np.array(domain_indices[domain_name])
            d_labels = all_labels[d_indices]
            partitions = dirichlet_partition(d_labels, 1, args.alpha, seed=args.seed)
            client_indices.append(d_indices[partitions[0]])
            client_domain_labels.append(domain_name)

    elif args.domain_mode == "all_mixed":
        all_indices = np.arange(len(full_pacs))
        partitions = dirichlet_partition(all_labels, args.num_clients, args.alpha, seed=args.seed)
        for p in partitions:
            client_indices.append(all_indices[p])
            client_domain_labels.append('mixed')

    # 打印客户端信息
    print(f"\nDomain mode: {args.domain_mode}")
    for i, (idx, domain) in enumerate(zip(client_indices, client_domain_labels)):
        client_labels = all_labels[idx]
        unique, counts = np.unique(client_labels, return_counts=True)
        print(f"  Client {i}: domain={domain}, {len(idx)} samples, "
              f"labels={dict(zip(unique.tolist(), counts.tolist()))}")

    # 创建DataLoader
    client_loaders = []
    for idx in client_indices:
        dataset = PACSClientDataset(full_pacs, idx, transform=train_transform)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        client_loaders.append(loader)

    # 测试集
    test_dataset = PACSClientDataset(full_pacs, np.arange(len(full_pacs)), transform=test_transform)
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
            local_model.load_state_dict(global_model.state_dict())
            train_client(local_model, client_loaders[cid], device, args.local_epochs, args.lr)
            client_params_list.append([val.cpu().numpy() for val in local_model.state_dict().values()])
            num_samples_list.append(len(client_indices[cid]))

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
