"""PACS盲区实验v2：公平对比

修复v1的问题：
- v1问题：uniform_photo只用photo域(1670张)，four_domain用全部4个域(9991张)
  数据量差异导致不公平对比
- v2修复：所有实验都使用全部9991张数据，只是分配方式不同
  - uniform：所有客户端从全部域随机分区（特征均匀）
  - two_domain：5个客户端从photo+art分配，5个从sketch+cartoon分配
  - four_domain：每个客户端从1个域分配（域隔离）
  - 关键：所有客户端数据量大致相同
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

from src.data.partition import dirichlet_partition
from src.utils.logging import ExperimentLogger


PACS_CLASSES = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']
PACS_DOMAINS = ['art_painting', 'cartoon', 'photo', 'sketch']


class PACSLocalDataset(Dataset):
    def __init__(self, root_dir, domains=None, transform=None):
        self.samples = []
        self.transform = transform
        if domains is None:
            domains = PACS_DOMAINS
        for domain in domains:
            domain_dir = os.path.join(root_dir, domain)
            if not os.path.exists(domain_dir):
                continue
            for class_idx, class_name in enumerate(PACS_CLASSES):
                class_dir = os.path.join(domain_dir, class_name)
                if not os.path.exists(class_dir):
                    continue
                for fname in os.listdir(class_dir):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        self.samples.append((os.path.join(class_dir, fname), class_idx, domain))
        print(f"  Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, domain = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label, domain

    def get_domain_indices(self):
        domain_indices = {}
        for i, (_, _, domain) in enumerate(self.samples):
            if domain not in domain_indices:
                domain_indices[domain] = []
            domain_indices[domain].append(i)
        return domain_indices

    def get_labels(self):
        return np.array([s[1] for s in self.samples])


class PACSClientDataset(Dataset):
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
    parser = argparse.ArgumentParser(description="PACS blind spot experiment v2")
    parser.add_argument("--pacs_root", type=str, default="./data/PACS")
    parser.add_argument("--num_clients", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--domain_mode", type=str, default="four_domain",
                        choices=["uniform", "two_domain", "four_domain"])
    parser.add_argument("--num_rounds", type=int, default=200)
    parser.add_argument("--local_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="./results/pacs_v2")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

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

    print("Loading PACS dataset...")
    full_pacs = PACSLocalDataset(args.pacs_root, transform=None)

    domain_indices = full_pacs.get_domain_indices()
    all_labels = full_pacs.get_labels()

    print(f"\nDomain distribution:")
    for d, idx_list in domain_indices.items():
        labels_d = all_labels[idx_list]
        unique, counts = np.unique(labels_d, return_counts=True)
        print(f"  {d}: {len(idx_list)} samples")

    # 核心修改：所有实验都用全部数据，只是分区方式不同
    client_indices = []
    client_domain_labels = []

    if args.domain_mode == "uniform":
        # 对照组：所有域混在一起，随机分区（每个客户端都有所有域的数据）
        all_idx = np.arange(len(full_pacs))
        partitions = dirichlet_partition(all_labels, args.num_clients, args.alpha, seed=args.seed)
        for p in partitions:
            client_indices.append(all_idx[p])
            client_domain_labels.append('mixed')

    elif args.domain_mode == "two_domain":
        # 2个客户端从photo+art分区，2个从sketch+cartoon分区
        group1_idx = np.concatenate([np.array(domain_indices['photo']),
                                      np.array(domain_indices['art_painting'])])
        group2_idx = np.concatenate([np.array(domain_indices['sketch']),
                                      np.array(domain_indices['cartoon'])])
        group1_labels = all_labels[group1_idx]
        group2_labels = all_labels[group2_idx]

        p1 = dirichlet_partition(group1_labels, 2, args.alpha, seed=args.seed)
        p2 = dirichlet_partition(group2_labels, 2, args.alpha, seed=args.seed + 1)

        for p in p1:
            client_indices.append(group1_idx[p])
            client_domain_labels.append('photo+art')
        for p in p2:
            client_indices.append(group2_idx[p])
            client_domain_labels.append('sketch+cartoon')

    elif args.domain_mode == "four_domain":
        # 每个客户端只用1个域（最大域隔离）
        for domain_name in PACS_DOMAINS:
            d_idx = np.array(domain_indices[domain_name])
            d_labels = all_labels[d_idx]
            # 每个域1个客户端，不做标签分区（该客户端拥有该域全部数据）
            client_indices.append(d_idx)
            client_domain_labels.append(domain_name)

    # 打印客户端信息
    print(f"\nDomain mode: {args.domain_mode}")
    total_samples = 0
    for i, (idx, domain) in enumerate(zip(client_indices, client_domain_labels)):
        client_labels = all_labels[idx]
        unique, counts = np.unique(client_labels, return_counts=True)
        # 统计域分布
        client_domains = [full_pacs.samples[j][2] for j in idx]
        domain_unique, domain_counts = np.unique(client_domains, return_counts=True)
        total_samples += len(idx)
        print(f"  Client {i}: domain={domain}, {len(idx)} samples, "
              f"domain_dist={dict(zip(domain_unique.tolist(), domain_counts.tolist()))}")
    print(f"  Total samples across clients: {total_samples}")

    # 创建DataLoader
    client_loaders = []
    for idx in client_indices:
        dataset = PACSClientDataset(full_pacs, idx, transform=train_transform)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        client_loaders.append(loader)

    # 测试集：用所有域
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
