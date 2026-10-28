"""PACS v3：留出域（held-out）评估——修复v2的DEF-06缺陷

v2缺陷：test loader用np.arange(len(full_pacs))，9991张全部参训，
99.93%是训练集准确率，不是泛化性能。

v3修复：
- 每个域内分层80/20划分（split_seed=2026，独立于训练seed）
- 所有配置共用同一个留出测试集（公平对比）
- 训练只用80%子集，分区方式与v2一致（uniform/two_domain/four_domain）
- 结束时输出整体+分域的留出准确率
- 记录设备名（Q-1）

判定（对应D-4）：若four_domain ≈ uniform（留出）→ "风格偏移无害"升格回贡献；
若four_domain显著低于uniform → 降级为观察。
"""

import argparse
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
torch.set_num_threads(4)  # 限制intra-op并行，防止多进程并行训练时CPU跑满过热
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


def stratified_split(domain_indices, split_seed=2026, test_frac=0.2):
    """每域内部shuffle后80/20划分，所有配置共用同一split"""
    rng = np.random.default_rng(split_seed)
    train_idx, test_idx = [], []
    for d, idx_list in domain_indices.items():
        idx = np.array(idx_list)
        rng.shuffle(idx)
        n_test = max(1, int(len(idx) * test_frac))
        test_idx.extend(idx[:n_test].tolist())
        train_idx.extend(idx[n_test:].tolist())
    return np.array(sorted(train_idx)), np.array(sorted(test_idx))


def main():
    parser = argparse.ArgumentParser(description="PACS v3: held-out evaluation")
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
    parser.add_argument("--split_seed", type=int, default=2026)
    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument("--save_dir", type=str, default="./results/pacs_v3")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        device_name = torch.cuda.get_device_name(0)
    else:
        device_name = "cpu"
    print(f"Device: {device} ({device_name})")

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

    # 核心：分层留出划分（所有配置共用）
    train_split, test_split = stratified_split(domain_indices, args.split_seed, args.test_frac)
    print(f"\nHeld-out split (split_seed={args.split_seed}): "
          f"train={len(train_split)}, test={len(test_split)}")
    train_domain_mask = np.zeros(len(full_pacs), dtype=bool)
    train_domain_mask[train_split] = True
    for d, idx_list in domain_indices.items():
        n_tr = train_domain_mask[idx_list].sum()
        print(f"  {d}: train={n_tr}, test={len(idx_list)-n_tr}")

    train_domain_indices = {}
    for d, idx_list in domain_indices.items():
        idx_arr = np.array(idx_list)
        train_domain_indices[d] = idx_arr[train_domain_mask[idx_arr]]

    # 分区（与v2逻辑一致，但只在train split上）
    client_indices = []
    if args.domain_mode == "uniform":
        partitions = dirichlet_partition(all_labels[train_split], args.num_clients,
                                          args.alpha, seed=args.seed)
        for p in partitions:
            client_indices.append(train_split[p])
    elif args.domain_mode == "two_domain":
        group1_idx = np.concatenate([train_domain_indices['photo'],
                                      train_domain_indices['art_painting']])
        group2_idx = np.concatenate([train_domain_indices['sketch'],
                                      train_domain_indices['cartoon']])
        group1_labels = all_labels[group1_idx]
        group2_labels = all_labels[group2_idx]
        p1 = dirichlet_partition(group1_labels, 2, args.alpha, seed=args.seed)
        p2 = dirichlet_partition(group2_labels, 2, args.alpha, seed=args.seed + 1)
        for p in p1:
            client_indices.append(group1_idx[p])
        for p in p2:
            client_indices.append(group2_idx[p])
    elif args.domain_mode == "four_domain":
        for domain_name in PACS_DOMAINS:
            client_indices.append(train_domain_indices[domain_name])

    print(f"\nDomain mode: {args.domain_mode}")
    for i, idx in enumerate(client_indices):
        print(f"  Client {i}: {len(idx)} samples")

    client_loaders = []
    for idx in client_indices:
        dataset = PACSClientDataset(full_pacs, idx, transform=train_transform)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        client_loaders.append(loader)

    # 留出测试集（所有域混合）+ 分域测试集
    test_dataset = PACSClientDataset(full_pacs, test_split, transform=test_transform)
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    domain_testloaders = {}
    for d, idx_list in domain_indices.items():
        idx_arr = np.array(idx_list)
        held = idx_arr[~train_domain_mask[idx_arr]]
        domain_testloaders[d] = DataLoader(
            PACSClientDataset(full_pacs, held, transform=test_transform),
            batch_size=64, shuffle=False, num_workers=0)

    global_model = CNNPACS().to(device)
    exp_name = f"pacs_v3_{args.domain_mode}_alpha{args.alpha}_clients{args.num_clients}_seed{args.seed}"
    logger = ExperimentLogger(args.save_dir, exp_name)

    print(f"\nStarting FL: {args.num_rounds} rounds, domain_mode={args.domain_mode}")

    for round_num in tqdm(range(1, args.num_rounds + 1), desc=exp_name):
        client_params_list = []
        num_samples_list = []
        for cid in range(len(client_indices)):
            local_model = CNNPACS().to(device)
            local_model.load_state_dict(global_model.state_dict())
            train_client(local_model, client_loaders[cid], device,
                         args.local_epochs, args.lr)
            client_params_list.append([val.cpu().numpy() for val in local_model.state_dict().values()])
            num_samples_list.append(len(client_indices[cid]))

        global_params = [val.cpu().numpy() for val in global_model.state_dict().values()]
        new_params = fedavg_aggregate(global_params, client_params_list, num_samples_list)
        state_dict = {k: torch.tensor(v) for k, v in zip(global_model.state_dict().keys(), new_params)}
        global_model.load_state_dict(state_dict)

        loss, accuracy = evaluate(global_model, testloader, device)
        logger.log_round(round_num, {"loss": float(loss), "accuracy": float(accuracy)})

        if round_num % 20 == 0 or round_num <= 5:
            print(f"Round {round_num}: held-out loss={loss:.4f}, held-out acc={accuracy:.4f}")

    # 最终：整体+分域留出准确率
    final_loss, final_acc = evaluate(global_model, testloader, device)
    per_domain = {}
    for d, dl in domain_testloaders.items():
        _, acc_d = evaluate(global_model, dl, device)
        per_domain[d] = float(acc_d)

    print(f"\nFinal HELD-OUT accuracy: {final_acc:.4f}")
    for d, a in per_domain.items():
        print(f"  {d}: {a:.4f}")

    save_path = logger.save()
    # 追加结构化信息
    with open(save_path) as f:
        result = json.load(f) if save_path.endswith('.json') else None
    extra = {
        'device': device_name,
        'domain_mode': args.domain_mode,
        'split_seed': args.split_seed,
        'test_frac': args.test_frac,
        'n_train': int(len(train_split)),
        'n_test': int(len(test_split)),
        'heldout_final_accuracy': float(final_acc),
        'heldout_per_domain': per_domain,
    }
    meta_path = os.path.join(args.save_dir, exp_name + "_meta.json")
    with open(meta_path, 'w') as f:
        json.dump(extra, f, indent=2)
    print(f"Meta saved to {meta_path}")


if __name__ == "__main__":
    main()
