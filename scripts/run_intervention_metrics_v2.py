"""干预实验 v2：对变换后的图像【真实重算特征与聚类】，检验 v1 的簇号偏移构造是否成立。

背景（2026-09-30 发现的问题）：
  v1 (`run_intervention.py` 第 197-204 行) 计算反色客户端的度量时，用的是
  `client_clusters + num_clusters`——把簇号整体挪到网格右半边。这是**手工构造**，
  不是测量：反色客户端越多，被挪的簇越多，JS_joint 必然单调上升。
  于是 §5.3 的 "JS_joint 单调上升 (R²=0.8513)" 有循环论证嫌疑。

本脚本的做法：
  1. 对每个客户端，按其是否 invert，构造**变换后的图像集**（反色图真的反色）；
  2. 把变换后的图全部过一遍 ResNet-18 → 得到**变换后的真实特征**；
  3. 在变换后的特征上**重新拟合 k-means**（全局聚类，与 FMNIST/CIFAR-100 口径一致）；
  4. 在同一套簇号下统计每个客户端的 P(y,c)，计算 JS_joint / JS_label / JS_feature。
     注意：反色与非反色客户端现在共用同一套簇（因为簇是在混合后的特征上全局聚的），
     所以 P_global 是"混合世界"的全局分布——这正是"客户端之间不一样"的正确度量口径。

与 v1 的关键区别：
  v1: 簇号固定（原图聚的），反色客户端手工 +K  →  JS_joint 是构造出来的
  v2: 簇号在【变换后的混合特征】上重新聚    →  JS_joint 是测出来的

计算量说明：只做一次特征提取 + 一次聚类（全 50000 张），不打乱 FL 训练，
所以本脚本只输出度量对比，不重跑训练。FL 训练用的仍是 v1 的结果。

用法:
  python scripts/run_intervention_metrics_v2.py
  python scripts/run_intervention_metrics_v2.py --invert_ratio 0.5 --seed 42
"""

import argparse
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from PIL import ImageOps
from sklearn.cluster import KMeans

from src.data.partition import dirichlet_partition
from src.models.feature_extractor import FeatureExtractor
from src.metrics.coverage import compute_joint_distribution, compute_all_metrics


class InvertedDataset(Dataset):
    """按 should_invert 标记，对样本做颜色反转（结果与 MixedTransformDataset 一致）。"""

    def __init__(self, base_dataset, indices, should_invert):
        self.base_dataset = base_dataset
        self.indices = indices
        self.should_invert = should_invert

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img, label = self.base_dataset[real_idx]
        if self.should_invert[idx]:
            pil_img = transforms.ToPILImage()(img)
            img = transforms.ToTensor()(ImageOps.invert(pil_img))
        return img, label


def main():
    ap = argparse.ArgumentParser(description="Intervention metrics v2 (re-extract + re-cluster)")
    ap.add_argument("--num_clients", type=int, default=10)
    ap.add_argument("--alpha_label", type=float, default=0.3)
    ap.add_argument("--invert_ratio", type=float, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=10, help="全局聚类的簇数（与 FMNIST/CIFAR-100 口径一致）")
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--save_dir", type=str, default="./results/intervention_v2")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | alpha_L={args.alpha_label} | invert_ratio={args.invert_ratio} | k={args.k}")

    # ---- 1. 载入原始图像（只做 ToTensor，不做 normalize，保持与 v1 一致） ----
    base_train = datasets.CIFAR10(root=args.data_root, train=True, download=True,
                                  transform=transforms.ToTensor())
    labels_all = np.array(base_train.targets)
    num_classes = 10
    n_total = len(base_train)

    # ---- 2. 用与原实验完全相同的分区（保证可比） ----
    client_indices = dirichlet_partition(labels_all, args.num_clients, args.alpha_label, args.seed)
    rng = np.random.default_rng(args.seed)
    n_invert = int(args.num_clients * args.invert_ratio)
    invert_clients = set(rng.choice(args.num_clients, size=n_invert, replace=False))
    print(f"Invert clients: {sorted(invert_clients)} ({len(invert_clients)}/{args.num_clients})")

    # ---- 3. 构造【变换后的完整训练集】并一次性提特征 ----
    # 先把"每个样本是否属于 invert 客户端"摊平到全库索引上
    is_inverted = np.zeros(n_total, dtype=bool)
    for cid in invert_clients:
        is_inverted[client_indices[cid]] = True
    print(f"Inverted samples: {is_inverted.sum()} / {n_total}")

    # 构造全库的变换后数据集（顺序与 base_train 一致，便于索引对齐）
    class _FullInverted(Dataset):
        def __init__(self, base, mask):
            self.base = base
            self.mask = mask

        def __len__(self):
            return len(self.base)

        def __getitem__(self, i):
            img, lb = self.base[i]
            if self.mask[i]:
                pil = transforms.ToPILImage()(img)
                img = transforms.ToTensor()(ImageOps.invert(pil))
            return img, lb

    full_ds = _FullInverted(base_train, is_inverted)
    loader = DataLoader(full_ds, batch_size=256, shuffle=False, num_workers=0)

    print("Extracting features on TRANSFORMED images (this is the key step)...")
    extractor = FeatureExtractor(device=device)
    feats, labs = extractor.extract_with_labels(loader)
    print(f"Features: {feats.shape}")

    assert np.array_equal(labs, labels_all), "标签顺序错位，特征与分区无法对齐"

    # ---- 4. 在变换后的特征上重新全局聚类 ----
    print(f"Re-clustering globally with k={args.k} on transformed features...")
    km = KMeans(n_clusters=args.k, random_state=args.seed, n_init=10)
    cluster_ids_new = km.fit_predict(feats)
    num_clusters = args.k

    # ---- 5. 用新簇号计算度量（与 v1 同一套 compute_all_metrics） ----
    P_global_new = compute_joint_distribution(labels_all, cluster_ids_new, num_classes, num_clusters)

    client_metrics_new = []
    n_samples = []
    for cid in range(args.num_clients):
        idx = client_indices[cid]
        Pi = compute_joint_distribution(labels_all[idx], cluster_ids_new[idx],
                                        num_classes, num_clusters)
        client_metrics_new.append(compute_all_metrics(Pi, P_global_new))
        n_samples.append(len(idx))

    total = sum(n_samples)
    w = [n / total for n in n_samples]
    agg_new = {key: float(sum(wi * m[key] for wi, m in zip(w, client_metrics_new)))
               for key in client_metrics_new[0]}

    # ---- 6. 对照：v1 的构造式度量（簇号偏移），用原始 k10 簇文件复现 ----
    old_cluster_file = "./results/feature_clusters_cifar10_k10.npz"
    agg_old = None
    if os.path.exists(old_cluster_file):
        cd_old = np.load(old_cluster_file)['cluster_ids']
        K_old = int(cd_old.max() + 1)          # 100
        P_global_old = compute_joint_distribution(labels_all, cd_old, num_classes, K_old)
        ms_old = []
        for cid in range(args.num_clients):
            idx = client_indices[cid]
            cl = cd_old[idx]
            if cid in invert_clients:
                P_i = compute_joint_distribution(labels_all[idx], cl + K_old,
                                                 num_classes, K_old * 2)
                Pg = np.zeros((num_classes, K_old * 2))
                Pg[:, :K_old] = P_global_old
                ms_old.append(compute_all_metrics(P_i, Pg))
            else:
                P_i = compute_joint_distribution(labels_all[idx], cl, num_classes, K_old)
                ms_old.append(compute_all_metrics(P_i, P_global_old))
        agg_old = {key: float(sum(wi * m[key] for wi, m in zip(w, ms_old)))
                   for key in ms_old[0]}

    # ---- 7. 输出对比 ----
    print("\n" + "=" * 68)
    print(f"invert_ratio = {args.invert_ratio}   ({len(invert_clients)}/{args.num_clients} clients)")
    print("=" * 68)
    print(f"{'metric':>16} | {'v1 (构造式,簇号+K)':>20} | {'v2 (真实重算)':>16}")
    print("-" * 68)
    for key in ["JS_joint", "JS_conditional", "JS_label", "JS_feature", "Cov_weighted"]:
        v1 = f"{agg_old[key]:.4f}" if agg_old else "n/a"
        print(f"{key:>16} | {v1:>20} | {agg_new[key]:>16.4f}")
    print("=" * 68)

    # 反色客户端的簇分布：看反色图究竟跑到了哪里
    print("\n反色客户端在新簇空间的落点（前 3 个反色客户端）:")
    for cid in sorted(invert_clients)[:3]:
        idx = client_indices[cid]
        sel = cluster_ids_new[idx]
        u, ct = np.unique(sel, return_counts=True)
        top = sorted(zip(ct, u), reverse=True)[:5]
        print(f"  client {cid}: {len(idx)} samples, 占用 {len(u)}/{num_clusters} 簇, "
              f"top5 = {[(int(c), int(n)) for n, c in top]}")

    # ---- 8. 落盘 ----
    os.makedirs(args.save_dir, exist_ok=True)
    out = {
        "invert_ratio": args.invert_ratio,
        "alpha_label": args.alpha_label,
        "seed": args.seed,
        "num_invert_clients": int(len(invert_clients)),
        "invert_clients": sorted(int(x) for x in invert_clients),
        "n_inverted_samples": int(is_inverted.sum()),
        "metrics_v1_constructed": agg_old,
        "metrics_v2_recomputed": agg_new,
        "k": args.k,
        "note": "v1 = cluster-index shift by +K (constructed); v2 = features re-extracted and re-clustered on transformed images (measured)",
    }
    path = os.path.join(args.save_dir,
                        f"intervention_v2_inv{args.invert_ratio}_seed{args.seed}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
