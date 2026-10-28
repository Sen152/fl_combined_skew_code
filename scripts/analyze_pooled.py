"""池化配对检验：CW-exp vs FedAvg，全部4配置×全部seed，零挑选

预注册规则（2026-09-23，跑前定死）：
- 池 = 4配置 × 全部seed = 26对，一对不挑
- 主检验 = 全部26对 Wilcoxon（秩检验，崩溃对只是最大正秩，不被数值放大）
- 敏感性 = 去掉1个崩溃对的25对
- 双侧为主，单侧附报（方向假设有理论依据：下调高JS客户端权重不应有害）
- 升级条件 = 4个配置的差值均值方向全部为正 → CW定位可写"跨配置方向一致的小幅改进（n.s.）"
"""

import json
import os
import numpy as np
from scipy import stats

SEEDS_10 = [42, 123, 2024, 7, 99, 555, 777, 2025, 3407, 314]
SEEDS_3 = [42, 123, 2024]

# (aL, aF, 目录, seeds) —— 全部已有CW-exp vs FedAvg配对的配置，无遗漏无挑选
CONFIGS = [
    (0.1, 0.1, "results/cw_fedavg", SEEDS_10),
    (0.05, 0.1, "results/stress", SEEDS_10),
    (0.3, 0.1, "results/cw_fedavg", SEEDS_3),
    (0.3, 0.3, "results/cw_fedavg", SEEDS_3),
]

def load_acc(path):
    if not os.path.exists(path):
        return None, False
    with open(path) as f:
        d = json.load(f)
    acc = d["final_accuracy"]
    crashed = acc < 0.2 or str(d.get("final_loss", "")) == "NaN"
    return acc, crashed

pairs = []  # (config_tag, seed, fa, cw, diff, fa_crashed)
for aL, aF, d, seeds in CONFIGS:
    for s in seeds:
        fa, fa_cr = load_acc(f"{d}/cw_data_aL{aL}_aF{aF}_seed{s}.json")
        cw, cw_cr = load_acc(f"{d}/cw_cw_exp_aL{aL}_aF{aF}_seed{s}_b5.0.json")
        if fa is None or cw is None:
            print(f"[缺失] aL{aL}/aF{aF} seed{s}")
            continue
        pairs.append((f"aL{aL}/aF{aF}", s, fa, cw, cw - fa, fa_cr))

print("=" * 70)
print(f"池化配对检验：全部{len(pairs)}对（预注册：零挑选）")
print("=" * 70)

# 分配置汇总
print(f"\n{'配置':<14} {'n':<4} {'胜/负':<8} {'均值':<10} {'方向'}")
config_means = {}
for tag in dict.fromkeys(p[0] for p in pairs):  # 保序去重
    ps = [p for p in pairs if p[0] == tag]
    diffs = [p[4] for p in ps]
    wins = sum(1 for x in diffs if x > 0)
    config_means[tag] = np.mean(diffs)
    print(f"{tag:<14} {len(ps):<4} {wins}/{len(ps)-wins:<6} {np.mean(diffs):+.4f}    {'正✅' if np.mean(diffs) > 0 else '负❌'}")

all_consistent = all(v > 0 for v in config_means.values())
print(f"\n方向一致性: {'4/4配置均值全为正 ✅ 满足升级条件' if all_consistent else '存在负向配置 ❌'}")

# 主检验：全部26对
diffs_all = np.array([p[4] for p in pairs])
w_all, p_two_all = stats.wilcoxon(diffs_all, alternative="two-sided")
_, p_one_all = stats.wilcoxon(diffs_all, alternative="greater")
wins_all = int((diffs_all > 0).sum())
print(f"\n主检验（全部{len(diffs_all)}对，含1崩溃对）:")
print(f"  胜/负: {wins_all}/{len(diffs_all)-wins_all}, mean={diffs_all.mean():+.4f}, median={np.median(diffs_all):+.4f}")
print(f"  Wilcoxon 双侧 p={p_two_all:.4f} | 单侧 p={p_one_all:.4f}")

# 敏感性：25对非崩溃
mask = np.array([not p[5] for p in pairs])
diffs_nd = diffs_all[mask]
w_nd, p_two_nd = stats.wilcoxon(diffs_nd, alternative="two-sided")
_, p_one_nd = stats.wilcoxon(diffs_nd, alternative="greater")
wins_nd = int((diffs_nd > 0).sum())
print(f"\n敏感性（{len(diffs_nd)}对非崩溃）:")
print(f"  胜/负: {wins_nd}/{len(diffs_nd)-wins_nd}, mean={diffs_nd.mean():+.4f}, median={np.median(diffs_nd):+.4f}")
print(f"  Wilcoxon 双侧 p={p_two_nd:.4f} | 单侧 p={p_one_nd:.4f}")

# 符号检验（补充）
sign_p = stats.binomtest(wins_all, len(diffs_all), 0.5).pvalue
print(f"\n符号检验（{wins_all}/{len(diffs_all)}）: p={sign_p:.4f}")

# 判定
print("\n" + "=" * 70)
print("判定（按预注册规则）")
print("=" * 70)
sig_all = p_two_all < 0.05
sig_nd = p_two_nd < 0.05
if all_consistent and (sig_all or sig_nd):
    print("→ CW定位升级为：跨配置一致的小幅改进（Wilcoxon显著），Discussion可引用p值")
elif all_consistent:
    print("→ CW定位升级为：跨配置方向一致的小幅改进（n.s.）——仍限Discussion，标注不显著")
else:
    print("→ 维持'或有微益'，不作跨配置声明")
