"""探索性分析：aL=0.05下CW-exp vs FedAvg的逐seed配对精度（非预注册，仅记录趋势）"""

import json
import os
import numpy as np
from scipy import stats

SEEDS = [42, 123, 2024, 7, 99, 555, 777, 2025, 3407, 314]

def load_acc(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    return d["final_accuracy"]

diffs = []
print(f"{'seed':<8} {'FedAvg':<10} {'CW-exp':<10} {'diff':<10}")
for s in SEEDS:
    fa = load_acc(f"results/stress/cw_data_aL0.05_aF0.1_seed{s}.json")
    cw = load_acc(f"results/stress/cw_cw_exp_aL0.05_aF0.1_seed{s}_b5.0.json")
    if fa is not None and cw is not None:
        diffs.append(cw - fa)
        print(f"{s:<8} {fa:<10.4f} {cw:<10.4f} {cw-fa:+.4f}")

d = np.array(diffs)
t, p_t = stats.ttest_1samp(d, 0)
try:
    w, p_w = stats.wilcoxon(d)
except ValueError:
    p_w = float('nan')
wins = sum(1 for x in d if x > 0)
print(f"\nn={len(d)}, mean={d.mean():+.4f}, std={d.std(ddof=1):.4f}")
print(f"CW胜出: {wins}/{len(d)}")
print(f"配对t检验: t={t:.3f}, p={p_t:.4f}")
print(f"Wilcoxon: p={p_w:.4f}")
print(f"符号检验: p={stats.binomtest(wins, len(d), 0.5).pvalue:.4f}")
