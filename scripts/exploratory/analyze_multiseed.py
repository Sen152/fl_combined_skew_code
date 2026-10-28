"""多种子统计分析：干预实验 + CW-FedAvg（3 seeds）

输出：
1. 干预实验：每个ratio的 mean±std，跨seed合并的相关性分析
2. CW-FedAvg：每个配置的 FedAvg vs CW-exp mean±std，逐seed配对差值
3. 稳健性判定：std<0.5% 为稳健
"""

import json
import os
import glob
import numpy as np
from scipy import stats

SEEDS = [42, 123, 2024]

def load_json(path):
    with open(path) as f:
        return json.load(f)

# ============ Part 1: 干预实验 ============
print("=" * 70)
print("Part 1: 干预实验（3 seeds × 7 ratio = 21组）")
print("=" * 70)

ratios = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75]
interv_data = {}  # ratio -> {seed: acc}

for r in ratios:
    interv_data[r] = {}
    for s in SEEDS:
        fname = f"results/intervention/intervention_aL0.3_inv{r}_seed{s}.json"
        if os.path.exists(fname):
            interv_data[r][s] = load_json(fname)["final_accuracy"]

print("\n各ratio准确率（mean±std, 3 seeds）：")
print(f"{'ratio':<8} {'seed42':<10} {'seed123':<10} {'seed2024':<10} {'mean':<10} {'std':<10}")
all_accs, all_jsj = [], []
for r in ratios:
    accs = [interv_data[r][s] for s in SEEDS if s in interv_data[r]]
    if len(accs) == 0:
        continue
    m, sd = np.mean(accs), np.std(accs, ddof=1) if len(accs) > 1 else 0
    row = f"{r:<8}"
    for s in SEEDS:
        row += f"{interv_data[r].get(s, float('nan')):<10.4f}"
    print(f"{row}{m:<10.4f}{sd:<10.4f}")
    all_accs.extend(accs)

# 跨seed合并相关性（每个seed的JS_joint不同，逐seed读）
print("\n逐seed相关性（JS_joint vs Acc，7点）：")
r2_list = []
for s in SEEDS:
    accs_s, jsj_s = [], []
    for r in ratios:
        fname = f"results/intervention/intervention_aL0.3_inv{r}_seed{s}.json"
        if os.path.exists(fname):
            d = load_json(fname)
            accs_s.append(d["final_accuracy"])
            jsj_s.append(d["metrics"]["JS_joint"])
    if len(accs_s) >= 5:
        rr, pp = stats.pearsonr(jsj_s, accs_s)
        r2_list.append(rr**2)
        print(f"  seed{s}: r={rr:.4f}, R2={rr**2:.4f}, p={pp:.5f} (n={len(accs_s)})")

print(f"\n  R2 across seeds: {np.mean(r2_list):.4f} ± {np.std(r2_list, ddof=1):.4f}")

# 合并所有seed（21点散点）
accs_all, jsj_all = [], []
for s in SEEDS:
    for r in ratios:
        fname = f"results/intervention/intervention_aL0.3_inv{r}_seed{s}.json"
        if os.path.exists(fname):
            d = load_json(fname)
            accs_all.append(d["final_accuracy"])
            jsj_all.append(d["metrics"]["JS_joint"])
rr, pp = stats.pearsonr(jsj_all, accs_all)
print(f"  合并21点: r={rr:.4f}, R2={rr**2:.4f}, p={pp:.2e}")

# 单调性检验（Spearman）
rho, sp = stats.spearmanr(jsj_all, accs_all)
print(f"  合并21点Spearman: rho={rho:.4f}, p={sp:.2e}")

# 稳健性
stds = []
for r in ratios:
    accs = [interv_data[r][s] for s in SEEDS if s in interv_data[r]]
    if len(accs) > 1:
        stds.append(np.std(accs, ddof=1))
max_std = max(stds)
print(f"\n  最大跨seed std: {max_std:.4f} ({'稳健✅' if max_std < 0.005 else '方差偏大⚠️'}，阈值0.005)")

# ============ Part 2: CW-FedAvg ============
print("\n" + "=" * 70)
print("Part 2: CW-FedAvg（3 seeds × 3配置 × 2方法）")
print("=" * 70)

configs = [(0.1, 0.1), (0.3, 0.1), (0.3, 0.3)]

def find_cw_file(aL, aF, seed, mode):
    """mode: 'data' or 'exp'"""
    if mode == "data":
        p = f"results/cw_fedavg/cw_data_aL{aL}_aF{aF}_seed{seed}.json"
    else:
        p = f"results/cw_fedavg/cw_cw_exp_aL{aL}_aF{aF}_seed{seed}_b5.0.json"
    return p if os.path.exists(p) else None

def is_diverged(d):
    """检测训练崩溃：最终acc<0.2 或 loss为NaN 或 出现NaN轮"""
    if d["final_accuracy"] < 0.2:
        return True
    if str(d.get("final_loss", "")) == "NaN":
        return True
    for r in d.get("round_log", []):
        if str(r.get("loss", "")) == "NaN":
            return True
    return False

print("\n各配置准确率对比（mean±std，崩溃检测）：")
print(f"{'配置':<14} {'FedAvg(3s)':<24} {'CW-exp(3s)':<24} {'差值':<10}")
cw_results = {}
for aL, aF in configs:
    fa_accs, cw_accs = [], []
    fa_crash, cw_crash = [], []
    for s in SEEDS:
        fa_f = find_cw_file(aL, aF, s, "data")
        cw_f = find_cw_file(aL, aF, s, "exp")
        if fa_f:
            d = load_json(fa_f)
            if is_diverged(d):
                fa_crash.append((s, d["final_accuracy"]))
            else:
                fa_accs.append(d["final_accuracy"])
        if cw_f:
            d = load_json(cw_f)
            if is_diverged(d):
                cw_crash.append((s, d["final_accuracy"]))
            else:
                cw_accs.append(d["final_accuracy"])
    cw_results[(aL, aF)] = {"fa": fa_accs, "cw": cw_accs,
                             "fa_crash": fa_crash, "cw_crash": cw_crash}
    fa_str = f"{np.mean(fa_accs):.4f}±{np.std(fa_accs, ddof=1):.4f}(n={len(fa_accs)})" if fa_accs else "无数据"
    cw_str = f"{np.mean(cw_accs):.4f}±{np.std(cw_accs, ddof=1):.4f}(n={len(cw_accs)})" if cw_accs else "无数据"
    diff = f"{np.mean(cw_accs)-np.mean(fa_accs):+.4f}" if fa_accs and cw_accs else "-"
    print(f"aL{aL}/aF{aF:<8} {fa_str:<24} {cw_str:<24} {diff:<10}")
    if fa_crash:
        print(f"         ⚠️ FedAvg崩溃seeds: {fa_crash}")
    if cw_crash:
        print(f"         ⚠️ CW-exp崩溃seeds: {cw_crash}")

# 稳定性核心分析：崩溃率对比
print("\n【稳定性分析】崩溃率（acc<20%或loss=NaN）：")
for (aL, aF), r in cw_results.items():
    n_fa = len(r["fa"]) + len(r["fa_crash"])
    n_cw = len(r["cw"]) + len(r["cw_crash"])
    if n_fa and n_cw:
        print(f"  aL{aL}/aF{aF}: FedAvg崩溃 {len(r['fa_crash'])}/{n_fa} vs CW-exp崩溃 {len(r['cw_crash'])}/{n_cw}")

# 仅用未崩溃seeds做配对分析
print("\n配对统计分析（仅未崩溃seeds）：")
all_diffs = []
for (aL, aF), r in cw_results.items():
    common = set()
    # 逐seed配对（用原始文件重新读，保证同seed配对）
    for s in SEEDS:
        fa_f = find_cw_file(aL, aF, s, "data")
        cw_f = find_cw_file(aL, aF, s, "exp")
        if fa_f and cw_f:
            if not is_diverged(load_json(fa_f)) and not is_diverged(load_json(cw_f)):
                common.add(s)
    diffs = []
    detail = []
    for s in sorted(common):
        fa = load_json(find_cw_file(aL, aF, s, "data"))["final_accuracy"]
        cw = load_json(find_cw_file(aL, aF, s, "exp"))["final_accuracy"]
        diffs.append(cw - fa)
        detail.append(f"s{s}:{cw-fa:+.4f}")
    if diffs:
        all_diffs.extend(diffs)
        print(f"  aL{aL}/aF{aF} (n={len(diffs)}): 均值{np.mean(diffs):+.4f}  {' '.join(detail)}")

if len(all_diffs) >= 3:
    ad = np.array(all_diffs)
    t, p = stats.ttest_1samp(ad, 0)
    print(f"\n  全部{len(ad)}个配对差值: mean={ad.mean():+.4f}, std={ad.std(ddof=1):.4f}")
    print(f"  单样本t检验(vs 0): t={t:.3f}, p={p:.4f}")
    print(f"  正差值比例: {sum(x > 0 for x in all_diffs)}/{len(all_diffs)}")

print("\n" + "=" * 70)
print("结论摘要")
print("=" * 70)
print(f"1. 干预实验: 跨seed最大std={max_std:.4f}，R2稳定在{np.mean(r2_list):.2f}±{np.std(r2_list, ddof=1):.2f}")
total_fa_crash = sum(len(r["fa_crash"]) for r in cw_results.values())
total_cw_crash = sum(len(r["cw_crash"]) for r in cw_results.values())
n_fa_total = sum(len(r["fa"]) + len(r["fa_crash"]) for r in cw_results.values())
n_cw_total = sum(len(r["cw"]) + len(r["cw_crash"]) for r in cw_results.values())
print(f"2. 崩溃率: FedAvg {total_fa_crash}/{n_fa_total} vs CW-exp {total_cw_crash}/{n_cw_total}")
if all_diffs:
    print(f"3. 精度差值（未崩溃seeds配对）: mean={np.mean(all_diffs):+.4f}，正差值{sum(x > 0 for x in all_diffs)}/{len(all_diffs)}")
print("4. CW-FedAvg新叙事: 从'提升精度'转向'防崩溃稳定性'（若崩溃证据成立）")
