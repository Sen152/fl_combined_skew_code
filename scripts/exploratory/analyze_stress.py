"""四臂压力测试分析：aL=0.05/aF=0.1 × 10 seeds × {FedAvg, FedAvg+clip, Median, CW-exp}

预注册判定规则（见FL_Research_Tree.md R.2.1.3.4，跑前写入不得修改）：
- 条件A：FedAvg崩>=4/10 且 CW崩0/10 → 与aL=0.1合并(FedAvg 1/10 vs CW 0/10)，
  Fisher单侧p<0.05 → 稳定性故事进正文
- 条件B：FedAvg崩1~3或0 → CW降级为讨论部分
崩溃定义：final_accuracy<0.2 或 round_log出现NaN loss
"""

import json
import os
import glob
import numpy as np
from scipy import stats

SEEDS = [42, 123, 2024, 7, 99, 555, 777, 2025, 3407, 314]
STRESS_DIR = "results/stress"
CW_DIR = "results/cw_fedavg"  # aL=0.1的旧数据在这里

def analyze_file(path):
    """返回 (crashed, final_acc, first_nan_round, max_acc) 或 None"""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception:
        return None
    acc = d.get("final_accuracy", 0)
    log = d.get("round_log", [])
    first_nan = None
    for r in log:
        lv = r.get("loss")
        if str(lv) == "NaN" or (isinstance(lv, float) and lv != lv):
            first_nan = r["round"]
            break
    max_acc = max((r["accuracy"] for r in log), default=acc)
    crashed = (acc < 0.2) or (first_nan is not None)
    return crashed, acc, first_nan, max_acc

# ============ 收集四臂数据 ============
print("=" * 78)
print("四臂压力测试：aL=0.05/aF=0.1 × 10 seeds")
print("=" * 78)

arms = {
    "FedAvg":      "cw_data_aL0.05_aF0.1_seed{s}.json",
    "FedAvg+clip": "cw_data_aL0.05_aF0.1_seed{s}_clip10.0.json",
    "Median":      "cw_median_aL0.05_aF0.1_seed{s}.json",
    "CW-exp":      "cw_cw_exp_aL0.05_aF0.1_seed{s}_b5.0.json",
}

results = {arm: {} for arm in arms}
for arm, pattern in arms.items():
    for s in SEEDS:
        results[arm][s] = analyze_file(os.path.join(STRESS_DIR, pattern.format(s=s)))

# ============ 逐seed明细表 ============
print(f"\n{'seed':<8}" + "".join(f"{a:<22}" for a in arms))
for s in SEEDS:
    row = f"{s:<8}"
    for arm in arms:
        r = results[arm][s]
        if r is None:
            row += f"{'缺失':<22}"
        elif r[0]:
            row += f"{'💥崩溃@R' + str(r[2]):<22}"
        else:
            row += f"{f'{r[1]:.4f}':<22}"
    print(row)

# ============ 崩溃率汇总 ============
print("\n" + "=" * 78)
print("崩溃率汇总（崩溃=acc<20%或NaN）")
print("=" * 78)
crash_counts = {}
valid_counts = {}
for arm in arms:
    runs = [results[arm][s] for s in SEEDS if results[arm][s] is not None]
    n = len(runs)
    c = sum(1 for r in runs if r[0])
    crash_counts[arm] = c
    valid_counts[arm] = n
    seeds_crashed = [s for s in SEEDS if results[arm][s] and results[arm][s][0]]
    print(f"{arm:<14}: {c}/{n} 崩溃  {f'(seeds: {seeds_crashed})' if seeds_crashed else ''}")

# ============ 正常样本精度 ============
print("\n正常（未崩溃）样本最终精度：")
for arm in arms:
    accs = [results[arm][s][1] for s in SEEDS
            if results[arm][s] is not None and not results[arm][s][0]]
    if accs:
        print(f"  {arm:<14}: {np.mean(accs):.4f} ± {np.std(accs, ddof=1):.4f} (n={len(accs)})")

# ============ 主检验：FedAvg vs CW-exp ============
print("\n" + "=" * 78)
print("主检验（预注册）：FedAvg vs CW-exp")
print("=" * 78)
fa_c, fa_n = crash_counts["FedAvg"], valid_counts["FedAvg"]
cw_c, cw_n = crash_counts["CW-exp"], valid_counts["CW-exp"]
odds, p_two = stats.fisher_exact([[fa_c, fa_n - fa_c], [cw_c, cw_n - cw_c]])
print(f"aL=0.05单独: FedAvg {fa_c}/{fa_n} vs CW {cw_c}/{cw_n}, Fisher双侧 p={p_two:.4f}")

# 合并aL=0.1旧数据（旧：FedAvg崩溃1/10, CW 0/10）
fa_c_pool = fa_c + 1
fa_n_pool = fa_n + 10
cw_c_pool = cw_c + 0
cw_n_pool = cw_n + 10
odds_p, p_pool = stats.fisher_exact([[fa_c_pool, fa_n_pool - fa_c_pool],
                                      [cw_c_pool, cw_n_pool - cw_c_pool]])
print(f"合并aL=0.1: FedAvg {fa_c_pool}/{fa_n_pool} vs CW {cw_c_pool}/{cw_n_pool}, "
      f"Fisher双侧 p={p_pool:.4f}")

# ============ 判定规则 ============
print("\n" + "=" * 78)
print("预注册判定规则执行")
print("=" * 78)
if fa_c >= 4 and cw_c == 0:
    verdict = "条件A成立：稳定性故事进正文（Section 6）"
elif fa_c == 0 and cw_c == 0:
    verdict = "条件B成立：aL=0.05下双方都稳 → CW降级为讨论部分"
else:
    verdict = "中间情况：按条件B处理（CW降级为讨论），但记录剂量趋势"
print(f"FedAvg崩溃={fa_c}/10, CW崩溃={cw_c}/10")
print(f"判定: {verdict}")

# ============ 机理分析（次要臂） ============
print("\n" + "=" * 78)
print("机理分析：clipping/Median臂的病因鉴别")
print("=" * 78)
clip_c = crash_counts["FedAvg+clip"]
med_c = crash_counts["Median"]
if fa_c > 0:
    if clip_c == 0 and med_c == 0:
        print(f"FedAvg崩{fa_c}但clip和Median都防住 → 病因是客户端梯度爆炸+聚合方式共同作用")
    elif clip_c == 0:
        print(f"clip防住({clip_c}/10) → 病因=梯度爆炸（客户端侧），CW等效于软性防护")
    elif med_c == 0:
        print(f"Median防住({med_c}/10) → 病因=加权平均被极端参数污染（服务器侧聚合问题）")
    else:
        print(f"clip崩{clip_c}/10, Median崩{med_c}/10 → 两种机制都不能完全防住")
else:
    print("FedAvg在aL=0.05下未崩溃，机理分析无事件可分析")

# 次要臂两两Fisher（探索性，标注不作为主判定）
for arm_b in ["FedAvg+clip", "Median"]:
    cb, nb = crash_counts[arm_b], valid_counts[arm_b]
    _, pb = stats.fisher_exact([[fa_c, fa_n - fa_c], [cb, nb - cb]])
    print(f"探索性: FedAvg({fa_c}/{fa_n}) vs {arm_b}({cb}/{nb}): p={pb:.4f}")

# ============ 含崩溃样本的期望性能 ============
print("\n含崩溃样本期望性能（部署视角，崩溃=10%）：")
for arm in arms:
    all_accs = [results[arm][s][1] for s in SEEDS if results[arm][s] is not None]
    if all_accs:
        print(f"  {arm:<14}: {np.mean(all_accs):.4f}")
