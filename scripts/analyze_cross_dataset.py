"""FMNIST跨数据集验证：R²分析"""

import json, os
import numpy as np
from scipy.stats import pearsonr

d = "results/cross_dataset"
results = []
for f in sorted(os.listdir(d)):
    if f.endswith(".json"):
        data = json.load(open(os.path.join(d, f)))
        results.append(data)

print("=" * 80)
print("FMNIST CROSS-DATASET VALIDATION")
print("=" * 80)

# 结果表
print(f"\n{'aL':>8} {'aF':>8} {'Accuracy':>10} {'JS_joint':>10} {'JS_label':>10} {'JS_feature':>10}")
print("-" * 60)
for r in results:
    m = r.get('metrics', {})
    print(f"{r['alpha_label']:8.2f} {r['alpha_feature']:8.2f} {r['final_accuracy']:10.4f} "
          f"{m.get('JS_joint',0):10.4f} {m.get('JS_label',0):10.4f} {m.get('JS_feature',0):10.4f}")

# 分类
label_only = [r for r in results if r['alpha_feature'] >= 100]
feature_only = [r for r in results if r['alpha_label'] >= 100]
combined = [r for r in results if r['alpha_feature'] < 100 and r['alpha_label'] < 100]

print(f"\n纯标签偏斜: {len(label_only)}组")
print(f"纯特征偏斜: {len(feature_only)}组")
print(f"组合偏斜: {len(combined)}组")

# R²分析
print(f"\n{'=' * 80}")
print("R² ANALYSIS BY SCENARIO")
print("=" * 80)

for name, subset in [("纯标签偏斜", label_only), ("纯特征偏斜", feature_only), ("组合偏斜", combined)]:
    if len(subset) < 3:
        print(f"\n{name}: too few ({len(subset)})")
        continue

    accs = np.array([r['final_accuracy'] for r in subset])
    print(f"\n--- {name} ({len(subset)} experiments) ---")

    for metric in ['JS_joint', 'JS_label', 'JS_feature', 'EMD_label', 'EMD_feature', 'Cov_weighted']:
        vals = np.array([r['metrics'].get(metric, float('nan')) for r in subset])
        valid = ~np.isnan(vals) & (vals > 0)
        if valid.sum() >= 3 and np.std(vals[valid]) > 1e-10:
            r_val, p_val = pearsonr(vals[valid], accs[valid])
            print(f"  {metric:20s}: R²={r_val**2:.4f}, r={r_val:+.4f}, p={p_val:.4f}")
        else:
            print(f"  {metric:20s}: insufficient variance")

# 与CIFAR-10对比
print(f"\n{'=' * 80}")
print("COMPARISON: FMNIST vs CIFAR-10")
print("=" * 80)
print("""
CIFAR-10 (Phase 1):
  组合偏斜: JS_joint=0.752, JS_label=0.542, JS_feature=0.712
  纯标签偏斜: JS_joint=0.792, JS_label=0.826, JS_feature=0.823
  纯特征偏斜: JS_joint=0.978, JS_label=0.733, JS_feature=0.972

FMNIST (本实验):
  (见上方结果)
""")

# 全量R²
print(f"\n--- 全部8组 ---")
accs = np.array([r['final_accuracy'] for r in results])
for metric in ['JS_joint', 'JS_label', 'JS_feature']:
    vals = np.array([r['metrics'].get(metric, float('nan')) for r in results])
    valid = ~np.isnan(vals) & (vals > 0)
    if valid.sum() >= 3:
        r_val, p_val = pearsonr(vals[valid], accs[valid])
        print(f"  {metric:20s}: R²={r_val**2:.4f}, r={r_val:+.4f}, p={p_val:.4f}")
