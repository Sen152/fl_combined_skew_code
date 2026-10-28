"""分析k值选择实验结果（初步版，数据点少）"""

import json
import os
import numpy as np
from scipy.stats import pearsonr

result_dir = "results/k_selection"
results = []

for fname in os.listdir(result_dir):
    if fname.startswith("aL") and fname.endswith(".json"):
        with open(os.path.join(result_dir, fname)) as f:
            results.append(json.load(f))

print(f"Loaded {len(results)} experiments\n")

# 基本结果表
print("=" * 80)
print("EXPERIMENT RESULTS")
print("=" * 80)
print(f"{'aL':>6} {'aF':>6} {'Accuracy':>10} {'Loss':>10}")
print("-" * 40)
for r in sorted(results, key=lambda x: (x['alpha_label'], x['alpha_feature'])):
    print(f"{r['alpha_label']:6.2f} {r['alpha_feature']:6.2f} {r['final_accuracy']:10.4f} {r['final_loss']:10.4f}")

# R²分析
print(f"\n{'=' * 80}")
print("R² ANALYSIS: metric vs accuracy")
print("=" * 80)

accuracies = np.array([r['final_accuracy'] for r in results])

for k in [4, 10, 20, 50]:
    key = f"k{k}"
    print(f"\n--- k={k} ---")
    for metric_name in ['JS_joint', 'JS_label', 'JS_feature', 'Cov_basic', 'Cov_weighted']:
        values = np.array([r['metrics_by_k'][key][metric_name] for r in results])
        if np.std(values) > 1e-10:
            r_val, p_val = pearsonr(values, accuracies)
            direction = "↑" if r_val > 0 else "↓"
            print(f"  {metric_name:15s}: r={r_val:+.4f}, r²={r_val**2:.4f}, p={p_val:.4f} {direction}")
        else:
            print(f"  {metric_name:15s}: no variance")

# 跨k对比：JS_joint的R²
print(f"\n{'=' * 80}")
print("KEY COMPARISON: JS_joint R² across k values")
print("=" * 80)
for k in [4, 10, 20, 50]:
    key = f"k{k}"
    values = np.array([r['metrics_by_k'][key]['JS_joint'] for r in results])
    if np.std(values) > 1e-10:
        r_val, p_val = pearsonr(values, accuracies)
        print(f"  k={k:3d}: r²={r_val**2:.4f}, r={r_val:+.4f}, p={p_val:.4f}")

# 跨k对比：JS_feature的R²
print(f"\nKEY COMPARISON: JS_feature R² across k values")
for k in [4, 10, 20, 50]:
    key = f"k{k}"
    values = np.array([r['metrics_by_k'][key]['JS_feature'] for r in results])
    if np.std(values) > 1e-10:
        r_val, p_val = pearsonr(values, accuracies)
        print(f"  k={k:3d}: r²={r_val**2:.4f}, r={r_val:+.4f}, p={p_val:.4f}")

# 跨k对比：Cov_weighted的R²
print(f"\nKEY COMPARISON: Cov_weighted R² across k values")
for k in [4, 10, 20, 50]:
    key = f"k{k}"
    values = np.array([r['metrics_by_k'][key]['Cov_weighted'] for r in results])
    if np.std(values) > 1e-10:
        r_val, p_val = pearsonr(values, accuracies)
        print(f"  k={k:3d}: r²={r_val**2:.4f}, r={r_val:+.4f}, p={p_val:.4f}")

# 稀疏度分析
print(f"\n{'=' * 80}")
print("SPARSITY ANALYSIS")
print("=" * 80)
for k in [4, 10, 20, 50]:
    key = f"k{k}"
    cov_values = np.array([r['metrics_by_k'][key]['Cov_basic'] for r in results])
    print(f"  k={k:3d}: Cov_basic mean={cov_values.mean():.4f}, "
          f"range=[{cov_values.min():.4f}, {cov_values.max():.4f}]")
