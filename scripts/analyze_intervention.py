"""干预实验分析：剂量-反应关系验证因果性"""

import json
import os
import numpy as np
from scipy.stats import pearsonr, spearmanr

result_dir = "results/intervention"
results = []

for fname in sorted(os.listdir(result_dir)):
    if fname.endswith(".json"):
        with open(os.path.join(result_dir, fname)) as f:
            results.append(json.load(f))

results.sort(key=lambda x: x['invert_ratio'])

print(f"Loaded {len(results)} experiments\n")

# 基本结果表
print("=" * 80)
print("INTERVENTION EXPERIMENT RESULTS")
print("=" * 80)
print(f"{'inv_ratio':>10} {'n_invert':>10} {'Accuracy':>10} {'JS_joint':>10} {'JS_label':>10} {'JS_feature':>10} {'JS_cond':>10}")
print("-" * 70)
for r in results:
    m = r['metrics']
    print(f"{r['invert_ratio']:10.2f} {r['num_invert_clients']:10d} {r['final_accuracy']:10.4f} "
          f"{m['JS_joint']:10.4f} {m['JS_label']:10.4f} {m['JS_feature']:10.4f} "
          f"{m.get('JS_conditional', float('nan')):10.4f}")

# 剂量-反应分析
print(f"\n{'=' * 80}")
print("DOSE-RESPONSE ANALYSIS")
print("=" * 80)

ratios = np.array([r['invert_ratio'] for r in results])
accuracies = np.array([r['final_accuracy'] for r in results])
js_joints = np.array([r['metrics']['JS_joint'] for r in results])
js_features = np.array([r['metrics']['JS_feature'] for r in results])
js_labels = np.array([r['metrics']['JS_label'] for r in results])

# 1. invert_ratio vs accuracy（期望U型：0.5最低）
print(f"\n1. invert_ratio vs accuracy (expect U-shape)")
for r in results:
    print(f"   ratio={r['invert_ratio']:.2f}: acc={r['final_accuracy']:.4f}")

# 检查是否U型
min_idx = np.argmin(accuracies)
if ratios[min_idx] == 0.5:
    print(f"   ✓ U-shape confirmed! Minimum at ratio=0.5 (acc={accuracies[min_idx]:.4f})")
else:
    print(f"   ✗ Not U-shape. Minimum at ratio={ratios[min_idx]:.2f} (acc={accuracies[min_idx]:.4f})")

# 2. invert_ratio vs JS_feature（期望倒U型：0.5最高）
print(f"\n2. invert_ratio vs JS_feature (expect inverted U-shape)")
for r in results:
    print(f"   ratio={r['invert_ratio']:.2f}: JS_feature={r['metrics']['JS_feature']:.4f}")

max_idx = np.argmax(js_features)
if ratios[max_idx] == 0.5:
    print(f"   ✓ Inverted U-shape confirmed! Maximum at ratio=0.5 (JS_feature={js_features[max_idx]:.4f})")
else:
    print(f"   ✗ Not inverted U-shape. Maximum at ratio={ratios[max_idx]:.2f} (JS_feature={js_features[max_idx]:.4f})")

# 3. JS_joint vs accuracy（期望单调递减——排除ratio=1.0后）
print(f"\n3. JS_joint vs accuracy (expect monotonic decrease)")
# 排除ratio=1.0（因为全部invert=无特征偏斜，但JS_joint计算方式不同）
mask_not1 = ratios < 1.0
if mask_not1.sum() >= 3:
    r_val, p_val = pearsonr(js_joints[mask_not1], accuracies[mask_not1])
    print(f"   Excluding ratio=1.0: r={r_val:.4f}, r²={r_val**2:.4f}, p={p_val:.4f}")
    if r_val < 0 and p_val < 0.05:
        print(f"   ✓ Significant negative correlation! Causality supported.")
    else:
        print(f"   ✗ No significant negative correlation.")

# 4. JS_feature vs accuracy（排除ratio=1.0）
print(f"\n4. JS_feature vs accuracy (excluding ratio=1.0)")
if mask_not1.sum() >= 3:
    r_val, p_val = pearsonr(js_features[mask_not1], accuracies[mask_not1])
    print(f"   r={r_val:.4f}, r²={r_val**2:.4f}, p={p_val:.4f}")

# 5. 全部5个点的相关性
print(f"\n5. All 5 points correlation")
for metric_name in ['JS_joint', 'JS_label', 'JS_feature', 'JS_conditional',
                     'EMD_label', 'EMD_feature', 'Cov_weighted']:
    vals = np.array([r['metrics'].get(metric_name, float('nan')) for r in results])
    valid = ~np.isnan(vals)
    if valid.sum() >= 3 and np.std(vals[valid]) > 1e-10:
        r_val, p_val = pearsonr(vals[valid], accuracies[valid])
        print(f"   {metric_name:20s}: r={r_val:+.4f}, r²={r_val**2:.4f}, p={p_val:.4f}")

# 6. 关键因果论证
print(f"\n{'=' * 80}")
print("CAUSALITY ARGUMENT")
print("=" * 80)

# 固定标签偏斜（alpha_label=0.3），JS_label应该几乎不变
print(f"\nJS_label across ratios (should be ~constant since alpha_label fixed):")
for r in results:
    print(f"   ratio={r['invert_ratio']:.2f}: JS_label={r['metrics']['JS_label']:.4f}")
js_label_std = np.std(js_labels)
js_label_range = np.max(js_labels) - np.min(js_labels)
print(f"   JS_label range: {js_label_range:.4f}, std: {js_label_std:.4f}")

if js_label_range < 0.1:
    print(f"   ✓ JS_label nearly constant — label skew is controlled!")
else:
    print(f"   ✗ JS_label varies too much — label skew not properly controlled")

# accuracy变化幅度
acc_range = np.max(accuracies) - np.min(accuracies)
print(f"\nAccuracy range: {acc_range:.4f} ({np.min(accuracies):.4f} ~ {np.max(accuracies):.4f})")

if acc_range > 0.05:
    print(f"   ✓ Significant accuracy variation (>5%) — feature skew has causal effect!")
else:
    print(f"   ✗ Accuracy variation too small (<5%) — feature skew effect unclear")

# 7. 训练曲线
print(f"\n{'=' * 80}")
print("TRAINING CURVES (selected rounds)")
print("=" * 80)
for r in results:
    log = r['round_log']
    key_rounds = [1, 10, 50, 100, 150, 200]
    accs = [log[i-1]['accuracy'] for i in key_rounds if i <= len(log)]
    print(f"   ratio={r['invert_ratio']:.2f}: " +
          " → ".join(f"{a:.3f}" for a in accs))
