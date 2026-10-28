"""度量对比基准综合分析

整合三类实验结果：
1. 纯标签偏斜（label_only）
2. 纯特征偏斜（feature_only）
3. 组合偏斜（combined，来自k_selection）

论证逻辑：
- 纯标签偏斜：JS_label能预测，JS_feature不能 → JS_joint ≈ JS_label
- 纯特征偏斜：JS_feature能预测，JS_label不能 → JS_joint ≈ JS_feature
- 组合偏斜：JS_joint > max(JS_label, JS_feature) → 联合度量严格优于单维度
"""

import json
import os
import numpy as np
from scipy.stats import pearsonr

# 加载k_selection结果（组合偏斜）
def load_k_selection_results(result_dir="results/k_selection"):
    results = []
    for fname in os.listdir(result_dir):
        if fname.startswith("aL") and fname.endswith(".json"):
            with open(os.path.join(result_dir, fname)) as f:
                data = json.load(f)
            # 转换为统一格式
            results.append({
                'skew_type': 'combined',
                'alpha_label': data['alpha_label'],
                'alpha_feature': data['alpha_feature'],
                'final_accuracy': data['final_accuracy'],
                'metrics': data['metrics_by_k']['k10'],  # 用k=10
            })
    return results

# 加载metric_comparison结果
def load_metric_comparison_results(result_dir="results/metric_comparison"):
    results = []
    for fname in os.listdir(result_dir):
        if fname.endswith(".json"):
            with open(os.path.join(result_dir, fname)) as f:
                results.append(json.load(f))
    return results

def analyze_subset(results, subset_name):
    """分析一个子集的度量-性能R²"""
    if not results:
        print(f"\n  No data for {subset_name}")
        return

    print(f"\n{'='*70}")
    print(f"  {subset_name} ({len(results)} experiments)")
    print(f"{'='*70}")

    accuracies = np.array([r['final_accuracy'] for r in results])

    # 打印基本结果
    print(f"\n  {'aL':>6} {'aF':>8} {'Accuracy':>10}")
    print(f"  {'-'*30}")
    for r in sorted(results, key=lambda x: (x.get('alpha_label',0), x.get('alpha_feature',0))):
        print(f"  {r.get('alpha_label',0):6.2f} {r.get('alpha_feature',0):8.2f} {r['final_accuracy']:10.4f}")

    # R²分析
    print(f"\n  {'Metric':20s} {'r':>8} {'r²':>8} {'p':>8} {'Direction':>10}")
    print(f"  {'-'*60}")

    metric_names = ['JS_joint', 'JS_label', 'JS_feature', 'JS_conditional',
                    'EMD_label', 'EMD_feature', 'Cov_basic', 'Cov_weighted']
    # 只分析存在的度量
    available_metrics = [m for m in metric_names if m in results[0]['metrics']]
    metric_names = available_metrics

    r2_results = {}
    for metric_name in metric_names:
        values = np.array([r['metrics'][metric_name] for r in results])
        if np.std(values) > 1e-10 and np.std(accuracies) > 1e-10:
            r_val, p_val = pearsonr(values, accuracies)
            direction = "↑" if r_val > 0 else "↓"
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            print(f"  {metric_name:20s} {r_val:+8.4f} {r_val**2:8.4f} {p_val:8.4f} {direction:>5} {sig}")
            r2_results[metric_name] = {'r': r_val, 'r2': r_val**2, 'p': p_val}
        else:
            print(f"  {metric_name:20s}   no variance")
            r2_results[metric_name] = {'r': 0, 'r2': 0, 'p': 1}

    return r2_results


def main():
    # 加载所有结果
    k_sel = load_k_selection_results()
    metric_comp = load_metric_comparison_results()

    # 按类型分组
    label_only = [r for r in metric_comp if r['skew_type'] == 'label_only']
    feature_only = [r for r in metric_comp if r['skew_type'] == 'feature_only']
    combined = k_sel  # k_selection就是组合偏斜

    print(f"Loaded: {len(label_only)} label_only, {len(feature_only)} feature_only, {len(combined)} combined")

    # 分别分析
    r2_label = analyze_subset(label_only, "纯标签偏斜 (aF=1000)")
    r2_feature = analyze_subset(feature_only, "纯特征偏斜 (aL=100)")
    r2_combined = analyze_subset(combined, "组合偏斜")

    # 汇总对比
    print(f"\n{'='*70}")
    print(f"  SUMMARY: JS_joint vs single-dimension metrics")
    print(f"{'='*70}")

    all_results = label_only + feature_only + combined
    if all_results:
        print(f"\n  All experiments ({len(all_results)} total):")
        accuracies = np.array([r['final_accuracy'] for r in all_results])
        all_metric_names = ['JS_joint', 'JS_label', 'JS_feature', 'JS_conditional',
                            'EMD_label', 'EMD_feature', 'Cov_weighted']
        for metric_name in all_metric_names:
            vals = [r['metrics'].get(metric_name) for r in all_results if metric_name in r['metrics']]
            if len(vals) < len(all_results):
                continue
            values = np.array(vals)
            if np.std(values) > 1e-10:
                r_val, p_val = pearsonr(values, accuracies)
                print(f"    {metric_name:20s}: r²={r_val**2:.4f}, p={p_val:.4f}")

    # 关键论证
    print(f"\n{'='*70}")
    print(f"  KEY ARGUMENT")
    print(f"{'='*70}")

    if r2_label:
        js_joint_r2 = r2_label.get('JS_joint', {}).get('r2', 0)
        js_label_r2 = r2_label.get('JS_label', {}).get('r2', 0)
        js_feature_r2 = r2_label.get('JS_feature', {}).get('r2', 0)
        print(f"\n  纯标签偏斜：")
        print(f"    JS_joint r²={js_joint_r2:.4f}, JS_label r²={js_label_r2:.4f}, JS_feature r²={js_feature_r2:.4f}")
        if js_label_r2 > js_feature_r2:
            print(f"    → JS_label > JS_feature ✓ (标签偏斜下标签度量更准)")
        if js_joint_r2 >= js_label_r2 * 0.9:
            print(f"    → JS_joint ≈ JS_label ✓ (联合度量不丢失标签信息)")

    if r2_feature:
        js_joint_r2 = r2_feature.get('JS_joint', {}).get('r2', 0)
        js_label_r2 = r2_feature.get('JS_label', {}).get('r2', 0)
        js_feature_r2 = r2_feature.get('JS_feature', {}).get('r2', 0)
        print(f"\n  纯特征偏斜：")
        print(f"    JS_joint r²={js_joint_r2:.4f}, JS_label r²={js_label_r2:.4f}, JS_feature r²={js_feature_r2:.4f}")
        if js_feature_r2 > js_label_r2:
            print(f"    → JS_feature > JS_label ✓ (特征偏斜下特征度量更准)")
        if js_joint_r2 >= js_feature_r2 * 0.9:
            print(f"    → JS_joint ≈ JS_feature ✓ (联合度量不丢失特征信息)")

    if r2_combined:
        js_joint_r2 = r2_combined.get('JS_joint', {}).get('r2', 0)
        js_label_r2 = r2_combined.get('JS_label', {}).get('r2', 0)
        js_feature_r2 = r2_combined.get('JS_feature', {}).get('r2', 0)
        print(f"\n  组合偏斜：")
        print(f"    JS_joint r²={js_joint_r2:.4f}, JS_label r²={js_label_r2:.4f}, JS_feature r²={js_feature_r2:.4f}")
        if js_joint_r2 > max(js_label_r2, js_feature_r2):
            print(f"    → JS_joint > max(JS_label, JS_feature) ✓ (联合度量严格优于单维度！)")
        else:
            print(f"    → JS_joint NOT > max(JS_label, JS_feature) ✗ (需要更多数据点)")


if __name__ == "__main__":
    main()
