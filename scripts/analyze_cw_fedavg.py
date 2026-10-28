"""Compare CW-FedAvg with standard FedAvg from all available results"""
import json, os, glob
import numpy as np

# Load CW-FedAvg results
cw_dir = "results/cw_fedavg"
cw_results = {}
for f in glob.glob(os.path.join(cw_dir, "cw_cw_*.json")):
    data = json.load(open(f))
    key = f"aL{data['alpha_label']}_aF{data['alpha_feature']}"
    cw_results[key] = data

# Load ALL FedAvg results from all directories
fedavg_results = {}
for result_dir in ["results/metric_comparison", "results/k_selection",
                   "results/intervention", "results/cross_dataset"]:
    if not os.path.exists(result_dir):
        continue
    for f in glob.glob(os.path.join(result_dir, "*.json")):
        try:
            data = json.load(open(f))
            if isinstance(data, dict) and 'alpha_label' in data and 'final_accuracy' in data:
                key = f"aL{data['alpha_label']}_aF{data['alpha_feature']}"
                # Keep the best (highest accuracy) FedAvg result for each config
                if key not in fedavg_results or data['final_accuracy'] > fedavg_results[key]['final_accuracy']:
                    fedavg_results[key] = data
        except:
            pass

# Also run standard FedAvg for the exact same configs if not available
# First, let's see what we have
print("=" * 90)
print("AVAILABLE FedAvg BASELINES")
print("=" * 90)
for key in sorted(fedavg_results.keys()):
    print(f"  {key}: {fedavg_results[key]['final_accuracy']:.4f}")

print(f"\n{'=' * 90}")
print("CW-FedAvg vs Standard FedAvg COMPARISON")
print("=" * 90)

print(f"\n{'Config':>20} {'FedAvg':>10} {'CW-FedAvg':>10} {'Delta':>10} {'%Improve':>10}")
print("-" * 65)

improvements = []
for key in sorted(cw_results.keys()):
    cw_acc = cw_results[key]['final_accuracy']
    if key in fedavg_results:
        fa_acc = fedavg_results[key]['final_accuracy']
        delta = cw_acc - fa_acc
        pct = delta / fa_acc * 100 if fa_acc > 0 else 0
        print(f"{key:>20} {fa_acc:>10.4f} {cw_acc:>10.4f} {delta:>+10.4f} {pct:>+10.2f}%")
        improvements.append(delta)
    else:
        print(f"{key:>20} {'N/A':>10} {cw_acc:>10.4f} {'N/A':>10} {'N/A':>10}")

# If no matching baselines, we need to run standard FedAvg
if not improvements:
    print("\nNo matching baselines found! Need to run standard FedAvg for comparison.")
    print("Configs needed:")
    for key in sorted(cw_results.keys()):
        parts = key.split("_")
        aL = parts[0].replace("aL", "")
        aF = parts[1].replace("aF", "")
        print(f"  python scripts/run_cw_fedavg.py --alpha_label {aL} --alpha_feature {aF} --weight_mode data --num_rounds 200")

# Summary
if improvements:
    print(f"\n{'=' * 90}")
    print("SUMMARY")
    print("=" * 90)
    print(f"  Mean improvement: {np.mean(improvements):+.4f}")
    print(f"  Max improvement:  {np.max(improvements):+.4f}")
    print(f"  Min improvement:  {np.min(improvements):+.4f}")
    print(f"  Positive cases:   {sum(1 for d in improvements if d > 0)}/{len(improvements)}")

# CW-FedAvg details
print(f"\n{'=' * 90}")
print("CW-FedAvg DETAILED RESULTS")
print("=" * 90)
for key in sorted(cw_results.keys()):
    data = cw_results[key]
    print(f"\n--- {key} ---")
    print(f"  Final accuracy: {data['final_accuracy']:.4f}")
    js_vals = data['client_js_joints']
    wt_vals = data['aggregation_weights']
    r = np.corrcoef(js_vals, wt_vals)[0, 1]
    print(f"  Weight-JS correlation: r={r:.4f}")
    print(f"  JS_joint range: [{min(js_vals):.4f}, {max(js_vals):.4f}]")
    print(f"  Weight range:    [{min(wt_vals):.4f}, {max(wt_vals):.4f}]")
