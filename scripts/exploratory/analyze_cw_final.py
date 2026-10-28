"""Final CW-FedAvg analysis: compare all modes"""
import json, os
import numpy as np

d = "results/cw_fedavg"
configs = {}

for f in sorted(os.listdir(d)):
    if not f.endswith('.json'):
        continue
    data = json.load(open(os.path.join(d, f)))
    aL = data['alpha_label']
    aF = data['alpha_feature']
    key = f"aL{aL}_aF{aF}"
    mode = data['weight_mode']
    beta = data.get('beta', '')

    if key not in configs:
        configs[key] = {}
    mode_key = mode if mode != 'cw_exp' else f'cw_exp_b{beta}'
    configs[key][mode_key] = data['final_accuracy']

print("=" * 90)
print("CW-FedAvg vs Standard FedAvg: COMPLETE COMPARISON")
print("=" * 90)

print(f"\n{'Config':>20} {'FedAvg(data)':>14} {'CW(1/JS)':>14} {'CW(exp,β=5)':>14} {'Best':>14}")
print("-" * 80)

for key in sorted(configs.keys()):
    modes = configs[key]
    data_acc = modes.get('data', float('nan'))
    cw_acc = modes.get('cw', float('nan'))
    cw_exp_acc = modes.get('cw_exp_b5.0', float('nan'))

    best = 'data'
    best_val = data_acc
    if cw_acc > best_val:
        best = 'cw'
        best_val = cw_acc
    if cw_exp_acc > best_val:
        best = 'cw_exp'
        best_val = cw_exp_acc

    data_s = f"{data_acc:.4f}" if not np.isnan(data_acc) else "N/A"
    cw_s = f"{cw_acc:.4f}" if not np.isnan(cw_acc) else "N/A"
    cw_exp_s = f"{cw_exp_acc:.4f}" if not np.isnan(cw_exp_acc) else "N/A"

    print(f"{key:>20} {data_s:>14} {cw_s:>14} {cw_exp_s:>14} {best:>14}")

# Detailed delta analysis
print(f"\n{'=' * 90}")
print("DELTA ANALYSIS (CW-FedAvg - FedAvg)")
print("=" * 90)

for key in sorted(configs.keys()):
    modes = configs[key]
    data_acc = modes.get('data', float('nan'))
    cw_acc = modes.get('cw', float('nan'))
    cw_exp_acc = modes.get('cw_exp_b5.0', float('nan'))

    if not np.isnan(data_acc):
        print(f"\n--- {key} ---")
        print(f"  FedAvg:       {data_acc:.4f}")
        if not np.isnan(cw_acc):
            delta = cw_acc - data_acc
            print(f"  CW(1/JS):     {cw_acc:.4f}  (Δ={delta:+.4f}, {delta/data_acc*100:+.2f}%)")
        if not np.isnan(cw_exp_acc):
            delta = cw_exp_acc - data_acc
            print(f"  CW(exp,β=5): {cw_exp_acc:.4f}  (Δ={delta:+.4f}, {delta/data_acc*100:+.2f}%)")
