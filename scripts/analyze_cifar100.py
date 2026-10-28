"""CIFAR-100 R2 analysis script"""
import json
from scipy import stats

results = []
for aL, aF in [(0.05,1000.0), (0.3,1000.0), (100.0,0.05), (100.0,0.3),
                (0.05,0.05), (0.05,0.3), (0.3,0.05), (0.3,0.3)]:
    fname = f'results/cifar100/cifar100_aL{aL}_aF{aF}_seed42.json'
    with open(fname) as f:
        r = json.load(f)
    results.append(r)

accs = [r['final_accuracy'] for r in results]
js_joint = [r['metrics']['JS_joint'] for r in results]
js_label = [r['metrics']['JS_label'] for r in results]
js_feature = [r['metrics']['JS_feature'] for r in results]
js_cond = [r['metrics']['JS_conditional'] for r in results]

print('=== CIFAR-100 Full 8-experiment Results ===')
for r in results:
    aL, aF = r['alpha_label'], r['alpha_feature']
    if aF >= 100:
        tp = 'PureLabel'
    elif aL >= 100:
        tp = 'PureFeat'
    else:
        tp = 'Combined'
    acc_pct = r['final_accuracy'] * 100
    jj = r['metrics']['JS_joint']
    jl = r['metrics']['JS_label']
    jf = r['metrics']['JS_feature']
    print(f'{tp:<12} aL={aL:<8} aF={aF:<8} Acc={acc_pct:.2f}% JS_joint={jj:.4f} JS_label={jl:.4f} JS_feature={jf:.4f}')

print()
print('=== R2 Analysis (ALL 8 experiments) ===')
for name, vals in [('JS_joint', js_joint), ('JS_label', js_label), ('JS_feature', js_feature), ('JS_conditional', js_cond)]:
    r, p = stats.pearsonr(vals, accs)
    R2 = r**2
    print(f'  {name}: r={r:.4f}, R2={R2:.4f}, p={p:.6f}')

print()
print('=== R2 by Scenario ===')

# Combined (4 points)
idx_cb = [4, 5, 6, 7]
r_j, p_j = stats.pearsonr([js_joint[i] for i in idx_cb], [accs[i] for i in idx_cb])
r_l, p_l = stats.pearsonr([js_label[i] for i in idx_cb], [accs[i] for i in idx_cb])
r_f, p_f = stats.pearsonr([js_feature[i] for i in idx_cb], [accs[i] for i in idx_cb])
r_c, p_c = stats.pearsonr([js_cond[i] for i in idx_cb], [accs[i] for i in idx_cb])
print(f'Combined (n=4): JS_joint R2={r_j**2:.4f}(p={p_j:.4f}), JS_label R2={r_l**2:.4f}(p={p_l:.4f}), JS_feature R2={r_f**2:.4f}(p={p_f:.4f}), JS_cond R2={r_c**2:.4f}(p={p_c:.4f})')

# Label+Combined (6 points)
idx_lc = [0, 1, 4, 5, 6, 7]
r_j2, p_j2 = stats.pearsonr([js_joint[i] for i in idx_lc], [accs[i] for i in idx_lc])
r_l2, p_l2 = stats.pearsonr([js_label[i] for i in idx_lc], [accs[i] for i in idx_lc])
print(f'Label+Combined (n=6): JS_joint R2={r_j2**2:.4f}(p={p_j2:.6f}), JS_label R2={r_l2**2:.4f}(p={p_l2:.6f})')

# Feature+Combined (6 points)
idx_fc = [2, 3, 4, 5, 6, 7]
r_j3, p_j3 = stats.pearsonr([js_joint[i] for i in idx_fc], [accs[i] for i in idx_fc])
r_f3, p_f3 = stats.pearsonr([js_feature[i] for i in idx_fc], [accs[i] for i in idx_fc])
print(f'Feature+Combined (n=6): JS_joint R2={r_j3**2:.4f}(p={p_j3:.6f}), JS_feature R2={r_f3**2:.4f}(p={p_f3:.6f})')

# Cross-dataset comparison
print()
print('=== Cross-Dataset R2 Summary ===')
print(f'CIFAR-100 (8 exps): JS_joint R2={stats.pearsonr(js_joint, accs)[0]**2:.4f}, JS_label R2={stats.pearsonr(js_label, accs)[0]**2:.4f}, JS_feature R2={stats.pearsonr(js_feature, accs)[0]**2:.4f}')

# Load CIFAR-10 results for comparison
c10_results = []
import os
c10_dir = 'results/metric_comparison'
if os.path.exists(c10_dir):
    for fname in os.listdir(c10_dir):
        if fname.endswith('.json'):
            with open(os.path.join(c10_dir, fname)) as f:
                c10_results.append(json.load(f))
    if c10_results:
        c10_accs = [r['final_accuracy'] for r in c10_results]
        c10_jj = [r['metrics']['JS_joint'] for r in c10_results]
        c10_jl = [r['metrics']['JS_label'] for r in c10_results]
        c10_jf = [r['metrics']['JS_feature'] for r in c10_results]
        print(f'CIFAR-10 ({len(c10_results)} exps): JS_joint R2={stats.pearsonr(c10_jj, c10_accs)[0]**2:.4f}, JS_label R2={stats.pearsonr(c10_jl, c10_accs)[0]**2:.4f}, JS_feature R2={stats.pearsonr(c10_jf, c10_accs)[0]**2:.4f}')

# Load FMNIST results
fm_results = []
fm_dir = 'results/cross_dataset'
if os.path.exists(fm_dir):
    for fname in os.listdir(fm_dir):
        if fname.endswith('.json') and 'fmnist' in fname:
            with open(os.path.join(fm_dir, fname)) as f:
                fm_results.append(json.load(f))
    if fm_results:
        fm_accs = [r['final_accuracy'] for r in fm_results]
        fm_jj = [r['metrics']['JS_joint'] for r in fm_results]
        fm_jl = [r['metrics']['JS_label'] for r in fm_results]
        fm_jf = [r['metrics']['JS_feature'] for r in fm_results]
        print(f'FMNIST ({len(fm_results)} exps): JS_joint R2={stats.pearsonr(fm_jj, fm_accs)[0]**2:.4f}, JS_label R2={stats.pearsonr(fm_jl, fm_accs)[0]**2:.4f}, JS_feature R2={stats.pearsonr(fm_jf, fm_accs)[0]**2:.4f}')
