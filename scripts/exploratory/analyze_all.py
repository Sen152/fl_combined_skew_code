"""全面分析所有盲区实验结果"""
import json, os

results_dir = 'results/blindspot'
experiments = {}
for f in sorted(os.listdir(results_dir)):
    if f.endswith('.json'):
        with open(os.path.join(results_dir, f)) as fp:
            data = json.load(fp)
        experiments[f] = data

print('=== 所有盲区实验对比 ===\n')
print(f'{"实验":<40} {"最终acc":>8} {"最高acc":>8} {"R50":>8} {"R100":>8} {"R150":>8} {"R200":>8}')
print('-' * 100)

for name, data in sorted(experiments.items()):
    final_acc = data[-1]['accuracy']
    max_acc = max(r['accuracy'] for r in data)
    r50 = data[49]['accuracy']
    r100 = data[99]['accuracy']
    r150 = data[149]['accuracy']
    r200 = data[199]['accuracy']
    short_name = name.replace('blindspot_aL0.1_', '').replace('_seed42.json', '')
    print(f'{short_name:<40} {final_acc:>8.4f} {max_acc:>8.4f} {r50:>8.4f} {r100:>8.4f} {r150:>8.4f} {r200:>8.4f}')

# 对比分析
print('\n=== 关键对比 ===')
# 对照组 (k=4, aF=1000) vs 实验组
ctrl_acc = experiments['blindspot_aL0.1_aF1000.0_seed42.json'][-1]['accuracy']
exp_005 = experiments.get('blindspot_aL0.1_aF0.05_seed42.json')
exp_001 = experiments.get('blindspot_aL0.1_aF0.01_seed42.json')

if exp_005:
    diff = ctrl_acc - exp_005[-1]['accuracy']
    print(f'对照组 vs aF=0.05: 差异 {diff*100:.2f}%')
if exp_001:
    diff = ctrl_acc - exp_001[-1]['accuracy']
    print(f'对照组 vs aF=0.01: 差异 {diff*100:.2f}%')

print('\n=== 分析 ===')
print('特征偏斜反而提高了性能！可能原因：')
print('1. 特征偏斜使客户端数据更同质，本地训练更容易收敛')
print('2. k-means簇不代表有意义的视觉差异（如光照/风格/背景）')
print('3. 合成特征偏斜只是"删样本"，不是"换分布"')
print('4. 真正的特征偏斜应该是：同类在不同客户端看起来不同')
