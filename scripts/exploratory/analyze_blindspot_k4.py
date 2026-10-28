"""分析盲区实验结果（k=4版本）"""
import json

files = {
    '对照组 k=10 aF=1000': 'results/blindspot/blindspot_aL0.1_aF1000.0_seed42.json',
    '实验组 k=10 aF=0.3': 'results/blindspot/blindspot_aL0.1_aF0.3_seed42.json',
}

# k=4的结果文件名可能不同，先检查
import os
for f in os.listdir('results/blindspot'):
    print(f)

# 读取k=4的结果
k4_ctrl = 'results/blindspot/blindspot_aL0.1_aF1000.0_seed42.json'
k4_exp = 'results/blindspot/blindspot_aL0.1_aF0.05_seed42.json'

with open(k4_ctrl) as f:
    ctrl = json.load(f)
with open(k4_exp) as f:
    exp = json.load(f)

print('\n=== 盲区实验结果 (k=4 per class, 40 total clusters) ===')
print('对照组 (alpha_feature=1000, 特征均匀):')
print(f'  最终 acc: {ctrl[-1]["accuracy"]:.4f}, loss: {ctrl[-1]["loss"]:.4f}')
print(f'  最高 acc: {max(r["accuracy"] for r in ctrl):.4f}')
for r in [49, 99, 149, 199]:
    print(f'  Round {r+1}: acc={ctrl[r]["accuracy"]:.4f}, loss={ctrl[r]["loss"]:.4f}')

print('\n实验组 (alpha_feature=0.05, 特征强偏斜):')
print(f'  最终 acc: {exp[-1]["accuracy"]:.4f}, loss: {exp[-1]["loss"]:.4f}')
print(f'  最高 acc: {max(r["accuracy"] for r in exp):.4f}')
for r in [49, 99, 149, 199]:
    print(f'  Round {r+1}: acc={exp[r]["accuracy"]:.4f}, loss={exp[r]["loss"]:.4f}')

diff = ctrl[-1]['accuracy'] - exp[-1]['accuracy']
print(f'\n性能差异: {diff*100:.2f}% (对照组 - 实验组)')
if abs(diff) >= 0.05:
    print('决策门: >=5% 方向正确，全力推进!')
elif abs(diff) >= 0.02:
    print('决策门: 2-5% 需更仔细分析，可能有效')
else:
    print('决策门: <2% 方向存疑，需重新评估')

# 对比k=10的结果
print('\n=== 对比 k=10 vs k=4 ===')
print('k=10, aF=0.3: 差异 0.65%')
print(f'k=4,  aF=0.05: 差异 {diff*100:.2f}%')
