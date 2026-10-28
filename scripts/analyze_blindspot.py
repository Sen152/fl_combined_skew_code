"""分析盲区实验结果"""
import json

with open('results/blindspot/blindspot_aL0.1_aF1000.0_seed42.json') as f:
    ctrl = json.load(f)
with open('results/blindspot/blindspot_aL0.1_aF0.3_seed42.json') as f:
    exp = json.load(f)

print('=== 盲区实验结果 ===')
print('对照组 (alpha_feature=1000, 特征均匀):')
print(f'  最终 acc: {ctrl[-1]["accuracy"]:.4f}, loss: {ctrl[-1]["loss"]:.4f}')
print(f'  最高 acc: {max(r["accuracy"] for r in ctrl):.4f}')
for r in [49, 99, 149, 199]:
    print(f'  Round {r+1}: {ctrl[r]["accuracy"]:.4f}')

print('\n实验组 (alpha_feature=0.3, 特征偏斜):')
print(f'  最终 acc: {exp[-1]["accuracy"]:.4f}, loss: {exp[-1]["loss"]:.4f}')
print(f'  最高 acc: {max(r["accuracy"] for r in exp):.4f}')
for r in [49, 99, 149, 199]:
    print(f'  Round {r+1}: {exp[r]["accuracy"]:.4f}')

diff = ctrl[-1]['accuracy'] - exp[-1]['accuracy']
print(f'\n性能差异: {diff*100:.2f}% (对照组 - 实验组)')
if abs(diff) >= 0.05:
    print('决策门: >=5% 方向正确，全力推进!')
elif abs(diff) >= 0.02:
    print('决策门: 2-5% 需更仔细分析，可能有效')
else:
    print('决策门: <2% 方向存疑，需重新评估')
