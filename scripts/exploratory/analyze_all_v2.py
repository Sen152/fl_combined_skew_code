"""全面分析所有盲区实验结果"""
import json, os

def load_results(directory):
    results = {}
    if not os.path.exists(directory):
        return results
    for f in sorted(os.listdir(directory)):
        if f.endswith('.json'):
            with open(os.path.join(directory, f)) as fp:
                data = json.load(fp)
            results[f.replace('.json', '')] = data
    return results

print('=' * 80)
print('盲区实验全面分析')
print('=' * 80)

# 1. k-means删样本实验（CIFAR-10）
print('\n### 实验1：k-means删样本（CIFAR-10）###')
r1 = load_results('results/blindspot')
for name, data in sorted(r1.items()):
    print(f'  {name}: final_acc={data[-1]["accuracy"]:.4f}')

# 2. 风格增强实验（CIFAR-10，训练时随机增强）
print('\n### 实验2：训练时随机风格增强（CIFAR-10）###')
r2 = load_results('results/style_blindspot')
for name, data in sorted(r2.items()):
    print(f'  {name}: final_acc={data[-1]["accuracy"]:.4f}')

# 3. 预变换域偏斜实验（CIFAR-10，不可逆变换）
print('\n### 实验3：预变换域偏斜（CIFAR-10）###')
r3 = load_results('results/domain_blindspot')
for name, data in sorted(r3.items()):
    print(f'  {name}: final_acc={data[-1]["accuracy"]:.4f}')

# 4. PACS实验
print('\n### 实验4：PACS自然域偏移 ###')
r4 = load_results('results/pacs_blindspot')
for name, data in sorted(r4.items()):
    print(f'  {name}: final_acc={data[-1]["accuracy"]:.4f}')

# 关键对比
print('\n' + '=' * 80)
print('关键对比')
print('=' * 80)

# 预变换实验
if r3:
    ctrl = r3.get('domain_uniform_alpha0.1_seed42')
    two = r3.get('domain_two_domain_alpha0.1_seed42')
    ext = r3.get('domain_extreme_alpha0.1_seed42')
    if ctrl and two:
        diff = ctrl[-1]['accuracy'] - two[-1]['accuracy']
        print(f'预变换 two_domain vs uniform: {diff*100:.2f}%')
    if ctrl and ext:
        diff = ctrl[-1]['accuracy'] - ext[-1]['accuracy']
        print(f'预变换 extreme vs uniform: {diff*100:.2f}%')

# PACS实验
if r4:
    pacs_ctrl = r4.get('pacs_uniform_photo_alpha0.5_clients4_seed42')
    pacs_four = r4.get('pacs_four_domain_alpha0.5_clients4_seed42')
    if pacs_ctrl and pacs_four:
        diff = pacs_ctrl[-1]['accuracy'] - pacs_four[-1]['accuracy']
        print(f'PACS four_domain vs uniform_photo: {diff*100:.2f}%')
    elif pacs_ctrl:
        print(f'PACS four_domain 还在跑...')

print('\n### 结论 ###')
print('1. k-means删样本：不产生真正的域偏移，差异<2.5%')
print('2. 训练时随机增强：反而帮助泛化，差异为正')
print('3. 预变换域偏斜：✅ 成功！grayscale -2.35%, invert -9.48%')
print('4. PACS自然域偏移：待出结果')
print('\n核心发现：特征偏斜确实有害，但前提是偏斜必须是"不可逆的视觉域差异"')
print('  - 删样本 ≠ 换分布（只是少了数据，不是不同类型的数据）')
print('  - 随机增强 ≠ 域偏移（每个epoch都不同，模型看到的是多样化数据）')
print('  - 预变换 = 真正的域偏移（不同客户端看到的是固定不同的视觉世界）')
