"""分析PACS v2实验结果"""
import json

files = {
    'uniform': 'results/pacs_v2/pacs_uniform_alpha0.5_clients4_seed42.json',
    'two_domain': 'results/pacs_v2/pacs_two_domain_alpha0.5_clients4_seed42.json',
    'four_domain': 'results/pacs_v2/pacs_four_domain_alpha0.5_clients4_seed42.json',
}

results = {}
for name, path in files.items():
    with open(path) as f:
        data = json.load(f)
    last = data[-1]
    best_acc = max(r['accuracy'] for r in data)
    results[name] = {'final_acc': last['accuracy'], 'best_acc': best_acc, 'final_loss': last['loss']}
    print(f'\n=== {name} ===')
    print(f'Final accuracy: {last["accuracy"]:.4f}')
    print(f'Best accuracy:  {best_acc:.4f}')
    print(f'Final loss:     {last["loss"]:.4f}')
    print(f'Total rounds:   {len(data)}')
    print('Key rounds:')
    for r in data:
        if r['round'] in [1, 5, 10, 20, 50, 100, 150, 200]:
            print(f'  Round {r["round"]:3d}: acc={r["accuracy"]:.4f}, loss={r["loss"]:.4f}')

print('\n\n=== COMPARISON ===')
u = results['uniform']['final_acc']
t = results['two_domain']['final_acc']
f = results['four_domain']['final_acc']
print(f'uniform:     {u:.4f} (baseline)')
print(f'two_domain:  {t:.4f} ({t-u:+.4f}, {(t-u)/u*100:+.2f}%)')
print(f'four_domain: {f:.4f} ({f-u:+.4f}, {(f-u)/u*100:+.2f}%)')
