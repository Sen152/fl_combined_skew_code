import json
with open('results/pacs_blindspot/pacs_uniform_photo_alpha0.5_clients4_seed42.json') as f:
    ctrl = json.load(f)
with open('results/pacs_blindspot/pacs_four_domain_alpha0.5_clients4_seed42.json') as f:
    four = json.load(f)

ctrl_final = ctrl[-1]['accuracy']
four_final = four[-1]['accuracy']
diff = ctrl_final - four_final

print(f'PACS uniform_photo: final_acc={ctrl_final:.4f}')
print(f'PACS four_domain:   final_acc={four_final:.4f}')
print(f'差异: {diff*100:.2f}%')
print(f'R50:  ctrl={ctrl[49]["accuracy"]:.4f}, four={four[49]["accuracy"]:.4f}')
print(f'R100: ctrl={ctrl[99]["accuracy"]:.4f}, four={four[99]["accuracy"]:.4f}')
print(f'R200: ctrl={ctrl[199]["accuracy"]:.4f}, four={four[199]["accuracy"]:.4f}')
