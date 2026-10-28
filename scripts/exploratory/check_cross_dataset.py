"""Quick check cross-dataset results"""
import json, os
d = "results/cross_dataset"
for f in sorted(os.listdir(d)):
    if f.endswith(".json"):
        data = json.load(open(os.path.join(d, f)))
        m = data.get('metrics', {})
        print(f"{f}: acc={data['final_accuracy']:.4f}, JS_joint={m.get('JS_joint',0):.4f}, JS_label={m.get('JS_label',0):.4f}, JS_feature={m.get('JS_feature',0):.4f}")
