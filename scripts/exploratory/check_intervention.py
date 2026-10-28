"""Quick check which JSON files are valid"""
import json, os
result_dir = "results/intervention"
for f in sorted(os.listdir(result_dir)):
    if f.endswith(".json"):
        try:
            data = json.load(open(os.path.join(result_dir, f)))
            print(f"OK  {f}: acc={data.get('final_accuracy', 'N/A')}")
        except Exception as e:
            print(f"BAD {f}: {e}")
