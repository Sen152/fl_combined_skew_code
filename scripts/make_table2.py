"""Regenerate Table 2 of the manuscript from the stored result artefacts.

Table 2 lists the final accuracy of every run in the scenario grid:
CIFAR-10 (4 label-skew, 4 feature-skew and 12 combined runs, seed 42),
FMNIST (8 runs) and CIFAR-100 (8 runs). It is printed as markdown so that the
table in the paper can be diffed against it.

Usage:  python scripts/make_table2.py
"""
import glob
import json
import os

A_INF = 100  # alpha values at or above this are treated as "axis effectively absent"


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def scenario(alpha_label, alpha_feature):
    lab = alpha_label < A_INF
    fea = alpha_feature < A_INF
    if lab and fea:
        return "combined"
    if lab:
        return "label skew only"
    if fea:
        return "feature skew only"
    return "no skew"


rows = []

# CIFAR-10: label-only and feature-only blocks
for f in sorted(glob.glob("results/metric_comparison/*.json")):
    d = load(f)
    if d["alpha_label"] < A_INF and d["alpha_feature"] < A_INF:
        continue  # combined block comes from k_selection
    rows.append(("CIFAR-10", d["alpha_label"], d["alpha_feature"], d["final_accuracy"]))

# CIFAR-10: combined block (seed 42 only)
for f in sorted(glob.glob("results/k_selection/aL*_seed42.json")):
    d = load(f)
    if d["alpha_label"] < A_INF and d["alpha_feature"] < A_INF:
        rows.append(("CIFAR-10", d["alpha_label"], d["alpha_feature"], d["final_accuracy"]))

# FMNIST and CIFAR-100, eight runs each (stored in different directories)
for sub, prefix, tag in (("cross_dataset", "fmnist", "FMNIST"),
                         ("cifar100", "cifar100", "CIFAR-100")):
    for f in sorted(glob.glob(f"results/{sub}/{prefix}_*.json")):
        d = load(f)
        rows.append((tag, d["alpha_label"], d["alpha_feature"], d["final_accuracy"]))

order = {"feature skew only": 0, "label skew only": 1, "combined": 2, "no skew": 3}
rows.sort(key=lambda r: (["CIFAR-10", "FMNIST", "CIFAR-100"].index(r[0]),
                         order[scenario(r[1], r[2])], r[1], r[2]))


def fmt(a):
    return f"{a:g}"


print("| Dataset | Scenario | $\\alpha_L$ | $\\alpha_F$ | Final accuracy |")
print("|---|---|---|---|---|")
for ds, al, af, acc in rows:
    print(f"| {ds} | {scenario(al, af)} | {fmt(al)} | {fmt(af)} | {acc:.4f} |")
print()
print(f"rows: {len(rows)}")
