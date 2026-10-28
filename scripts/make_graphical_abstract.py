"""Graphical abstract for the Neural Networks submission.

Canvas is set to the true print size (13.28 x 5.31 cm) so that the font sizes
below are the sizes the reader will actually see; dpi 508 then yields
2656 x 1062 px, which meets Elsevier's minimum (1328 x 531) with margin.

Panel (a): the measure that leads against accuracy changes with the skew type.
Panel (b): a shift that costs 19 accuracy points while the measured divergence
           stays flat.
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

OUT = "results/figures"
CM = 1 / 2.54

C_JOINT = "#1b4f72"
C_FEAT = "#c0392b"
C_ACC = "#111111"
C_JS = "#8e9a9e"
C_GRID = "#dcdfe2"

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 6.4,
    "axes.linewidth": 0.55,
    "axes.edgecolor": "#7b7d7d",
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.0,
    "ytick.major.size": 2.0,
    "xtick.labelsize": 6.0,
    "ytick.labelsize": 6.0,
})


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------ data
blocks = {"label only": [], "feature only": [], "combined": []}
for f in sorted(glob.glob("results/metric_comparison/*.json")):
    d = load(f)
    m = d.get("metrics") or {}
    if "JS_joint" not in m:
        continue
    if d["skew_type"] == "label_only":
        blocks["label only"].append((m["JS_joint"], m["JS_feature"], d["final_accuracy"]))
    elif d["skew_type"] == "feature_only":
        blocks["feature only"].append((m["JS_joint"], m["JS_feature"], d["final_accuracy"]))
for f in sorted(glob.glob("results/k_selection/aL*_seed42.json")):
    d = load(f)
    if d["alpha_label"] < 100 and d["alpha_feature"] < 100:
        m = d["metrics_by_k"]["k10"]
        blocks["combined"].append((m["JS_joint"], m["JS_feature"], d["final_accuracy"]))

names = ["label only", "feature only", "combined"]
r2j, r2f = [], []
for n in names:
    acc = [b[2] for b in blocks[n]]
    r2j.append(stats.pearsonr([b[0] for b in blocks[n]], acc)[0] ** 2)
    r2f.append(stats.pearsonr([b[1] for b in blocks[n]], acc)[0] ** 2)

doses = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75]
acc_by, js_by = {}, {}
for f in glob.glob("results/intervention/intervention_aL0.3_inv*_seed*.json"):
    d = load(f)
    if d["invert_ratio"] in doses:
        acc_by.setdefault(d["invert_ratio"], []).append(d["final_accuracy"])
for f in glob.glob("results/intervention_v2/intervention_v2_inv*_seed*.json"):
    d = load(f)
    if d["invert_ratio"] in doses:
        js_by.setdefault(d["invert_ratio"], []).append(
            d["metrics_v2_recomputed"]["JS_joint"])
acc_m = np.array([np.mean(acc_by[r]) for r in doses])
acc_s = np.array([np.std(acc_by[r], ddof=1) for r in doses])
js_m = np.array([np.mean(js_by[r]) for r in doses])

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(13.28 * CM, 5.31 * CM), dpi=508)
fig.patch.set_facecolor("white")

axA = fig.add_axes([0.105, 0.215, 0.325, 0.585])
axB = fig.add_axes([0.575, 0.215, 0.300, 0.585])
axB2 = axB.twinx()

# ---------------- (a) dumbbell: which measure leads ----------------------
ys = np.array([2.0, 1.0, 0.0])
for i, y in enumerate(ys):
    axA.plot([r2j[i], r2f[i]], [y, y], color="#aeb6bf", lw=1.0, zorder=2,
             solid_capstyle="round")
axA.scatter(r2j, ys, s=22, color=C_JOINT, zorder=3, marker="o",
            edgecolor="white", linewidth=0.5, label="joint  P(y, c)")
axA.scatter(r2f, ys, s=22, color=C_FEAT, zorder=3, marker="s",
            edgecolor="white", linewidth=0.5, label="feature marginal  P(c)")

for i, y in enumerate(ys):
    gap = r2f[i] - r2j[i]
    if abs(gap) < 0.010:
        txt, col = f"the two coincide ({abs(gap):.3f})", "#7f8c8d"
    elif gap > 0:
        txt, col = f"marginal ahead by {gap:.3f}", C_FEAT
    else:
        txt, col = f"joint ahead by {-gap:.3f}", C_JOINT
    if abs(gap) < 0.010:
        axA.text(1.012, y + 0.25, txt, ha="right", va="center",
                 fontsize=5.8, color=col, style="italic")
    else:
        axA.text(min(r2j[i], r2f[i]) - 0.010, y + 0.25, txt,
                 ha="left", va="center", fontsize=5.8, color=col, style="italic")

axA.set_yticks(ys, names)
axA.set_ylim(-0.62, 2.62)
axA.set_xlim(0.68, 1.02)
axA.set_xticks([0.7, 0.8, 0.9, 1.0])
axA.set_xlabel("R² against accuracy, inside one scenario block", labelpad=1.5)
axA.grid(axis="x", color=C_GRID, lw=0.4, zorder=0)
axA.set_axisbelow(True)
for s in ("top", "right", "left"):
    axA.spines[s].set_visible(False)
axA.tick_params(axis="y", length=0)
axA.legend(loc="upper left", bbox_to_anchor=(-0.01, 1.14), ncol=2, frameon=False,
           handletextpad=0.25, columnspacing=0.8, fontsize=6.0,
           markerscale=1.15, borderaxespad=0)

# ---------------- (b) accuracy collapses, measure flat -------------------
xx = np.array(doses)
axB.errorbar(xx, acc_m, yerr=acc_s, color=C_ACC, marker="o", ms=2.4, lw=1.15,
             capsize=1.6, elinewidth=0.55, zorder=4, label="accuracy")
axB.set_ylim(0.56, 0.88)
axB.set_xlim(-0.03, 0.78)
axB.set_ylabel("Test accuracy", labelpad=1.5)
axB.set_xlabel("fraction of clients with shifted features", labelpad=1.5)
axB.set_yticks([0.6, 0.7, 0.8])
axB.set_xticks([0, 0.25, 0.5, 0.75])
axB.grid(color=C_GRID, lw=0.4, zorder=0)
axB.set_axisbelow(True)
axB.spines["top"].set_visible(False)

axB2.plot(xx, js_m, color=C_JS, marker="s", ms=2.0, lw=1.0, ls=(0, (3, 1.6)),
          zorder=3, label="joint divergence (measured)")
axB2.set_ylim(0.415, 0.468)
axB2.set_yticks([0.42, 0.44, 0.46])
axB2.set_ylabel("JS$_{joint}$ (measured)", color="#6e7b7f", labelpad=1.5)
axB2.tick_params(axis="y", colors="#6e7b7f", labelsize=5.8)
axB2.spines["top"].set_visible(False)

# the two headline numbers, each in a free corner, no leaders
axB.text(0.015, 0.848, f"\u2212{100*(acc_m[0]-acc_m[-1]):.1f} points",
         fontsize=7.2, fontweight="bold", color=C_ACC, va="center")
axB2.text(0.015, 0.4185, "JS$_{joint}$ range %.3f" % (js_m.max()-js_m.min()),
          fontsize=5.9, color="#5d6d7e", va="center")

# ---------------- titles ------------------------------------------------
fig.text(0.105, 0.945, "Which measure leads depends on the skew",
         fontsize=7.6, fontweight="bold", color="#1c2833")
fig.text(0.575, 0.945, "Accuracy collapses, measures do not",
         fontsize=7.6, fontweight="bold", color="#1c2833")
for x0, x1 in ((0.105, 0.425), (0.575, 0.855)):
    fig.add_artist(plt.Line2D([x0, x1], [0.905, 0.905], color="#b3b6b7", lw=0.5))

fig.text(0.105, 0.045,
         "CIFAR-10, 10 clients, 200 rounds. Left: 4 / 4 / 12 runs per block, seed 42. "
         "Right: mean ± SD over 3 seeds; divergence re-measured after the shift.",
         fontsize=5.1, color="#5d6d7e")

fig.savefig(os.path.join(OUT, "graphical_abstract.png"), dpi=508, facecolor="white")
plt.close(fig)
print("saved")
print("a:", [f"{v:.4f}" for v in r2j], [f"{v:.4f}" for v in r2f])
print("b acc", f"{acc_m[0]:.4f}->{acc_m[-1]:.4f}", "| js", f"{js_m.min():.4f}..{js_m.max():.4f}")
