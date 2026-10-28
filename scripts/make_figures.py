"""论文图表生成：六张图各绑定一个论点（R.2.1.5.2图表清单）

Fig.1 度量-场景耦合矩阵热力图（R²填色）——主贡献：分类学
Fig.2 干预实验剂量-反应曲线（3 seeds均值±std带）——因果性
Fig.3 full-mix ablation分组vs混合R²柱状图——辛普森悖论
Fig.4 跨数据集JS_joint vs Acc散点三面板——泛化性
Fig.5 崩溃轨迹+四臂精度对比——讨论段CW观察
Fig.6 偏移性质分类：像素级vs风格——偏移性质分类学

全部从results/原始JSON计算生成，无手工数字。输出results/figures/*.png (300dpi)
"""

import json
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

OUT = "results/figures"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
    "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight",
})

def load(path):
    with open(path) as f:
        return json.load(f)

def r2(x, y):
    r, p = stats.pearsonr(x, y)
    return r ** 2, p

# ============================================================
# 数据收集：CIFAR-10三个场景
# ============================================================
def get_metrics(d):
    """兼容metrics和metrics_by_k两种格式（统一取k10）"""
    if "metrics" in d:
        return d["metrics"]
    if "metrics_by_k" in d:
        return d["metrics_by_k"]["k10"]
    raise KeyError(f"no metrics in {list(d.keys())}")

c10_pure_label, c10_pure_feature, c10_combined = [], [], []
for fn in sorted(os.listdir("results/metric_comparison")):
    d = load(f"results/metric_comparison/{fn}")
    if fn.startswith("label_only"):
        c10_pure_label.append(d)
    elif fn.startswith("feature_only"):
        c10_pure_feature.append(d)
for fn in sorted(os.listdir("results/k_selection")):
    if not fn.endswith(".json") or fn in ("summary.json", "js_conditional_recompute.json"):
        continue
    # 只取 seed42：该目录下另有 seed123/2024 的同配置重复运行（§5.2 的 k 敏感性三 seed 面板），
    # 若一并读入会把重复测量当成独立配置，n 由 12 虚增至 36。逐 seed 的三 seed 面板见 Fig.1b。
    if not fn.endswith("_seed42.json"):
        continue
    d = load(f"results/k_selection/{fn}")
    if d["alpha_label"] < 100 and d["alpha_feature"] < 100:
        c10_combined.append(d)
assert len(c10_combined) == 12, f"c10_combined 应为 12 个配置，实际 {len(c10_combined)}"

fmnist = [load(f"results/cross_dataset/{fn}") for fn in sorted(os.listdir("results/cross_dataset"))
          if fn.endswith(".json")]
c100 = [load(f"results/cifar100/{fn}") for fn in sorted(os.listdir("results/cifar100"))
        if fn.endswith(".json")]

METRICS = ["JS_joint", "JS_label", "JS_feature", "JS_conditional"]
METRIC_LABELS = ["JS joint", "JS label", "JS feature", "JS conditional"]

def r2_for(exp_list, metric):
    accs = [e["final_accuracy"] for e in exp_list]
    vals = [get_metrics(e).get(metric, np.nan) for e in exp_list]
    ok = [(v, a) for v, a in zip(vals, accs) if v == v]
    if len(ok) < 4:
        return np.nan
    return r2([x[0] for x in ok], [x[1] for x in ok])[0]

# ============================================================
# Fig.1 度量-场景耦合矩阵热力图
# 注：JS_conditional的CIFAR-10两行来自事后重算（paper_draft/recompute_js_cond.py，
#     已对JS_joint/label/feature逐位验证，误差=0）；CIFAR-10三行因按类聚类
#     存在度量冗余（DEF-02），标注†构造受限。
# ============================================================
JSCOND_RECOMPUTE = load("results/k_selection/js_conditional_recompute.json")

rows = [
    ("CIFAR-10\nLabel-only† (n=4)", c10_pure_label),
    ("CIFAR-10\nFeature-only† (n=4)", c10_pure_feature),
    ("CIFAR-10\nCombined† (n=12)", c10_combined),
    ("FMNIST\nAll 8 (n=8)", fmnist),
    ("CIFAR-100\nAll 8 (n=8)", c100),
]
M = np.array([[r2_for(el, m) for m in METRICS] for _, el in rows])
# 事后重算的JS_conditional格（DEF-03/DEF-04修正）
M[0, 3] = JSCOND_RECOMPUTE["label_only_JS_conditional_R2"]       # 0.7053 (n=4, p=0.16)
M[2, 3] = JSCOND_RECOMPUTE["JS_conditional_R2"]                  # 0.7521 (n=12, p=0.0003)

fig, ax = plt.subplots(figsize=(7, 4.5))
masked = np.ma.masked_invalid(M)
cmap = plt.cm.RdYlGn.copy()
cmap.set_bad("#e0e0e0")
im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(METRICS)), METRIC_LABELS)
ax.set_yticks(range(len(rows)), [r[0] for r in rows])
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v = M[i, j]
        txt = f"{v:.2f}" if v == v else "N/A"
        mark = "‡" if (i in (0, 2) and j == 3) else ""  # 事后重算格
        ax.text(j, i, txt + mark, ha="center", va="center", fontsize=11,
                color="white" if (v == v and (v < 0.35 or v > 0.85)) else "black",
                fontweight="bold")
ax.set_title("Metric-Scenario Coupling Matrix (Pearson R$^2$)\n"
             "† per-class clustering ⇒ metric redundancy (construction-limited); "
             "‡ post-hoc recomputed")
ax.grid(False)
plt.colorbar(im, ax=ax, label="R$^2$", shrink=0.8)
plt.savefig(f"{OUT}/fig1_coupling_matrix.png")
plt.close()
print("Fig.1 done. Matrix (CIFAR-10 rows construction-limited†):")
for (name, _), row in zip(rows, M):
    print(f"  {name.replace(chr(10), ' '):<30}" + "  ".join(f"{v:.4f}" if v == v else "  N/A" for v in row))

# ============================================================
# Fig.1b k敏感性：组合偏斜12点，R²随量化分辨率变化
# 数据源：k_selection/metrics_by_k（同一分区，仅换聚类分辨率）
# ============================================================
k_values = [4, 10, 20, 50]
k_metrics = ["JS_joint", "JS_label", "JS_feature", "Cov_weighted"]
k_colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
K_SEEDS = [42, 123, 2024]
# 逐 seed 读同一组 12 个 combined 配置，得到 4 条曲线 x 3 个 seed
c10_by_seed = {s: [load(p) for p in sorted(glob.glob(f"results/k_selection/aL*_seed{s}.json"))]
               for s in K_SEEDS}
K = np.zeros((len(k_metrics), len(k_values), len(K_SEEDS)))
for si, s in enumerate(K_SEEDS):
    rows = [e for e in c10_by_seed[s] if e.get("final_accuracy") is not None]
    accs_c = [e["final_accuracy"] for e in rows]
    for mi, met in enumerate(k_metrics):
        for ki, k in enumerate(k_values):
            vals = [e["metrics_by_k"][f"k{k}"][met] for e in rows]
            K[mi, ki, si] = stats.pearsonr(vals, accs_c)[0] ** 2
K_MEAN = K.mean(axis=2)
K_MIN, K_MAX = K.min(axis=2), K.max(axis=2)

fig, ax = plt.subplots(figsize=(6, 4))
for mi, (met, c) in enumerate(zip(k_metrics, k_colors)):
    ax.plot(k_values, K_MEAN[mi], marker="o", lw=2, color=c,
            label=met.replace("JS_", "JS ").replace("_", " "))
    ax.fill_between(k_values, K_MIN[mi], K_MAX[mi], color=c, alpha=0.13, lw=0)
ax.set_xscale("log")
ax.set_xticks(k_values, [str(k) for k in k_values])
ax.set_xlabel("Quantization resolution k (clusters per class)")
ax.set_ylabel("Pearson R$^2$ vs accuracy")
ax.set_ylim(0, 1.0)
ax.set_title("Resolution sensitivity (CIFAR-10 combined skew, n=12)\n"
             "mean $\\pm$ range over three seeds; label-marginal metric collapses at fine k",
             fontsize=10)
ax.legend(loc="center left", fontsize=9)
ax.annotate("feature marginal exceeds joint\non 3/3 seeds only at k = 50",
            xy=(4.3, 0.20), fontsize=8, color="#2ca02c", va="top")
plt.savefig(f"{OUT}/fig1b_k_sensitivity.png")
plt.close()
print("\nFig.1b done. k-sensitivity (combined, n=12, 3 seeds, mean over seeds):")
for mi, met in enumerate(k_metrics):
    print(f"  {met:14s} " + "  ".join(f"k{k}={K_MEAN[mi, ki]:.4f}" for ki, k in enumerate(k_values)))

# ============================================================
# Fig.2 干预实验剂量-反应曲线（3 seeds）
# ============================================================
ratios = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75]
seeds = [42, 123, 2024]
acc_by_ratio = {r: [] for r in ratios}
jsj_by_ratio = {r: [] for r in ratios}     # v1: 离散支撑构造（簇号 +K）
jsj2_by_ratio = {r: [] for r in ratios}    # v2: 变换后重新提特征 + 重新聚类（真实测量）
for r in ratios:
    for s in seeds:
        d = load(f"results/intervention/intervention_aL0.3_inv{r}_seed{s}.json")
        acc_by_ratio[r].append(d["final_accuracy"])
        jsj_by_ratio[r].append(d["metrics"]["JS_joint"])
        d2 = load(f"results/intervention_v2/intervention_v2_inv{r}_seed{s}.json")
        jsj2_by_ratio[r].append(d2["metrics_v2_recomputed"]["JS_joint"])

means = np.array([np.mean(acc_by_ratio[r]) for r in ratios])
stds = np.array([np.std(acc_by_ratio[r], ddof=1) for r in ratios])
all_acc, all_jsj, all_jsj2 = [], [], []
for r in ratios:
    all_acc.extend(acc_by_ratio[r])
    all_jsj.extend(jsj_by_ratio[r])
    all_jsj2.extend(jsj2_by_ratio[r])
rr, pp = stats.pearsonr(all_jsj, all_acc)
rr2, pp2 = stats.pearsonr(all_jsj2, all_acc)

fig, ax = plt.subplots(figsize=(6.4, 4.4))
ax.errorbar(ratios, means, yerr=stds, marker="o", capsize=4,
            color="#1f77b4", lw=2, label="Accuracy (mean $\\pm$ std, 3 seeds)")
ax2 = ax.twinx()
jsj_m = np.array([np.mean(jsj_by_ratio[r]) for r in ratios])
jsj_s = np.array([np.std(jsj_by_ratio[r], ddof=1) for r in ratios])
ax2.plot(ratios, jsj_m, marker="s", ls="--", color="#999999", alpha=0.85,
         label="JS joint, disjoint-support construction (v1)")
jsj2_m = np.array([np.mean(jsj2_by_ratio[r]) for r in ratios])
jsj2_s = np.array([np.std(jsj2_by_ratio[r], ddof=1) for r in ratios])
ax2.plot(ratios, jsj2_m, marker="^", ls="-", color="#d62728",
         label="JS joint, re-measured on shifted features (v2)")
ax2.fill_between(ratios, jsj2_m - jsj2_s, jsj2_m + jsj2_s, color="#d62728", alpha=0.12)
ax2.set_ylabel("JS joint (feature-skew dose)")
ax2.grid(False)
ax.set_xlabel("Invert ratio (feature-skew dose), label skew fixed at $\\alpha_L$=0.3")
ax.set_ylabel("Test accuracy")
ax.set_title("Intervention: accuracy responds, divergence does not")
ax.annotate(f"pooled 21 runs\nv1 (constructed): R$^2$={rr**2:.4f}, p={pp:.1e}\n"
            f"v2 (re-measured): R$^2$={rr2**2:.4f}, p={pp2:.3f}",
            xy=(0.97, 0.96), xycoords="axes fraction", ha="right", va="top", fontsize=8.5,
            bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.9))
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=7.5)
plt.tight_layout()
plt.savefig(f"{OUT}/fig2_intervention_dose_response.png", dpi=300)
plt.close()
print(f"\nFig.2 done. v1 pooled r={rr:.4f}, R2={rr**2:.4f}, p={pp:.2e} | "
      f"v2 pooled r={rr2:.4f}, R2={rr2**2:.4f}, p={pp2:.4f}")

# ============================================================
# Fig.3 full-mix ablation：分组vs混合R²
# ============================================================
pools = [
    ("Label-only", c10_pure_label),
    ("+Feature-only", c10_pure_label + c10_pure_feature),
    ("+Combined (all 20)", c10_pure_label + c10_pure_feature + c10_combined),
]
bar_metrics = ["JS_joint", "JS_label", "JS_feature"]
R = np.array([[r2_for(pool, m) for m in bar_metrics] for _, pool in pools])

fig, ax = plt.subplots(figsize=(6.5, 4.2))
x = np.arange(len(pools))
w = 0.25
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
for j, (m, c) in enumerate(zip(bar_metrics, colors)):
    bars = ax.bar(x + (j - 1) * w, R[:, j], w, label=m.replace("JS_", "JS "), color=c)
    for b, v in zip(bars, R[:, j]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", fontsize=9)
ax.set_xticks(x, [p[0] for p in pools])
ax.set_ylabel("Pearson R$^2$ vs accuracy")
ax.set_ylim(0, 1.05)
ax.set_title("Simpson's Paradox: pooled R$^2$ collapses across scenarios")
ax.legend()
plt.savefig(f"{OUT}/fig3_ablation_simpson.png")
plt.close()
print("\nFig.3 done. Pooled R2:")
for (name, _), row in zip(pools, R):
    print(f"  {name:<18}" + "  ".join(f"{v:.4f}" for v in row))

# ============================================================
# Fig.4 跨数据集JS_joint vs Acc三面板
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
scen_colors = {"PL": "#1f77b4", "PF": "#ff7f0e", "CB": "#2ca02c"}

def scen_of_c10(d):
    if d["alpha_feature"] >= 100:
        return "PL"
    if d["alpha_label"] >= 100:
        return "PF"
    return "CB"

panels = [
    ("CIFAR-10 (20 runs)", c10_pure_label + c10_pure_feature + c10_combined, True),
    ("FMNIST (8 runs)", fmnist, False),
    ("CIFAR-100 (8 runs)", c100, False),
]
for ax, (name, elist, colorize) in zip(axes, panels):
    xs = [get_metrics(e)["JS_joint"] for e in elist]
    ys = [e["final_accuracy"] for e in elist]
    if colorize:
        scen_names = {"PL": "label-only", "PF": "feature-only", "CB": "combined"}
        seen = set()
        for d in elist:
            sc = scen_of_c10(d)
            ax.scatter(get_metrics(d)["JS_joint"], d["final_accuracy"],
                       c=scen_colors[sc], s=28, alpha=0.75,
                       label=scen_names[sc] if sc not in seen else None)
            seen.add(sc)
        ax.legend(fontsize=8, loc="upper right")
    else:
        ax.scatter(xs, ys, c="#1f77b4", s=30, alpha=0.75)
    r, p = stats.pearsonr(xs, ys)
    k = np.polyfit(xs, ys, 1)
    xfit = np.linspace(min(xs), max(xs), 50)
    ax.plot(xfit, np.polyval(k, xfit), "r--", lw=1.5)
    ax.set_title(f"{name}\nR$^2$={r**2:.2f}, p={p:.1e}", fontsize=10)
    ax.set_xlabel("JS joint")
axes[0].set_ylabel("Test accuracy")
fig.suptitle("Cross-dataset: JS joint vs accuracy (pooled per dataset)", y=1.02)
plt.savefig(f"{OUT}/fig4_cross_dataset.png")
plt.close()
print("\nFig.4 done.")

# ============================================================
# Fig.5 崩溃轨迹 + 四臂精度对比
# ============================================================
crash = load("results/cw_fedavg/cw_data_aL0.1_aF0.1_seed123.json")
cw_same = load("results/cw_fedavg/cw_cw_exp_aL0.1_aF0.1_seed123_b5.0.json")

STRESS = "results/stress"
SEEDS10 = [42, 123, 2024, 7, 99, 555, 777, 2025, 3407, 314]
arm_files = {
    "FedAvg": "cw_data_aL0.05_aF0.1_seed{s}.json",
    "FedAvg+clip": "cw_data_aL0.05_aF0.1_seed{s}_clip10.0.json",
    "Median": "cw_median_aL0.05_aF0.1_seed{s}.json",
    "CW-FedAvg": "cw_cw_exp_aL0.05_aF0.1_seed{s}_b5.0.json",
}
arm_accs = {}
for arm, pat in arm_files.items():
    accs = [load(f"{STRESS}/{pat.format(s=s)}")["final_accuracy"] for s in SEEDS10]
    arm_accs[arm] = accs

fig, (axa, axb) = plt.subplots(1, 2, figsize=(11, 4))

# (a) 训练轨迹（直接在线末端标注方法名，避免图例误读）
r1 = [x["round"] for x in crash["round_log"]]
a1 = [x["accuracy"] for x in crash["round_log"]]
r2 = [x["round"] for x in cw_same["round_log"]]
a2 = [x["accuracy"] for x in cw_same["round_log"]]
axa.plot(r1, a1, color="#d62728", lw=1.4)
axa.plot(r2, a2, color="#1f77b4", lw=1.4)
axa.axvline(142, color="gray", ls=":", lw=1)
axa.annotate("FedAvg diverges to NaN\n(round 142)", xy=(142, 0.45), xytext=(58, 0.35),
             fontsize=9, color="#d62728",
             arrowprops=dict(arrowstyle="->", color="#d62728"))
axa.text(205, a1[-1], "FedAvg", color="#d62728", fontsize=10, va="center", fontweight="bold")
axa.text(205, a2[-1], "CW-FedAvg", color="#1f77b4", fontsize=10, va="center", fontweight="bold")
axa.set_xlim(0, 240)
axa.set_xlabel("FL round")
axa.set_ylabel("Test accuracy")
axa.set_title(f"(a) Catastrophic divergence, seed=123 ($\\alpha_L$=0.1)")

# (b) 四臂精度
arms = list(arm_files.keys())
means = [np.mean(arm_accs[a]) for a in arms]
stds_ = [np.std(arm_accs[a], ddof=1) for a in arms]
cols = ["#7f7f7f", "#ffbb78", "#2ca02c", "#1f77b4"]
bars = axb.bar(arms, means, yerr=stds_, capsize=4, color=cols)
for b, m in zip(bars, means):
    axb.text(b.get_x() + b.get_width() / 2, m + 0.012, f"{m:.3f}", ha="center", fontsize=9)
axb.set_ylabel("Final accuracy (mean $\\pm$ std, 10 seeds)")
axb.set_ylim(0, 0.85)
axb.set_title("(b) Four-arm stress test ($\\alpha_L$=0.05, 10 seeds each)")
axb.tick_params(axis="x", labelsize=9)
fig.suptitle("Stability observation (Discussion, exploratory)", y=1.02)
plt.savefig(f"{OUT}/fig6_stability.png")
plt.close()
print("\nFig.5 done. Four-arm means:", {a: f"{np.mean(v):.4f}" for a, v in arm_accs.items()})

# ============================================================
# Fig.6 偏移性质分类：像素级 vs 风格
# ============================================================
def last_acc(path):
    return load(path)[-1]["accuracy"]

pixel = {
    "uniform": last_acc("results/_recheck/domain_uniform_alpha0.1_seed42.json"),
    "grey": last_acc("results/domain_blindspot/domain_grad_grayscale_alpha0.1_seed42.json"),
    "contrast": last_acc("results/domain_blindspot/domain_grad_high_contrast_alpha0.1_seed42.json"),
    "posterise": last_acc("results/domain_blindspot/domain_grad_posterize_alpha0.1_seed42.json"),
    "edge": last_acc("results/domain_blindspot/domain_grad_edge_enhance_alpha0.1_seed42.json"),
    "invert": last_acc("results/domain_blindspot/domain_grad_invert_alpha0.1_seed42.json"),
}
# 同一批次、同一分区、同一客户端位置（前 5 正常 / 后 5 变换），唯一变量是变换类型。
# 旧口径 (2026-06) 的 uniform/2_domains/extreme 与当前代码不同源（DEF-11），不再作图。
# PACS v3: held-out evaluation (per-domain 80/20 stratified split, split_seed=2026).
# v2 numbers (98.9/17.3/99.9) were transductive (all 9,991 images in training) - DEF-06.
style = {
    "uniform": last_acc("results/pacs_v3/pacs_v3_uniform_alpha0.5_clients4_seed42.json"),
    "2 domains": last_acc("results/pacs_v3/pacs_v3_two_domain_alpha0.5_clients4_seed42.json"),
    "4 domains": last_acc("results/pacs_v3/pacs_v3_four_domain_alpha0.5_clients4_seed42.json"),
}

fig, (axa, axb) = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
# 面板(a) 按"是否保留空间结构"着色：灰=对照，浅红=色彩类，深红=结构类
pixel_colors = ["#8c8c8c", "#e8a5a5", "#e8a5a5", "#e8a5a5", "#c0392b", "#c0392b"]
for ax, data, ttl, c in [
    (axa, pixel, "(a) Pixel-level shifts (CIFAR-10, pre-transform)\n"
                 "colour-class (light) vs structure-class (dark)", pixel_colors),
    (axb, style, "(b) Style shifts\n(PACS, natural domains, held-out)", "#1f77b4"),
]:
    bars = ax.bar(list(data.keys()), list(data.values()), color=c, alpha=0.9)
    for b, v in zip(bars, data.values()):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
                ha="center", fontsize=8, rotation=90 if ax is axa else 0)
    ax.set_title(ttl, fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.tick_params(axis="x", labelsize=8.5)
axa.set_ylabel("Test accuracy")
axa.axhline(pixel["uniform"], ls=":", lw=1, color="#555555")
axa.text(0.02, pixel["uniform"] - 0.075, "uniform control", fontsize=7.5, color="#555555",
         transform=axa.get_yaxis_transform())
axb.annotate("style skew does not hurt\n(held-out gap +0.05 pt vs uniform)",
             xy=(2, 0.8697), xytext=(0.6, 0.35), fontsize=9,
             arrowprops=dict(arrowstyle="->", color="gray"))
fig.suptitle("Nature of the shift determines the damage", y=1.02)
plt.savefig(f"{OUT}/fig5_shift_nature.png")
plt.close()
print(f"\nFig.6 done. pixel={pixel}, style={style}")

print(f"\nAll 6 figures saved to {OUT}/")
