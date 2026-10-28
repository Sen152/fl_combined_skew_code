"""§5.2 k 敏感性的三 seed 稳定性分析。

问题：论文报告 JS_joint 的 R² 随 k 单调上升（0.7230 / 0.7519 / 0.7828 / 0.8163），
且 JS_feature 在 k>=20 反超。这两条都来自 seed42 单次运行，间隔不到 1 个点。
本脚本用 seed 123 / 2024 的补跑检验：排序是否跨 seed 稳定，差距是否超过噪声底（0.70 pt 准确率 / ~0.02 R²）。
"""
import glob
import json
import os

import numpy as np
from scipy import stats

D = "results/k_selection"
KS = ["k4", "k10", "k20", "k50"]
METRICS = ["JS_joint", "JS_conditional", "JS_feature", "JS_label", "Cov_weighted"]
SEEDS = [42, 123, 2024]


def load(seed):
    fs = sorted(glob.glob(os.path.join(D, f"aL*_seed{seed}.json")))
    rows = []
    for f in fs:
        j = json.load(open(f))
        if j.get("final_accuracy") is None or not j.get("metrics_by_k"):
            continue
        rows.append(j)
    return rows


def r2_table(rows, metric):
    """每个 k 下，用 12 个配置的 metric 值回归 final_accuracy，返回 R² 与 p。"""
    acc = np.array([r["final_accuracy"] for r in rows])
    out = {}
    for k in KS:
        x = np.array([r["metrics_by_k"][k][metric] for r in rows])
        if np.std(x) < 1e-12:
            out[k] = (float("nan"), float("nan"))
            continue
        res = stats.linregress(x, acc)
        out[k] = (res.rvalue ** 2, res.pvalue)
    return out


def main():
    print("=" * 78)
    print("§5.2 k 敏感性：三 seed 稳定性")
    print("=" * 78)

    data = {}
    for s in SEEDS:
        rows = load(s)
        data[s] = rows
        accs = [r["final_accuracy"] for r in rows]
        print(f"seed {s:4d}: n={len(rows):2d}  acc {min(accs):.4f}-{max(accs):.4f}  "
              f"跨距 {max(accs)-min(accs):.4f}")

    # ---- 1. 每个 seed 的 R² 表 ----
    for metric in ["JS_joint", "JS_feature"]:
        print()
        print(f"--- {metric}: R²（行=seed，列=k，n=12 配置）---")
        print(f"{'seed':>6s}" + "".join(f"{k:>10s}" for k in KS) + f"{'排序':>14s}")
        tabs = {}
        for s in SEEDS:
            tab = r2_table(data[s], metric)
            tabs[s] = tab
            vals = [tab[k][0] for k in KS]
            mono = all(vals[i] <= vals[i + 1] for i in range(3))
            anti = all(vals[i] >= vals[i + 1] for i in range(3))
            tag = "单调升" if mono else ("单调降" if anti else "非单调")
            print(f"{s:>6d}" + "".join(f"{v:>10.4f}" for v in vals) + f"{tag:>14s}")

        # 跨 seed 的均值和标准差
        arr = np.array([[tabs[s][k][0] for k in KS] for s in SEEDS])
        print(f"{'均值':>6s}" + "".join(f"{v:>10.4f}" for v in arr.mean(axis=0)))
        print(f"{'SD':>6s}" + "".join(f"{v:>10.4f}" for v in arr.std(axis=0, ddof=1)))
        print(f"{'极差':>6s}" + "".join(f"{v:>10.4f}" for v in (arr.max(axis=0) - arr.min(axis=0))))

        # 关键检验：k50 vs k4 的差，是否大于跨 seed 波动
        d_seed42 = tabs[42]["k50"][0] - tabs[42]["k4"][0]
        d_mean = arr.mean(axis=0)[3] - arr.mean(axis=0)[0]
        noise = arr.std(axis=0, ddof=1).mean()
        print(f"\n  k50-k4: seed42 = {d_seed42:+.4f} | 三 seed 均值 = {d_mean:+.4f} "
              f"| 跨 seed SD(均值) = {noise:.4f}")
        print(f"  判定：{'差异超过噪声' if abs(d_mean) > 2*noise else '差异与噪声同量级（不可主张）'}")

        # 逐 seed 看 k50 是否仍优于 k4
        better = sum(1 for s in SEEDS if tabs[s]["k50"][0] > tabs[s]["k4"][0])
        print(f"  三个 seed 中 k50 > k4 的有 {better}/3")

    # ---- 2. JS_feature 反超 JS_joint 的检验 ----
    print()
    print("--- 论文主张：JS_feature 在 k>=20 反超 JS_joint ---")
    print(f"{'seed':>6s}{'k':>6s}{'JS_joint R²':>14s}{'JS_feature R²':>15s}{'反超?':>8s}")
    over = {k: 0 for k in KS}
    for s in SEEDS:
        tj = r2_table(data[s], "JS_joint")
        tf = r2_table(data[s], "JS_feature")
        for k in KS:
            flag = tf[k][0] > tj[k][0]
            over[k] += int(flag)
            print(f"{s:>6d}{k:>6s}{tj[k][0]:>14.4f}{tf[k][0]:>15.4f}{'是' if flag else '否':>8s}")
    print()
    print("跨 seed 反超次数：" + "  ".join(f"{k}={over[k]}/3" for k in KS))


if __name__ == "__main__":
    main()
