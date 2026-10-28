"""§5.3 干预实验：v1（+K 构造）与 v2（真实重测）的 21 次运行统计

用途：为 §5.3 改写提供可核对数字。
环境：C:\\Users\\Windows\\miniconda3\\python.exe
"""

import json
import os
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V1 = os.path.join(ROOT, "results", "intervention")
V2 = os.path.join(ROOT, "results", "intervention_v2")

DOSES = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75]
SEEDS = [42, 123, 2024]


def acc_of(path):
    with open(path) as f:
        j = json.load(f)
    return j["final_accuracy"]


def v1_metrics(path):
    with open(path) as f:
        j = json.load(f)
    return j.get("metrics", {})


def v2_metrics(path):
    with open(path) as f:
        j = json.load(f)
    return j.get("metrics_v2_recomputed", {})


def main():
    rows = []
    for s in SEEDS:
        for r in DOSES:
            p1 = os.path.join(V1, f"intervention_aL0.3_inv{r}_seed{s}.json")
            p2 = os.path.join(V2, f"intervention_v2_inv{r}_seed{s}.json")
            if not (os.path.exists(p1) and os.path.exists(p2)):
                print(f"MISSING seed={s} r={r}")
                continue
            a = acc_of(p1)
            m = v2_metrics(p2)
            m1 = v1_metrics(p1)
            rows.append({
                "seed": s, "r": r, "acc": a,
                "js_joint": m.get("JS_joint"),
                "js_conditional": m.get("JS_conditional"),
                "js_feature": m.get("JS_feature"),
                "js_label": m.get("JS_label"),
                "js_joint_v1": m1.get("JS_joint"),
            })

    print(f"共 {len(rows)} 条（应为 21）\n")

    # ---- 剂量均值 ----
    print("| dose | mean acc | mean JS_joint(v2) | mean JS_cond(v2) | mean JS_feat(v2) |")
    print("|---|---|---|---|---|")
    for r in DOSES:
        sub = [x for x in rows if x["r"] == r]
        if not sub:
            continue
        f = lambda k: np.mean([x[k] for x in sub if x[k] is not None])
        print(f"| {r} | {f('acc'):.4f} | {f('js_joint'):.4f} | "
              f"{f('js_conditional'):.4f} | {f('js_feature'):.4f} |")

    # ---- 池化回归：acc vs JS_joint（v2）----
    print("\n=== 池化回归（n=21，acc vs v2 JS_joint）===")
    for key in ("js_joint", "js_conditional", "js_feature", "js_label"):
        xs = np.array([x[key] for x in rows if x[key] is not None])
        ys = np.array([x["acc"] for x in rows if x[key] is not None])
        if len(xs) < 3:
            print(f"{key}: 数据不足")
            continue
        lr = stats.linregress(xs, ys)
        rho, prho = stats.spearmanr(xs, ys)
        print(f"{key:16s} n={len(xs):2d}  r={lr.rvalue:+.4f}  R²={lr.rvalue**2:.4f}  "
              f"p={lr.pvalue:.4f}  rho={rho:+.4f} (p={prho:.4f})")

    # ---- 每 seed 单独回归 ----
    print("\n=== 每 seed 单独（7 点，acc vs v2 JS_joint）===")
    for s in SEEDS:
        sub = [x for x in rows if x["seed"] == s]
        xs = np.array([x["js_joint"] for x in sub])
        ys = np.array([x["acc"] for x in sub])
        lr = stats.linregress(xs, ys)
        print(f"seed {s:5d}  R²={lr.rvalue**2:.4f}  p={lr.pvalue:.4f}")

    # ---- 与剂量本身的相关（说明"度量是否随剂量单调"）----
    print("\n=== v2 JS_joint 与剂量 r 的相关 ===")
    xs = np.array([x["r"] for x in rows])
    ys = np.array([x["js_joint"] for x in rows])
    lr = stats.linregress(xs, ys)
    rho, prho = stats.spearmanr(xs, ys)
    print(f"R²={lr.rvalue**2:.4f}  p={lr.pvalue:.4f}  rho={rho:+.4f} (p={prho:.4f})")

    # ---- 输出 JSON 供后续引用 ----
    out = os.path.join(ROOT, "results", "intervention_v2", "_summary_3seeds.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n明细已存: {out}")


if __name__ == "__main__":
    main()
