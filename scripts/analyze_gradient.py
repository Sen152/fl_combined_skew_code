"""像素级偏移梯度分析：把 §5.6 的"两点对照"升级为"强度梯度"

统一口径：一律 5 正常 + 5 变换，客户端位置固定（client 0-4 正常，5-9 变换），
标签分区固定为 alpha_L=0.1 / seed=42，故横排可直接比较，唯一变量是变换类型。

用法：
    python scripts/analyze_gradient.py
"""

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "results", "domain_blindspot")

# 统一口径的梯度臂（grad_ 前缀），按"破坏程度"从轻到重人工排序待实测检验
GRADIENT = [
    ("grad_high_contrast", "高对比度", "high_contrast"),
    ("grad_edge_enhance", "边缘增强", "edge_enhance"),
    ("grad_posterize", "色调分离", "posterize"),
    ("grad_grayscale", "灰度", "grayscale"),
    ("grad_invert", "颜色反转", "invert"),
]
# 原有 baseline（口径与 grad_* 不同，仅作参照）
LEGACY = [
    ("uniform", "全正常（对照）"),
    ("two_domain", "5正常+5灰度（原口径）"),
    ("extreme", "5正常+5反色（原口径）"),
]


def load(name):
    """name 为 domain_mode，落盘文件名为 domain_<mode>_alpha0.1_seed42.json"""
    p = os.path.join(DB, f"domain_{name}_alpha0.1_seed42.json")
    if not os.path.exists(p):
        return None, p
    with open(p) as f:
        return json.load(f), p


def summarize(log):
    accs = [r["accuracy"] for r in log]
    return {
        "final": log[-1]["accuracy"],
        "best": max(accs),
        "best_round": accs.index(max(accs)) + 1,
        "loss": log[-1]["loss"],
        "nan_loss": any(r["loss"] != r["loss"] for r in log),
    }


def main():
    print("=" * 72)
    print("像素级预变换梯度（统一口径：5 正常 + 5 变换，位置固定）")
    print("=" * 72)

    base = None
    log, _ = load("uniform")
    if log:
        base = log[-1]["accuracy"]
        print(f"\n基线（全正常，原口径）: {base:.4f}\n")

    print(f"{'变换':<12} {'终值acc':>9} {'相对基线':>10} {'峰值acc':>9} {'峰值轮':>7} {'终loss':>8}")
    print("-" * 72)

    rows = []
    missing = []
    for name, label, tag in GRADIENT:
        log, path = load(name)
        if log is None:
            missing.append((tag, path))
            continue
        s = summarize(log)
        delta = s["final"] - base if base else float("nan")
        rows.append((tag, label, s))
        print(f"{label:<12} {s['final']:>9.4f} {delta*100:>+9.2f}pt "
              f"{s['best']:>9.4f} {s['best_round']:>7d} {s['loss']:>8.4f}")

    print("\n" + "=" * 72)
    print("原口径 baseline（供参照，位置/数量与上面不同，不可直接横排）")
    print("=" * 72)
    print(f"{'配置':<22} {'终值acc':>9} {'相对基线':>10}")
    print("-" * 72)
    for name, label in LEGACY:
        log, _ = load(name)
        if log is None:
            continue
        s = summarize(log)
        delta = s["final"] - base if base else float("nan")
        print(f"{label:<22} {s['final']:>9.4f} {delta*100:>+9.2f}pt")

    # 等价性验证：extreme（原口径）与 grad_invert（新口径）应当逐位一致
    print("\n" + "=" * 72)
    print("口径等价性验证")
    print("=" * 72)
    a, _ = load("extreme")
    b, _ = load("grad_invert")
    if a and b:
        same = all(
            abs(x["accuracy"] - y["accuracy"]) < 1e-12 and
            abs(x["loss"] - y["loss"]) < 1e-12
            for x, y in zip(a, b)
        )
        print(f"domain_extreme vs grad_invert 逐轮完全一致: {same}")
        if not same:
            md = max(abs(x["accuracy"] - y["accuracy"]) for x, y in zip(a, b))
            print(f"  最大逐轮 accuracy 偏差: {md:.2e}")
    else:
        print("（grad_invert 尚未跑完，无法验证）")

    if missing:
        print("\n尚未产出：")
        for tag, p in missing:
            print(f"  - {tag}")
        return

    # 单调性检查
    print("\n" + "=" * 72)
    print("单调性检查（按人工排序的破坏程度）")
    print("=" * 72)
    try:
        from scipy import stats
        import numpy as np
        order = [r[2]["final"] for r in rows]
        idx = np.arange(len(order))
        rho, p = stats.spearmanr(idx, order)
        print(f"排列顺序 vs 终值准确率: Spearman rho = {rho:.4f}, p = {p:.4f}")
        r2 = stats.linregress(idx, order)
        print(f"线性拟合: R² = {r2.rvalue**2:.4f}, p = {r2.pvalue:.4f}, slope = {r2.slope:.4f}")
        if rho < -0.7:
            print("→ 破坏程度越高、准确率越低：梯度成立")
        elif rho < -0.3:
            print("→ 弱下降趋势，需补 seed 加强")
        else:
            print("→ 未观察到单调下降：分类不按'程度'排序（可能按'性质'而非'强度'）")
    except ImportError:
        print("（无 scipy，跳过统计）")


if __name__ == "__main__":
    main()
