# Reproducing the figures and tables

All commands are run from the repository root. Every figure is drawn from files in
`results/`; none of them retrains anything. Regenerating the whole set takes a few seconds.

## Figures and tables

| Item | Command | Reads |
|---|---|---|
| Fig. 1 (coupling matrix) | `python scripts/make_figures.py` | `metric_comparison/`, `k_selection/`, `cross_dataset/`, `cifar100/`, `k_selection/js_conditional_recompute.json` |
| Fig. 1b (resolution sensitivity, 3 seeds) | same | `k_selection/aL*_seed{42,123,2024}.json` |
| Fig. 2 (intervention) | same | `intervention/`, `intervention_v2/` |
| Fig. 3 (pooled regression) | same | `metric_comparison/`, `k_selection/` |
| Fig. 4 (cross-dataset) | same | `cross_dataset/`, `cifar100/`, `metric_comparison/`, `k_selection/` |
| Fig. 5 (nature of the shift) → `fig5_shift_nature.png` | same | `domain_blindspot/`, `_recheck/`, `pacs_v3/` |
| Fig. 6 (stability, two panels) → `fig6_stability.png` | same | `stress/` |
| Optional summary figure (not used in the submitted manuscript) | `python scripts/make_graphical_abstract.py` | `metric_comparison/`, `k_selection/`, `intervention/`, `intervention_v2/` |
| Table 2 | `python scripts/make_table2.py` | `metric_comparison/`, `k_selection/aL*_seed42.json`, `cross_dataset/`, `cifar100/` |

`make_table2.py` prints the table as markdown so it can be diffed against the manuscript
rather than eyeballed. It reproduces all 36 rows and every accuracy value exactly.

Figures are written to `results/figures/` at 300 dpi. The graphical abstract is written at
the print size Elsevier asks for (13.28 x 5.31 cm, 2656 x 1062 px); adjust `dpi` in the
script if a different size is wanted, not `figsize`, or the font sizes will drift.

## Artefact directories

| Directory | Contents |
|---|---|
| `metric_comparison/` | CIFAR-10, 4 label-skew and 4 feature-skew runs |
| `k_selection/` | CIFAR-10 combined runs: 12 configurations x 3 seeds, each carrying measures at four resolutions (k = 4, 10, 20, 50), plus `js_conditional_recompute.json` |
| `cross_dataset/` | FMNIST, 8 runs |
| `cifar100/` | CIFAR-100, 8 runs |
| `intervention/` | the dose-response runs: original divergence construction (v1) |
| `intervention_v2/` | the same runs with the divergence recomputed from features extracted *after* the transformation and re-clustered (v2) |
| `domain_blindspot/` | pixel-level pre-transformations, one run per transformation |
| `_recheck/` | the untransformed control and the two legacy configurations, rerun under current code |
| `pacs_v3/` | PACS, held-out per-domain split, 4 clients |
| `stress/` | four-arm aggregation stress test, 10 seeds per arm |
| `cw_fedavg/` | coverage-weighted FedAvg runs |
| `figures/` | rendered output |

## Retraining from scratch

The stored artefacts make retraining unnecessary, but each block can be regenerated.
Wall-clock times are for one RTX 5090 D.

```bash
# scenario grid (20 CIFAR-10 runs)
python scripts/run_metric_comparison.py --dataset cifar10 --num_rounds 200 --seed 42
# k-sensitivity, one configuration at a time (12 configs x 3 seeds per resolution set)
python scripts/run_k_selection.py --seed 42 --alpha_label 0.05 --alpha_feature 0.05
# dose-response: the original construction, then the measured recomputation
python scripts/run_intervention.py --invert_ratio 0.5 --seed 42
python scripts/run_intervention_metrics_v2.py --invert_ratio 0.5 --seed 42
# pixel-level transformations (five arms, then the two legacy configurations)
bash scripts/run_gradient_all.sh
# FMNIST / CIFAR-100
python scripts/run_cross_dataset.py --dataset fmnist --alpha_label 0.05 --alpha_feature 0.05
# PACS, held-out split
python scripts/run_pacs_v3.py --domain_mode four_domain --alpha 0.5 --num_clients 4
# aggregation stress test
python scripts/run_cw_fedavg.py --alpha_label 0.1 --alpha_feature 0.1 --weight_mode cw
```

Each run writes one JSON file into `results/<block>/`. The analyses in `analyze_*.py`
read those files and print the statistics quoted in the paper.

## Two things worth knowing before you diff numbers

1. **The pipeline is not bit-reproducible.** Three repeats of one configuration span
   0.70 accuracy points (Section 5.6 of the paper). Differences of that size between your
   run and the stored one carry no information.
2. **`intervention/` and `intervention_v2/` are different measurements of the same runs.**
   The accuracy in both is identical, because it comes from the same training. The
   divergence differs: v1 relocated the shifted clients on the joint grid, v2 re-extracts
   features after the transformation and re-clusters. Section 5.3 reports both and explains
   why only v2 is a measurement.
