# Metric–Scenario Coupling in Federated Learning

Code and result artefacts for the manuscript

> **When Metrics Meet Scenarios: Metric–Scenario Coupling in Federated Learning under Combined Label and Feature Skew**
> Zhong Yusen, Institute of Digital Technology and Applications, Guangzhou Huashang College

## What this repository contains

The manuscript asks a narrow question: if a distribution-heterogeneity measure predicts
accuracy well under one kind of skew, does it still predict it under another?

To answer it, each client partition is represented as a joint distribution `P(y, c)` over
the class label `y` and a discretised feature cluster `c`. Eight divergence and coverage
measures are then computed on that representation, and each is correlated with the final
accuracy of a full federated run. The grid covers label skew, feature skew and their
combination on CIFAR-10, FMNIST and CIFAR-100, plus a four-domain panel on PACS.

Three things are in here that are not usually in a repository of this kind, and they are
deliberate:

* the **stored result artefacts** for every run reported in the paper, so that every figure
  and table can be regenerated without retraining;
* the **intervention code** that separates a *measured* divergence from a *constructed* one
  (`run_intervention.py` versus `run_intervention_metrics_v2.py`), because that distinction
  is what Section 5.3 of the paper turns on;
* a statement of the **reproducibility floor** (see below), so that nobody mistakes
  run-to-run variation for an effect.

## Install

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10 or newer. A CUDA device is not required but is strongly recommended: one
200-round run with 10 clients takes roughly half an hour on an RTX 5090 and considerably
longer on CPU.

## Layout

```
src/                       simulation library (partitioning, model, features, metrics)
  data/partition.py          Dirichlet partitions for label and feature skew
  data/datasets.py           dataset loaders
  models/cnn.py              the small CNN used for CIFAR-10 and FMNIST
  models/feature_extractor.py  ImageNet-pretrained ResNet-18, 512-d penultimate features
  metrics/coverage.py        the eight measures; single implementation, no duplicates
  fl/client.py               local training and FedAvg aggregation
scripts/                   experiment runners and analyses
  run_*.py                   one script per experiment block
  analyze_*.py               statistics over the stored artefacts
  make_figures.py            Figures 1, 1b and 2-6
  make_graphical_abstract.py a two-panel summary figure (optional; not used in the submitted manuscript)
  make_table2.py             Table 2, printed as markdown for diffing
  exploratory/               one-off diagnostics kept for the record, not on the paper path
results/                   stored artefacts (JSON). Every number in the paper comes from here.
```

## Data

CIFAR-10, CIFAR-100 and Fashion-MNIST are downloaded automatically by torchvision into
`./data` on first use.

PACS is not downloaded automatically. Obtain the four domains (art painting, cartoon,
photo, sketch) and either lay them out as `data/PACS/<domain>/<class>/*.jpg`, or convert a
parquet shard with

```bash
python scripts/extract_pacs.py --parquet /path/to/train-00000-of-00001.parquet --out ./data/PACS
```

## Reproducing the paper

See [REPRODUCE.md](REPRODUCE.md) for the command behind each figure and table, and for what
each artefact contains.

Short version, from a clean checkout with `results/` intact:

```bash
python scripts/make_figures.py            # Figures 1, 1b, 2, 3, 4, 5, 6
python scripts/make_graphical_abstract.py # optional summary figure
python scripts/make_table2.py             # Table 2 as markdown
```

## On reproducibility

The training pipeline is **not** bit-reproducible: three repeats of one identical
configuration (same partition, same seed) finish 0.70 accuracy points apart, which comes
from non-deterministic cuDNN/cuBLAS kernels rather than from the code. The paper measures
this floor and interprets no difference below it. If you rerun a configuration and get a
number a few tenths of a point away from the stored one, that is expected and is not a bug.

## Figure numbering

The file names under `results/figures/` follow the numbering used in the manuscript
(`fig5_shift_nature.png` is Figure 5, `fig6_stability.png` is Figure 6). The plots carry no
figure number inside the artwork, so a renumbering never leaves a stale label in the image.

## License

MIT — see [LICENSE](LICENSE).

## Contact

Zhong Yusen — 1035427530@qq.com
