#!/usr/bin/env bash
# §5.2 k 敏感性的多 seed 补跑：12 个 combined 配置 × seed 123 / 2024
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
PAIRS="0.05:0.05 0.05:0.1 0.05:0.3 0.05:1.0 0.1:0.05 0.1:0.3 0.3:0.05 0.3:0.3 0.5:0.05 0.5:0.3 1.0:0.05 1.0:0.3"
for s in 123 2024; do
  for p in $PAIRS; do
    aL=${p%%:*}; aF=${p##*:}
    echo "##### seed=$s aL=$aL aF=$aF #####"
    "$PY" scripts/run_k_selection.py --seed $s --alpha_label $aL --alpha_feature $aF 2>&1 | grep -v Warning | tail -2
  done
done
echo K_SEEDS_DONE
