#!/usr/bin/env bash
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
for s in 123 2024; do
  for r in 0 0.125 0.25 0.375 0.5 0.625 0.75; do
    echo "##### seed=$s inv=$r #####"
    "$PY" scripts/run_intervention_metrics_v2.py --invert_ratio $r --seed $s 2>&1 | grep -v Warning | tail -2
  done
done
echo V2_EXTRA_DONE
