#!/usr/bin/env bash
# 统一口径的像素级偏移梯度：一律 5 正常 + 5 变换，位置固定
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
for m in grad_high_contrast grad_posterize grad_edge_enhance grad_grayscale grad_invert; do
  echo "########## $m ##########"
  "$PY" scripts/run_domain_blindspot.py --domain_mode "$m" --num_rounds 200 \
      --save_dir ./results/domain_blindspot 2>&1 | grep -v Warning | tail -4
done
echo "########## three_domain ##########"
"$PY" scripts/run_domain_blindspot.py --domain_mode three_domain --num_rounds 200 \
    --save_dir ./results/domain_blindspot 2>&1 | grep -v Warning | tail -4
echo "ALL_DONE"
