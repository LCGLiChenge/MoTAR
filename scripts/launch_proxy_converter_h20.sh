#!/usr/bin/env bash
# One-H20 bounded pilot for f_1d -> RGB-reencoded dense proxy-token mapping.
set -euo pipefail
cd "$(dirname "$0")/.."
export MOTAR_ASSETS="${MOTAR_ASSETS:-/root/data/heyuanyu/yefei/lichenge/MoTAR_assets}"
export MOTAR_RESULTS="${MOTAR_RESULTS:-/root/data/heyuanyu/yefei/lichenge/MoTAR_halton_proxy_20260916/results}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
python_bin="${PYTHON_BIN:-/root/data/heyuanyu/yefei/lichenge/MoTAR/.venv/bin/python}"
run_name="${RUN_NAME:-proxy_converter_1k_seed20260917}"
exec "$python_bin" experiments/feature_to_token_20260916/train_proxy_converter.py \
  --assets-root "$MOTAR_ASSETS" \
  --model "${MODEL:-full}" --rank "${MAPPER_RANK:-24}" \
  --proxy-root "${PROXY_ROOT:-/root/data/heyuanyu/yefei/lichenge/MoTAR_rgb_proxy_cache_20260916/train}" \
  --output "$MOTAR_RESULTS/$run_name" \
  --steps "${STEPS:-1000}" --batch "${BATCH:-768}" \
  --eval-batch "${EVAL_BATCH:-128}" --eval-every "${EVAL_EVERY:-200}" \
  --eval-sources "${EVAL_SOURCES:-512}" --lr "${LR:-5e-4}" \
  --max-seconds "${MAX_SECONDS:-3600}"
