#!/usr/bin/env bash
# Fresh 8-H20 native-codebook mapper run: global batch 512, 40k updates.
set -euo pipefail
cd "$(dirname "$0")/.."

export MOTAR_ASSETS="${MOTAR_ASSETS:-/root/data/heyuanyu/yefei/lichenge/MoTAR_assets}"
export MOTAR_RESULTS="${MOTAR_RESULTS:-/root/data/heyuanyu/yefei/lichenge/MoTAR_proxy_codebook8_20260917/results}"
export PROXY_ROOT="${PROXY_ROOT:-/root/data/heyuanyu/yefei/lichenge/MoTAR_rgb_proxy_cache_20260916/train}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

python_bin="${PYTHON_BIN:-/root/data/heyuanyu/yefei/lichenge/MoTAR/.venv/bin/python}"
run_name="${RUN_NAME:-proxy_converter_codebook8_40k_b512_seed20260918}"
IFS="," read -r -a visible_gpus <<< "$CUDA_VISIBLE_DEVICES"
if [[ ${#visible_gpus[@]} -ne 8 ]]; then
  echo "expected exactly 8 visible H20 GPUs, got ${#visible_gpus[@]}" >&2
  exit 2
fi

exec "$python_bin" -m torch.distributed.run --standalone --nproc_per_node=8 \
  experiments/feature_to_token_20260916/train_proxy_converter.py \
  --assets-root "$MOTAR_ASSETS" \
  --model codebook \
  --proxy-root "$PROXY_ROOT" \
  --output "$MOTAR_RESULTS/$run_name" \
  --steps 40000 --batch 64 \
  --eval-batch 64 --eval-every 500 --eval-sources 512 \
  --seed 20260918 --eval-seed 20260917 \
  --lr 5e-4 --warmup 500 \
  --repeat-train-rows --save-every 1000 --keep-steps 20000 \
  --log-every 20 --memory-fraction 0.92 --min-free-gib 75 \
  --max-seconds 43200 --wandb-project motar-proxy-converter
