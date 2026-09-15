#!/usr/bin/env bash
# One packed epoch + paired full-refine 5k FID, then exit. No automatic long run.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${CUDA_VISIBLE_DEVICES:?Set exactly the GPUs allocated to this comparison}"
: "${MOTAR_ASSETS:?Set the downloaded asset directory}"
: "${MOTAR_RESULTS:?Set the data-disk output root}"
export GLOBAL_BATCH="${GLOBAL_BATCH:-3200}"
export BATCH_SCALING="${BATCH_SCALING:-adamw-sde}"
export OUTPUT="${OUTPUT:-$MOTAR_RESULTS/bert_h20_scaling_b${GLOBAL_BATCH}_${BATCH_SCALING}_seed0}"
export MICRO="${MICRO:-0}"
export EVAL_EVERY=1
# Treat this as a bounded test entry point; fixed target, no arbitrary CLI overrides.
if (( $# )); then
  echo 'This bounded test script accepts environment configuration, not CLI overrides.' >&2
  exit 2
fi
bash scripts/launch_bert_sparse2d_40epoch.sh --epochs 40 --test-epochs 1 --eval-every 1
