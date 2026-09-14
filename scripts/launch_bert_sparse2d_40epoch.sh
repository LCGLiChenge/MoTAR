#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${CUDA_VISIBLE_DEVICES:?Set explicitly allocated GPUs, e.g. 0,1,2,3,4,5,6,7}"
: "${MOTAR_ASSETS:?Set the downloaded asset directory outside this repository}"
: "${MOTAR_RESULTS:?Set the output directory on your data disk}"
export USE_TF=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
python -m bert2d.launch \
  --output "${OUTPUT:-$MOTAR_RESULTS/bert_sparse2d_40epoch}" \
  --assets-root "$MOTAR_ASSETS" \
  --epochs "${EPOCHS:-40}" \
  --eval-every "${EVAL_EVERY:-2}" \
  --eval-batch "${EVAL_BATCH:-8}" \
  --global-batch "${GLOBAL_BATCH:-448}" \
  --micro "${MICRO:-0}" \
  --recompute-layers "${RECOMPUTE_LAYERS:-10}" \
  --wandb-project "${WANDB_PROJECT:-motar-bert-sparse2d}" \
  --wandb-mode "${WANDB_MODE:-online}" "$@"
