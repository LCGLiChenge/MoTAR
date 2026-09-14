#!/usr/bin/env bash
set -euo pipefail

# H20/H200 launcher for selected sparse 2D-only MaskGIT.
# Generates only E117/Router-selected LlamaGen VQ-16 tokens.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/delivery_h20_20260910:$ROOT_DIR:${PYTHONPATH:-}"
export USE_TF=0

: "${MOTAR_ASSETS:=$ROOT_DIR/h20_local_assets}"
: "${MOTAR_MASKGIT_RESULT_ROOT:=/mnt/data/heyefei/lichenge/Mixture-of-Tokenizer/AR/results/maskgit_optimization_sweep_20260910}"
: "${HALTON_CKPT:=$ROOT_DIR/external_weights/halton_maskgit/ImageNet_256_base.pth}"
: "${NPROC_PER_NODE:=8}"
: "${MICRO_PER_GPU:=320}"
: "${UPDATES:=50000}"
: "${WARMUP:=1000}"
: "${WORKERS:=8}"
: "${LR_FEATURE:=0.0001}"
: "${LR_PRETRAINED:=0.00001}"
: "${EVAL_N:=512}"
: "${FEATURE_CHUNK:=4}"
: "${WANDB_PROJECT:=motar-selected2d-maskgit}"
: "${WANDB_MODE:=online}"
: "${RUN_NAME:=halton_fullctx_sparse2d_h20_8gpu_u${UPDATES}_m${MICRO_PER_GPU}}"
: "${MOTAR_OUTPUT:=$MOTAR_MASKGIT_RESULT_ROOT/$RUN_NAME}"

GLOBAL_BATCH=${GLOBAL_BATCH:-$((MICRO_PER_GPU * NPROC_PER_NODE))}
# Full ImageNet train packed cache has two augmentations: 1,281,167 * 2 samples.
SAVE_EVERY=${SAVE_EVERY:-$(((2562334 + GLOBAL_BATCH - 1) / GLOBAL_BATCH))}
EVAL_EVERY=${EVAL_EVERY:-$SAVE_EVERY}

if [[ ! -f "$HALTON_CKPT" ]]; then
  echo "Missing HALTON_CKPT: $HALTON_CKPT" >&2
  echo "Download it with:" >&2
  echo "  hf download llvictorll/Halton-MaskGIT ImageNet_256_base.pth --local-dir external_weights/halton_maskgit" >&2
  exit 2
fi

python -m h20.assets --root "$MOTAR_ASSETS" --profile joint --verify-only
mkdir -p "$MOTAR_MASKGIT_RESULT_ROOT"

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" \
  -m experiments.maskgit_optimization_sweep_20260910.halton_sparse2d_trial \
  --output "$MOTAR_OUTPUT" \
  --halton-ckpt "$HALTON_CKPT" \
  --context-style fullctx \
  --updates "$UPDATES" \
  --warmup "$WARMUP" \
  --allow-long \
  --micro "$MICRO_PER_GPU" \
  --global-batch "$GLOBAL_BATCH" \
  --workers "$WORKERS" \
  --eval-every "$EVAL_EVERY" \
  --eval-n "$EVAL_N" \
  --save-every "$SAVE_EVERY" \
  --feature-chunk "$FEATURE_CHUNK" \
  --lr-feature "$LR_FEATURE" \
  --lr-pretrained "$LR_PRETRAINED" \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-name "$RUN_NAME" \
  --wandb-mode "$WANDB_MODE"
