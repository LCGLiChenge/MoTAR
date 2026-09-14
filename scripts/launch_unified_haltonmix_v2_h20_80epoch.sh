#!/usr/bin/env bash
set -euo pipefail

# H20/H200 launcher for UnifiedHaltonMixMaskGIT v2.
# One unified checkpoint generates TiTok-L32 1D tokens and selected sparse
# LlamaGen VQ-16 2D tokens, then decodes by direct replacement.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export USE_TF=0
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

: "${MOTAR_ASSETS:=$ROOT_DIR/h20_local_assets}"
: "${MOTAR_MASKGIT_RESULT_ROOT:=/mnt/data/heyefei/lichenge/Mixture-of-Tokenizer/AR/results/maskgit_optimization_sweep_20260910}"
: "${HALTON_CKPT:=$ROOT_DIR/external_weights/halton_maskgit/ImageNet_256_base.pth}"
: "${NPROC_PER_NODE:=8}"
: "${MICRO_PER_GPU:=224}"
: "${EPOCHS:=80}"
: "${WORKERS:=8}"
: "${EVAL_N:=512}"
: "${FEATURE_CHUNK:=4}"
: "${LR_1D:=0.00001}"
: "${LR_2D:=0.00001}"
: "${LR_FEATURE:=0.0001}"
: "${FREEZE_1D_UPDATES:=50}"
: "${WARMUP_2D_PRETRAINED:=20}"
: "${WANDB_PROJECT:=motar-unified-haltonmix-maskgit}"
: "${WANDB_MODE:=online}"
: "${RUN_NAME:=unified_haltonmix_v2_h20_${NPROC_PER_NODE}gpu_e${EPOCHS}_m${MICRO_PER_GPU}}"
: "${MOTAR_OUTPUT:=$MOTAR_MASKGIT_RESULT_ROOT/$RUN_NAME}"

GLOBAL_BATCH=${GLOBAL_BATCH:-$((MICRO_PER_GPU * NPROC_PER_NODE))}
# Full ImageNet train packed cache has two entries per image: 1,281,167 * 2.
UPDATES_PER_EPOCH=$(((2562334 + GLOBAL_BATCH - 1) / GLOBAL_BATCH))
SAVE_EVERY=${SAVE_EVERY:-$UPDATES_PER_EPOCH}
EVAL_EVERY=${EVAL_EVERY:-$UPDATES_PER_EPOCH}
LOG_EVERY=${LOG_EVERY:-100}

if [[ ! -f "$HALTON_CKPT" ]]; then
  echo "Missing HALTON_CKPT: $HALTON_CKPT" >&2
  echo "Download it with:" >&2
  echo "  hf download llvictorll/Halton-MaskGIT ImageNet_256_base.pth --local-dir external_weights/halton_maskgit" >&2
  exit 2
fi

python -m h20.assets --root "$MOTAR_ASSETS" --profile joint --verify-only
mkdir -p "$MOTAR_MASKGIT_RESULT_ROOT"

echo "Launching $RUN_NAME"
echo "  GPUs:            ${CUDA_VISIBLE_DEVICES:-all visible} / nproc=$NPROC_PER_NODE"
echo "  epochs:          $EPOCHS"
echo "  micro/GPU:       $MICRO_PER_GPU"
echo "  global batch:    $GLOBAL_BATCH"
echo "  updates/epoch:   $UPDATES_PER_EPOCH"
echo "  save/eval every: $SAVE_EVERY / $EVAL_EVERY"
echo "  output:          $MOTAR_OUTPUT"

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" \
  -m experiments.maskgit_optimization_sweep_20260910.unified_fullctx_trial \
  --variant haltonmix_v2 \
  --output "$MOTAR_OUTPUT" \
  --assets-root "$MOTAR_ASSETS" \
  --halton-ckpt "$HALTON_CKPT" \
  --epochs "$EPOCHS" \
  --allow-long \
  --micro "$MICRO_PER_GPU" \
  --global-batch "$GLOBAL_BATCH" \
  --workers "$WORKERS" \
  --eval-initial \
  --eval-every "$EVAL_EVERY" \
  --eval-n "$EVAL_N" \
  --save-every "$SAVE_EVERY" \
  --log-every "$LOG_EVERY" \
  --feature-chunk "$FEATURE_CHUNK" \
  --lr-1d "$LR_1D" \
  --lr-2d "$LR_2D" \
  --lr-feature "$LR_FEATURE" \
  --freeze-1d-updates "$FREEZE_1D_UPDATES" \
  --warmup-2d-pretrained "$WARMUP_2D_PRETRAINED" \
  --attention-implementation sdpa \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-name "$RUN_NAME" \
  --wandb-mode "$WANDB_MODE"
