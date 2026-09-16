#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${MOTAR_ASSETS:?Set MOTAR_ASSETS to the existing data-disk assets directory}"
: "${MOTAR_RESULTS:?Set MOTAR_RESULTS to a new data-disk experiment directory}"
: "${CUDA_VISIBLE_DEVICES:?Explicit allocation required: 0,1,2,3,4,5,6,7}"
if [[ "$CUDA_VISIBLE_DEVICES" != '0,1,2,3,4,5,6,7' ]]; then
  echo 'This bounded launcher requires explicit allocation of H20 physical GPUs0..7.' >&2
  exit 2
fi
export USE_TF=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HOME="${HF_HOME:-$MOTAR_RESULTS/.hf}"
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0
PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" experiments/feature_to_token_20260916/test_grid_h20.py
"$PYTHON_BIN" -m bert2d.assets --root "$MOTAR_ASSETS" --fid --verify-only
"$PYTHON_BIN" experiments/feature_to_token_20260916/download_h20_checkpoint.py
exec "$PYTHON_BIN" -u experiments/feature_to_token_20260916/h20_grid_pipeline.py \
  --resume "$MOTAR_RESULTS/input6000" --output "$MOTAR_RESULTS/run_seed0_g448"
