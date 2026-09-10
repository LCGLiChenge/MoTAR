#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${CUDA_VISIBLE_DEVICES:?Set exactly the GPU IDs you are allowed to use}"
: "${MOTAR_ASSETS:?Set the downloaded H20 asset root}"
: "${MOTAR_OUTPUT:?Set a fresh persistent training directory}"
export USE_TF=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
exec python -m h20.launch --assets "$MOTAR_ASSETS" --output "$MOTAR_OUTPUT" "$@"
