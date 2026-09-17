#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
ASSETS_ROOT="${MOTAR_ASSETS:-$ROOT/h20_local_assets}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT to the downloaded partial cache directory}"
BATCH="${BATCH:-32}"
MEMORY_FRACTION="${MEMORY_FRACTION:-0.50}"
MIN_FREE_GIB="${MIN_FREE_GIB:-50}"

cd "$ROOT"
COMMON=(--assets-root "$ASSETS_ROOT" --output "$OUTPUT_ROOT" --world-size 8 --batch "$BATCH" --limit 0 --memory-fraction "$MEMORY_FRACTION" --min-free-gib "$MIN_FREE_GIB")
PIDS=()
for RANK in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES="$RANK" "$PYTHON_BIN" -u -m experiments.feature_to_token_20260916.cache_rgb_proxy_tokens extract "${COMMON[@]}" --rank "$RANK" &
  PIDS+=("$!")
done

STATUS=0
for PID in "${PIDS[@]}"; do
  wait "$PID" || STATUS=$?
done
if [[ "$STATUS" -ne 0 ]]; then
  echo "At least one H20 extraction rank failed; inspect status_rank*.json. No automatic retry." >&2
  exit "$STATUS"
fi
"$PYTHON_BIN" -u -m experiments.feature_to_token_20260916.cache_rgb_proxy_tokens finalize "${COMMON[@]}"
