#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

: "${MATCHED_MICRO_BATCH_SIZE:?Set MATCHED_MICRO_BATCH_SIZE to the disjoint run's selected per-GPU batch}"
if [[ ! "${MATCHED_MICRO_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MATCHED_MICRO_BATCH_SIZE must be a positive integer." >&2
  exit 2
fi

export CONFIG="${CONFIG:-configs/h200_8gpu_150epoch_matched_baseline.yaml}"
export EXP_NAME="${EXP_NAME:-motar_titok_l32_mot199440_matched_baseline_150ep_h200}"
export MICRO_BATCH_SIZE="${MATCHED_MICRO_BATCH_SIZE}"

exec bash "${SCRIPT_DIR}/launch_h200.sh"
