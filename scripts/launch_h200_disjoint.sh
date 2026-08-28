#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-configs/h200_8gpu_150epoch_disjoint.yaml}"
GPU_LIST="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC="${NPROC_PER_NODE:-8}"
EXP_NAME="${EXP_NAME:-motar_titok_l32_mot199440_disjoint_150ep_h200}"
PACKED_CODE_ROOT="${PACKED_CODE_ROOT:?Set PACKED_CODE_ROOT to the completed packed-code directory}"
RESULTS_DIR="${RESULTS_DIR:?Set RESULTS_DIR to persistent storage}"
WANDB_PROJECT="${WANDB_PROJECT:-motar-unified-ar}"
NUM_WORKERS="${NUM_WORKERS_PER_RANK:-8}"
LOG_EVERY="${LOG_EVERY:-20}"
MEMORY_FRACTION="${H200_MEMORY_FRACTION:-0.90}"
CANDIDATES="${H200_BATCH_CANDIDATES:-768,704,640,576,512,448,384,320,256}"
MASTER_PORT="${MASTER_PORT:-29500}"
ALLOW_NON_H200="${ALLOW_NON_H200:-0}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_LIST}"
if [[ "${#GPU_ARRAY[@]}" -ne 8 || "${NPROC}" -ne 8 ]]; then
  echo "This launch requires exactly 8 visible GPUs and NPROC_PER_NODE=8." >&2
  exit 2
fi

VALIDATE_ARGS=(
  --packed-code-root "${PACKED_CODE_ROOT}"
  --expected-gpus 8
)
if [[ "${ALLOW_NON_H200}" == "1" ]]; then
  VALIDATE_ARGS+=(--allow-non-h200)
fi

CHECKPOINT_ARGS=()
LATEST_DIR="${RESULTS_DIR}/${EXP_NAME}/checkpoints/latest"
if [[ -d "${LATEST_DIR}" ]]; then
  CHECKPOINT_ARGS+=(--resume auto)
  VALIDATE_ARGS+=(--checkpoint "${LATEST_DIR}")
fi

CUDA_VISIBLE_DEVICES="${GPU_LIST}" python scripts/validate_disjoint_setup.py "${VALIDATE_ARGS[@]}"

BATCH_CACHE="${RESULTS_DIR}/${EXP_NAME}/h200_micro_batch_size.txt"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-auto}"
if [[ "${MICRO_BATCH_SIZE}" == "auto" && -f "${BATCH_CACHE}" ]]; then
  MICRO_BATCH_SIZE="$(<"${BATCH_CACHE}")"
fi
if [[ "${MICRO_BATCH_SIZE}" == "auto" ]]; then
  PROBE_GPU="${GPU_ARRAY[0]}"
  SELECT_ARGS=(
    --config "${CONFIG}"
    --candidates "${CANDIDATES}"
    --memory-fraction "${MEMORY_FRACTION}"
  )
  if [[ "${ALLOW_NON_H200}" == "1" ]]; then
    SELECT_ARGS+=(--allow-non-h200)
  fi
  MICRO_BATCH_SIZE="$(
    CUDA_VISIBLE_DEVICES="${PROBE_GPU}" \
      python scripts/select_h200_disjoint_batch.py "${SELECT_ARGS[@]}"
  )"
fi
if [[ ! "${MICRO_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid selected MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE@Q}" >&2
  exit 3
fi

mkdir -p "${RESULTS_DIR}/${EXP_NAME}"
printf '%s\n' "${MICRO_BATCH_SIZE}" > "${BATCH_CACHE}"
echo "Launching 8xH200 disjoint sidecar: micro_batch_per_gpu=${MICRO_BATCH_SIZE}, global_batch=$((MICRO_BATCH_SIZE * 8))"
echo "Checkpoint policy: update checkpoints/latest after every epoch; no versioned checkpoints."

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export WANDB_PROJECT
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"

CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
torchrun \
  --standalone \
  --nproc_per_node=8 \
  --master_port="${MASTER_PORT}" \
  train_disjoint.py \
  --config "${CONFIG}" \
  --exp-name "${EXP_NAME}" \
  --packed-code-root "${PACKED_CODE_ROOT}" \
  --results-dir "${RESULTS_DIR}" \
  --micro-batch-size "${MICRO_BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --log-every "${LOG_EVERY}" \
  "${CHECKPOINT_ARGS[@]}"
