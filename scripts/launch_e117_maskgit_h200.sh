#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-configs/e117_maskgit_h200_8gpu.yaml}"
GPU_LIST="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC="${NPROC_PER_NODE:-8}"
EXP_NAME="${EXP_NAME:-e117-maskgit-h200-8gpu-seed0}"
PACKED_CODE_ROOT="${PACKED_CODE_ROOT:?Set PACKED_CODE_ROOT to the completed train packed-code directory}"
E117_ROUTE_CACHE="${E117_ROUTE_CACHE:?Set E117_ROUTE_CACHE to the completed train route-cache directory}"
EVAL_PACKED_CODE_ROOT="${EVAL_PACKED_CODE_ROOT:?Set EVAL_PACKED_CODE_ROOT to the completed val packed-code directory}"
E117_EVAL_ROUTE_CACHE="${E117_EVAL_ROUTE_CACHE:?Set E117_EVAL_ROUTE_CACHE to the completed val route-cache directory}"
RESULTS_DIR="${RESULTS_DIR:?Set RESULTS_DIR to persistent storage}"
WANDB_PROJECT="${WANDB_PROJECT:-motar-maskgit}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"
NUM_WORKERS="${NUM_WORKERS_PER_RANK:-8}"
MEMORY_FRACTION="${H200_MEMORY_FRACTION:-0.90}"
CANDIDATES="${H200_BATCH_CANDIDATES:-768,704,640,576,512,448,416,384,352,320,288,256,240,224,208,192,176,160,144,128,112,96,80,64,48,32,24,16,8}"
MASTER_PORT="${MASTER_PORT:-29500}"
ALLOW_NON_H200="${ALLOW_NON_H200:-0}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_LIST}"
if [[ "${#GPU_ARRAY[@]}" -ne 8 || "${NPROC}" -ne 8 ]]; then
  echo "Formal launch requires exactly 8 visible GPUs and NPROC_PER_NODE=8." >&2
  exit 2
fi

VALIDATE_ARGS=(
  --packed-code-root "${PACKED_CODE_ROOT}"
  --route-cache "${E117_ROUTE_CACHE}"
  --eval-packed-code-root "${EVAL_PACKED_CODE_ROOT}"
  --eval-route-cache "${E117_EVAL_ROUTE_CACHE}"
  --expected-gpus 8
)
PROBE_EXTRA=()
if [[ "${ALLOW_NON_H200}" == "1" ]]; then
  VALIDATE_ARGS+=(--allow-non-h200)
  PROBE_EXTRA+=(--allow-non-h200)
fi

python scripts/validate_wandb_auth.py --mode "${WANDB_MODE}"
CUDA_VISIBLE_DEVICES="${GPU_LIST}" python scripts/validate_e117_maskgit_setup.py "${VALIDATE_ARGS[@]}"

OUTPUT_DIR="${RESULTS_DIR}/${EXP_NAME}"
BATCH_CACHE="${OUTPUT_DIR}/h200_micro_batch_size.txt"
RESOLVED_CONFIG="${OUTPUT_DIR}/resolved_config.yaml"
mkdir -p "${OUTPUT_DIR}"

MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-auto}"
if [[ -f "${OUTPUT_DIR}/latest.pt" ]]; then
  if [[ ! -f "${BATCH_CACHE}" || ! -f "${RESOLVED_CONFIG}" ]]; then
    echo "latest.pt exists without the immutable batch/config files; refusing resume." >&2
    exit 3
  fi
  MICRO_BATCH_SIZE="$(<"${BATCH_CACHE}")"
elif [[ "${MICRO_BATCH_SIZE}" == "auto" ]]; then
  PROBE_GPU="${GPU_ARRAY[0]}"
  MICRO_BATCH_SIZE="$(CUDA_VISIBLE_DEVICES="${PROBE_GPU}" python scripts/select_e117_maskgit_h200_batch.py --config "${CONFIG_TEMPLATE}" --candidates "${CANDIDATES}" --memory-fraction "${MEMORY_FRACTION}" "${PROBE_EXTRA[@]}")"
else
  PROBE_GPU="${GPU_ARRAY[0]}"
  CUDA_VISIBLE_DEVICES="${PROBE_GPU}" python scripts/probe_e117_maskgit_batch_size.py --config "${CONFIG_TEMPLATE}" --candidate "${MICRO_BATCH_SIZE}" --world-size 8 --memory-fraction "${MEMORY_FRACTION}" "${PROBE_EXTRA[@]}"
fi
if [[ ! "${MICRO_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid selected MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE@Q}" >&2
  exit 4
fi
printf '%s\n' "${MICRO_BATCH_SIZE}" > "${BATCH_CACHE}"

PREPARE_ARGS=(
  --template "${CONFIG_TEMPLATE}"
  --output-dir "${OUTPUT_DIR}"
  --experiment-name "${EXP_NAME}"
  --packed-code-root "${PACKED_CODE_ROOT}"
  --route-cache "${E117_ROUTE_CACHE}"
  --eval-packed-code-root "${EVAL_PACKED_CODE_ROOT}"
  --eval-route-cache "${E117_EVAL_ROUTE_CACHE}"
  --micro-batch-size "${MICRO_BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --wandb-project "${WANDB_PROJECT}"
)
if [[ -n "${WANDB_ENTITY}" ]]; then
  PREPARE_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi
python scripts/prepare_e117_maskgit_run.py "${PREPARE_ARGS[@]}"

if [[ -f "${OUTPUT_DIR}/latest.pt" ]]; then
  python scripts/validate_e117_maskgit_latest.py "${OUTPUT_DIR}" --expected-world-size 8
  echo "Resuming only this H200 run's native latest.pt."
else
  echo "Starting the H200 MaskGIT model from random initialization."
fi
echo "micro_batch_per_gpu=${MICRO_BATCH_SIZE}; global_batch=$((MICRO_BATCH_SIZE * 8))"
echo "W&B project=${WANDB_PROJECT}; mode=${WANDB_MODE}; no train.log is created."
echo "Checkpoint policy: atomically replace latest.pt after every completed data epoch."

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export WANDB_PROJECT WANDB_ENTITY WANDB_MODE
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"

CUDA_VISIBLE_DEVICES="${GPU_LIST}" accelerate launch --multi_gpu --num_machines 1 --num_processes 8 --mixed_precision bf16 --main_process_port "${MASTER_PORT}" train_e117_sparse_maskgit.py --config "${RESOLVED_CONFIG}" --output-dir "${OUTPUT_DIR}"
