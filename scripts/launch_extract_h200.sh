#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

GPU_LIST="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC="${NPROC_PER_NODE:-8}"
ASSET_ROOT="${ASSET_ROOT:?Set ASSET_ROOT to persistent tokenizer assets storage}"
IMAGENET_TRAIN="${IMAGENET_TRAIN:?Set IMAGENET_TRAIN to the ImageNet train ImageFolder}"
PACKED_CODE_ROOT="${PACKED_CODE_ROOT:?Set PACKED_CODE_ROOT to persistent packed-code output}"
EXTRACT_BATCH_SIZE="${EXTRACT_BATCH_SIZE:-64}"
NUM_WORKERS="${EXTRACT_NUM_WORKERS_PER_RANK:-8}"
MASTER_PORT="${EXTRACT_MASTER_PORT:-29501}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_LIST}"
if [[ "${#GPU_ARRAY[@]}" -ne 8 || "${NPROC}" -ne 8 ]]; then
  echo "Production extraction requires exactly 8 visible GPUs and NPROC_PER_NODE=8." >&2
  exit 2
fi
if [[ ! "${EXTRACT_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "EXTRACT_BATCH_SIZE must be a positive integer." >&2
  exit 3
fi

CUDA_VISIBLE_DEVICES="${GPU_LIST}" python scripts/validate_h200_devices.py --expected-gpus 8

python scripts/prepare_extraction_assets.py --asset-root "${ASSET_ROOT}"

TITOK_ROOT="${ASSET_ROOT}/repos/titok"
LLAMAGEN_ROOT="${ASSET_ROOT}/repos/llamagen"
TITOK_CONFIG="${TITOK_ROOT}/configs/infer/TiTok/titok_l32.yaml"
TITOK_CKPT="${ASSET_ROOT}/checkpoints/titok/tokenizer_titok_l32.bin"
MOT_CKPT="${ASSET_ROOT}/checkpoints/mot/latest.pt"

for path in "${IMAGENET_TRAIN}" "${TITOK_ROOT}" "${LLAMAGEN_ROOT}" "${TITOK_CONFIG}" "${TITOK_CKPT}" "${MOT_CKPT}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Required extraction input not found: ${path}" >&2
    exit 4
  fi
done

RESUME_ARGS=()
if [[ -f "${PACKED_CODE_ROOT}/written.npy" ]]; then
  RESUME_ARGS+=(--resume)
fi

echo "Extracting on 8 GPUs: batch_per_gpu=${EXTRACT_BATCH_SIZE}, precision=fp32, augmentations=original+flip"
CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
torchrun \
  --standalone \
  --nproc_per_node=8 \
  --master_port="${MASTER_PORT}" \
  scripts/extract_codes.py \
  --data-path "${IMAGENET_TRAIN}" \
  --output-root "${PACKED_CODE_ROOT}" \
  --mot-ckpt "${MOT_CKPT}" \
  --mot-state-key model_ema \
  --batch-size "${EXTRACT_BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --mixed-precision none \
  --titok-root "${TITOK_ROOT}" \
  --titok-config "${TITOK_CONFIG}" \
  --titok-ckpt "${TITOK_CKPT}" \
  --llamagen-root "${LLAMAGEN_ROOT}" \
  "${RESUME_ARGS[@]}"
