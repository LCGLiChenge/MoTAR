# MoTAR: E117-routed unified MaskGIT

This repository contains the formal scratch-training handoff for one shared
bidirectional generator:

- stage 1: class -> 32 TiTok-L32 1D codes (vocabulary 4096);
- frozen E117: completed 1D codes -> a deterministic K=64 or K=128 route;
- stage 2: class + completed 1D codes + routed coordinates -> sparse
  MoT/LlamaGen VQ-16 2D codes (vocabulary 16384).

The stages share a 24-layer, width-768 pre-norm LLaMA transformer and have
separate input/output vocabularies. Training uses TiTok's arccos mask schedule,
0.1 label smoothing, 0.1 visible-token loss weight, class dropout, AdamW
(betas 0.9/0.96, weight decay 0.03), full-model EMA, 1.5x 1D loss, and 1.0x 2D
loss. Attention is explicitly bidirectional: this is MaskGIT, not causal AR.

The H200 run starts the generator from random initialization. Do not copy the
RTX 5090 generator checkpoint. A checkpoint later created by this same H200 run
may be resumed automatically.

The previous 150-epoch causal-AR handoff remains in
[docs/LEGACY_CAUSAL_AR.md](docs/LEGACY_CAUSAL_AR.md). It is not the launch
command for this experiment.

## Current evidence

The identical model/configuration was run from scratch on four RTX 5090s with
global batch 128:

| step | 1D CE | 2D CE | global 1D context delta | global 1D label delta | 2D context delta | 2D 1D-prefix delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 8.471 | 9.857 | baseline | baseline | baseline | baseline |
| 30k | 8.151 | 9.488 | 0.00149 | 0.00031 | 0.20173 | 0.01943 |
| 40k | 8.104 | 9.318 | 0.01003 | 0.00197 | 0.28661 | 0.11895 |

At 40k every preregistered feasibility check passed for the first time. This
shows that both branches and cross-stage conditioning learn; it is not a
generation-FID claim. Final claims still require multi-seed 50k-sample eval.

## Required assets

Training reads only discrete codes and E117 routes. Tokenizer and E117 weights
are not loaded into the training process.

| asset | size here | clean-server status |
| --- | ---: | --- |
| ImageNet train packed codes, 2 augmentations | about 1.4 GiB | reproducible from ImageNet + Hugging Face |
| ImageNet val packed codes, 1 augmentation | about 28 MiB | reproducible from ImageNet + Hugging Face |
| full-train E117 route cache | 156 MiB allocated, about 328 MiB logical | **not yet on Hugging Face; must be transferred** |
| validation E117 route cache | 3.2 MiB allocated, about 6.8 MiB logical | **not yet on Hugging Face; must be transferred** |
| frozen E117 checkpoint | 138,821,223 bytes | not needed with caches; not yet on Hugging Face |

Registered E117 checkpoint SHA256:

```text
a5b84689d2b29f579d2442da7594ac093292b6386760867a0668ca02f82e6156
```

Route metadata embeds this hash, and validation rejects another checkpoint.

Packed-code extraction downloads and verifies:

- `sophiaa/MoT-1-checkpoints/latest.pt`, revision
  `0ed66fb6f5f3edc79205fab87c39139772caab4d`, 6,400,628,829 bytes,
  SHA256 `86c8f9da5e61261ab93066c73d7719203e8c00b69f05b805c5937e6b7319b446`;
- `fun-research/TiTok/tokenizer_titok_l32.bin`, revision
  `ab646ed225080a3acb7c78440a574d7f67f16fa7`, 2,564,477,610 bytes,
  SHA256 `b8f0bf61e9ee1791d8b76fa723bdcb2c85a039a7d027e597f685db492935c31f`;
- pinned TiTok and LlamaGen source commits.

The MoT checkpoint has all adapted LlamaGen VQ parameters, but not the TiTok
encoder/codebook. TiTok's tokenizer file is therefore needed for extraction.
The original `vq_ds16_c2i.pt` is not needed.

## Clean H200 setup

Use an H200-compatible PyTorch installation:

```bash
git clone https://github.com/LCGLiChenge/MoTAR.git
cd MoTAR
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
wandb login
```

### Regenerate packed codes

The target host needs ImageNet in ImageFolder layout:

```bash
export ASSET_ROOT=/persistent/assets/motar-extraction
export IMAGENET_TRAIN=/persistent/datasets/imagenet/train
export IMAGENET_VAL=/persistent/datasets/imagenet/val
export PACKED_CODE_ROOT=/persistent/data/imagenet-titok_l32-mot199440ema-adm-256_packed
export EVAL_PACKED_CODE_ROOT=/persistent/data/imagenet-val-titok_l32-mot199440ema-none-256_packed

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/launch_extract_h200.sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/launch_extract_val_h200.sh
```

Both are resumable through `written.npy`. Extraction runs tokenizer
inference in FP32 and is separate from generator batch probing.

Alternatively copy packed caches, then verify:

```bash
(cd "${PACKED_CODE_ROOT}" && sha256sum -c /path/to/MoTAR/docs/mot199440_packed_manifest.sha256)
(cd "${EVAL_PACKED_CODE_ROOT}" && sha256sum -c /path/to/MoTAR/docs/mot199440_val_packed_manifest.sha256)
```

### Transfer the E117 routes

Until these caches are on Hugging Face, copy these exact directories:

```text
/mnt/data/heyefei/lichenge/Mixture-of-Tokenizer/AR/artifacts/e117_routes_full_train_e116
/mnt/data/heyefei/lichenge/Mixture-of-Tokenizer/AR/artifacts/e117_routes_imagenet_val_e116
```

On H200:

```bash
export E117_ROUTE_CACHE=/persistent/data/e117_routes_full_train_e116
export E117_EVAL_ROUTE_CACHE=/persistent/data/e117_routes_imagenet_val_e116

(cd "${E117_ROUTE_CACHE}" && sha256sum -c /path/to/MoTAR/docs/e117_routes_train_manifest.sha256)
(cd "${E117_EVAL_ROUTE_CACHE}" && sha256sum -c /path/to/MoTAR/docs/e117_routes_val_manifest.sha256)
```

## Start the 8-H200 formal run

```bash
export PACKED_CODE_ROOT=/persistent/data/imagenet-titok_l32-mot199440ema-adm-256_packed
export EVAL_PACKED_CODE_ROOT=/persistent/data/imagenet-val-titok_l32-mot199440ema-none-256_packed
export E117_ROUTE_CACHE=/persistent/data/e117_routes_full_train_e116
export E117_EVAL_ROUTE_CACHE=/persistent/data/e117_routes_imagenet_val_e116
export RESULTS_DIR=/persistent/results/motar
export WANDB_PROJECT=motar-maskgit
export WANDB_ENTITY=your-wandb-entity

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash scripts/launch_e117_maskgit_h200.sh
```

The launcher requires exactly eight H200s with bf16 support, both complete
packed datasets, both exact route caches, and W&B authentication. Set
`WANDB_MODE=offline` only deliberately. No `train.log` or
TensorBoard file is created.

## H200 memory policy

The batch is not guessed. One H200 first runs the exact worst-case K=128 bf16
workload: full 1D and sparse 2D forwards, combined backward, gradient clipping,
fused AdamW, and resident fp32 EMA. Candidates from 768 down to 8 per GPU run
in isolated processes. The largest below 90% peak reserved memory is saved in:

```text
${RESULTS_DIR}/${EXP_NAME}/h200_micro_batch_size.txt
```

This leaves DDP/NCCL margin. With co-tenancy:

```bash
H200_MEMORY_FRACTION=0.85 bash scripts/launch_e117_maskgit_h200.sh
```

A manual `MICRO_BATCH_SIZE` is still subjected to the real probe.

## Budget, W&B, and checkpoints

The formal budget matches TiTok by image exposure:

```text
500,000 reference steps * global batch 2,048 = 1,024,000,000 examples
```

After selecting batch, the launcher rounds `max_steps` to the nearest
optimizer step, keeps warmup at 2%, and records the at-most-half-step exposure
error in `run_plan.json`. LR stays `1e-4` rather than being silently
batch-scaled. This is about 799 ImageNet-source-equivalent epochs. A completed
data epoch is one traversal of the two-augmentation route cache. The old
150-epoch target applies only to causal AR.

W&B records total/1D/2D loss, top-1/top-5, mask ratios, gradient norm, LR,
throughput, source-equivalent epoch, and fixed-eval metrics.
`wandb_run_id.txt` preserves the run identity across native resumes.

Only one stable checkpoint exists:

```text
${RESULTS_DIR}/${EXP_NAME}/latest.pt
```

It includes model, fp32 EMA, AdamW, scheduler, eight-rank RNG states, config
digest, step, and completed data epoch. `latest.pt.tmp` is used only for
atomic replacement. No step/best/final checkpoint is created.

```bash
python scripts/validate_e117_maskgit_latest.py \
  "${RESULTS_DIR}/${EXP_NAME}" --expected-world-size 8
```

## Tests

These checks do not start formal training:

```bash
bash -n scripts/launch_e117_maskgit_h200.sh scripts/launch_extract_val_h200.sh
python -m py_compile e117_sparse_*.py train_e117_sparse_maskgit.py scripts/*.py
pytest -q
```

The real capacity probe must run on H200; a local GPU cannot establish H200
capacity.

## Publication protocol

The 40k result only authorizes continued training. Publication claims require
frozen independent seeds, identical splits/decoder/evaluator, 50k generated
images per method, FID uncertainty, throughput/peak-memory reporting, and
disclosure of the selected H200 batch. See
[docs/MASKGIT_EXPERIMENT_PROTOCOL.md](docs/MASKGIT_EXPERIMENT_PROTOCOL.md).
