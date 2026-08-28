# MoTAR

Unified autoregressive training for:

- original TiTok-L32 1D codes: 32 tokens, vocabulary 4096;
- MoT-finetuned original LlamaGen VQ-16 2D codes: 256 tokens, vocabulary 16384;
- one class-conditional AR model trained from scratch, with a modest 1.5x loss
  weight on the slower-learning 1D stream.

This handoff is registered for **8 NVIDIA H200 GPUs** and **150 total epochs**.
The first launch always initializes the AR model randomly and trains it from
scratch. It does not accept a checkpoint from the current 4-GPU experiment.

## What is included

- the verified 24-layer, 1024-wide unified AR model;
- the minimal MIT-licensed RandAR model subset;
- reproducible downloads for the exact MoT, TiTok-L32, and LlamaGen VQ-16
  extraction assets;
- packed-code validation;
- real H200 forward/backward/AdamW memory probing;
- automatic per-GPU micro-batch selection;
- 150-epoch training with sqrt learning-rate scaling from the historical
  global batch 576;
- a latest-only checkpoint policy.

Tokenizer weights and image datasets are not needed during AR training because
the trainer consumes discrete codes. You may either copy the verified 1.4 GiB
packed dataset or regenerate it from ImageNet on the H200 host. See
[docs/DATA_AND_CHECKPOINTS.md](docs/DATA_AND_CHECKPOINTS.md) for exact
asset checksums.

## H200 quick start

Use an H200-compatible PyTorch image or install the CUDA build recommended for
the target cluster first. Then:

```bash
git clone https://github.com/LCGLiChenge/MoTAR.git
cd MoTAR

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If packed codes are not already present, prepare them first:

```bash
export ASSET_ROOT=/persistent/assets/motar-extraction
export IMAGENET_TRAIN=/persistent/datasets/imagenet/train
export PACKED_CODE_ROOT=/persistent/data/imagenet-titok_l32-mot199440ema-adm-256_packed

bash scripts/launch_extract_h200.sh
```

This downloads and SHA256-verifies:

- `sophiaa/MoT-1-checkpoints/latest.pt` (6.40 GB, `model_ema`, step 199440);
- `fun-research/TiTok/tokenizer_titok_l32.bin` (2.56 GB);
- `FoundationVision/LlamaGen/vq_ds16_c2i.pt` (288 MB);
- pinned TiTok and LlamaGen source commits.

Extraction uses exactly eight GPUs, deterministic ADM center crop plus
horizontal flip, FP32 tokenizer inference, and a per-GPU batch of 64. This is
separate from the AR training batch. If the target H200 environment has unusual
co-tenancy, lower only `EXTRACT_BATCH_SIZE`; extraction is resumable through
`written.npy`.

Set persistent paths:

```bash
export PACKED_CODE_ROOT=/persistent/data/imagenet-titok_l32-mot199440ema-adm-256_packed
export RESULTS_DIR=/persistent/results/motar
export WANDB_PROJECT=motar-unified-ar
wandb login
```

Start the new H200 experiment:

```bash
bash scripts/launch_h200.sh
```

Do not copy or provide an old AR checkpoint. On its first launch, the output
directory must not contain `checkpoints/latest`. After this H200 run has saved
an epoch, the launcher will use only that same output directory's `latest` to
recover an interruption.

The launcher requires exactly eight visible H200s. Override their indices if
needed:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/launch_h200.sh
```

## Automatic H200 batch sizing

The default is `MICRO_BATCH_SIZE=auto`. Before torchrun starts, the launcher
uses one visible H200 and performs a real bf16 forward, backward, gradient clip,
and fused AdamW step for descending candidates:

```text
768, 704, 640, 576, 512, 448, 384, 320, 256
```

The largest candidate whose peak reserved memory is at most 90% of H200 memory
is selected. Each probe runs in a separate process, so an OOM fully releases
memory before trying the next candidate. The selected value is saved in:

```text
$RESULTS_DIR/$EXP_NAME/h200_micro_batch_size.txt
```

and reused on restart. The training global batch is:

```text
micro_batch_per_gpu * 8
```

The base learning rate is scaled from `1.75e-5 @ global batch 576` with the
square root of the batch ratio. The exact batch and learning rates are recorded
in `run_plan.json` and every checkpoint's `metadata.json`.

For a more conservative margin:

```bash
H200_MEMORY_FRACTION=0.85 bash scripts/launch_h200.sh
```

For a deliberate fixed value:

```bash
MICRO_BATCH_SIZE=512 bash scripts/launch_h200.sh
```

Do not guess a larger batch without running the probe.

## Why the 1D loss weight is 1.5

The two cross-entropy losses are reduced independently over their own tokens,
so the 32-token versus 256-token sequence lengths do not make 1D one eighth of
the objective. Nevertheless, the scratch pilot showed materially slower 1D
learning: by step 55k, 1D loss moved from 8.52 to 7.59 with 0.56% token
accuracy, while 2D moved from 9.70 to 6.66 with 5.27% accuracy. Because the 2D
stream is generated after and conditioned on the 1D prefix, the registered H200
run uses a conservative 1.5x 1D weight.

Do not change the ratio during a run. Both raw losses and accuracies remain
logged separately, and the exact weights are written into checkpoint metadata.

## Epoch and resume semantics

The target is 150 total ImageNet-equivalent epochs.

- Fresh run: epochs 0 through 149.
- Native H200 restart: the launcher detects
  `$RESULTS_DIR/$EXP_NAME/checkpoints/latest` automatically.

There is no command-line path for importing a prior AR checkpoint. Native H200
restarts load their own model and AdamW state, then rebuild the cosine schedule
from the completed epoch recorded in `latest/metadata.json`.

## Checkpoint policy

There is exactly one stable training checkpoint:

```text
$RESULTS_DIR/$EXP_NAME/checkpoints/latest
```

It is replaced after every epoch, including epoch 150. No `epoch_*`,
`iters_*`, `best`, or separate final checkpoint is created.

Saving uses two hidden transactional directories only while rotating:

```text
.latest-next
.latest-previous
```

After a successful save, both are removed. If the host dies during rotation,
startup recovery chooses the last complete state. Validate the current state
with:

```bash
python scripts/validate_latest.py \
  "$RESULTS_DIR/$EXP_NAME/checkpoints/latest"
```

## Fail-fast validation

The launcher automatically checks:

- exactly eight visible H200s and bf16 support;
- all five packed dataset files;
- array shapes, completion metadata, and `written.npy`;
- code/label ranges on deterministic rows;
- checkpoint completeness.

For the one-time full data hash audit:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python scripts/validate_setup.py \
  --packed-code-root "$PACKED_CODE_ROOT" \
  --expected-gpus 8 \
  --verify-manifest
```

## Monitoring

All scalar training logs go to W&B: total/1D/2D loss, 1D/2D accuracy, gradient
norm, learning rate, throughput, epoch progress, and global batch size. The W&B
run name equals `EXP_NAME`; `wandb_run_id.txt` keeps an interrupted native
H200 resume attached to the same W&B run.

The launcher deliberately does not create `train.log` or TensorBoard event
files. Keep torchrun attached to a persistent job/session and use the W&B
dashboard plus GPU telemetry:

```bash
nvidia-smi dmon
```

Set `WANDB_MODE=offline` only when the H200 host cannot reach W&B; sync the
resulting W&B run directory later with `wandb sync`.

Stop the owned torchrun cleanly with SIGINT if there is a NaN/Inf, OOM after
probing, sustained NCCL stall, missing rank, corrupted data, or unexpected GPU
co-tenancy. Do not signal unrelated processes.

## Fixed 1D diagnostic

Use the same first 8192 packed samples for comparable teacher-forced 1D
position metrics:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 evaluate_1d.py \
  --checkpoint-dir "$RESULTS_DIR/$EXP_NAME/checkpoints/latest" \
  --packed-code-root "$PACKED_CODE_ROOT" \
  --output-json "$RESULTS_DIR/$EXP_NAME/eval/diagnostic_1d_positions_8192.json" \
  --num-samples 8192 \
  --batch-size 512
```

The report includes overall, per-position, and 0-3/4-7/8-15/16-23/24-31
cross-entropy and top-1/top-5 accuracy.

## Tests

CPU/static tests do not start the 150-epoch job:

```bash
python -m pip install -r requirements-dev.txt
python -m py_compile train.py motar/*.py scripts/*.py
bash -n scripts/launch_h200.sh scripts/launch_extract_h200.sh
pytest -q
```

The real H200 memory probe cannot be truthfully validated on a non-H200 host;
the production launcher therefore performs it before allocating all eight
cards.

## Reproducibility record

The packed target was extracted from TiTok-L32 plus MoT checkpoint step 199440
(`model_ema`), whose fixed-37.5% reconstruction evaluation reached FID
1.3032376766204834 on 50k ImageNet validation images. This is a tokenizer
reconstruction FID, not AR generation FID.

The full packed-data audit is in:

- [docs/mot199440_packed_audit.json](docs/mot199440_packed_audit.json)
- [docs/mot199440_packed_manifest.sha256](docs/mot199440_packed_manifest.sha256)

The AR still predicts all 256 2D tokens. The MoT decoder later selects exactly
96 of 256 spatial positions for mixture decoding.
