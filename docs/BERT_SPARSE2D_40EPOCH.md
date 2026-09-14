# BERT sparse 2D-only: 40-epoch handoff

This is the current requested experiment, not the older unified/Halton/fusion
models retained elsewhere in this repository. The implementation is in `bert2d/`.

## Model and initialization

- One BERT encoder, 24 layers, width 768, 16 heads, FFN 3072, about 196.46M
  trainable parameters. Sequence: `[class | 256 frozen 1D features | K sparse 2D]`.
- Only Router-selected 2D tokens are predicted (K=64/128 in this cache). All
  16x16 base features are visible; the model does **not** generate 1D tokens.
- Initialize BERT encoder, input LayerNorm and class embeddings from official
  TiTok-L32 MaskGIT. The 2D vocabulary/head, spatial/modality embeddings and
  feature projection start fresh. This is **not** random initialization of the
  whole model and does **not** require a Halton checkpoint.
- Training uses cached real-image 1D codes and frozen MoT EMA spatial features.
  Evaluation first generates 32 codes with the separate, unchanged official
  TiTok generator, then predicts sparse 2D codes. Selected features directly
  replace the base grid before decoding: no half blending or learned fusion.
- Full packed TRAIN cache, arccos masking, 0.1 label smoothing, 0.1 visible CE,
  0.1 class dropout (spatial condition retained), AdamW betas=(0.9,0.96), WD=0.03,
  grad clip=1. New LR=1e-4 with 50-update ramp; pretrained LR=1e-5, zero for the
  first 20 updates and ramped over 100 updates. Rates then remain constant,
  matching the short experiment; this is not claimed to be a proven long-run schedule.

## 1. Environment

Activate your existing compatible environment. Known training versions are
Python 3.10, torch 2.10.0/CUDA 12.8, torchvision 0.25.0, and the Transformers
commit pinned by `requirements-h20.txt`. Do not casually upgrade Transformers:
the BERT gradient-checkpointing API is version-dependent.

```bash
git clone https://github.com/LCGLiChenge/MoTAR.git
cd MoTAR
# If the environment needs installing (run only in the intended environment):
bash scripts/install_h20_environment.sh --active-env
# Extra dependencies for ADM-FID (also needed when training env already exists):
python -m pip install -r requirements-bert2d-eval.txt
python -m pip check
USE_TF=0 python -m unittest bert2d.test_model bert2d.test_resume
```

The installation script without `--active-env` creates the `motar-h20` conda
environment. A working CUDA driver is still required. TensorFlow must see a GPU
for evaluation; it is not needed for the training forward pass.

## 2. Download all large dependencies

Use a data disk, not the Git checkout. Change `/mnt/data/YOUR_NAME` below to your
own directory; keep these exports in the training/evaluation shell.

```bash
export MOTAR_ASSETS=/mnt/data/YOUR_NAME/MoTAR/assets
export MOTAR_RESULTS=/mnt/data/YOUR_NAME/MoTAR/bert2d
python -m bert2d.assets --root "$MOTAR_ASSETS" --fid
python -m bert2d.assets --root "$MOTAR_ASSETS" --fid --verify-only
```

Downloads use pinned revisions and SHA-256/size checks in
`configs/h20_assets.json`. Files split for transport are reassembled and verified.
`python -m bert2d.assets --fid --list` shows the exact file list and revisions.
The HF download cache also stays under `MOTAR_ASSETS/.hf_download_cache`.
Allow at least 60GB free for assets, temporary download duplication, a training
checkpoint and evaluation features; training startup requires 20GiB free after
the downloads. Do not delete the source weights to make room.

| Dependency | Download source | Use |
| --- | --- | --- |
| Packed train codes and E117 routes | [Chloeeeeeeee123/MoT-1](https://huggingface.co/Chloeeeeeeee123/MoT-1), `h20-titok-bert-20260910/` manifest entries | Full cached training data |
| `generator_titok_l32.bin` | [fun-research/TiTok](https://huggingface.co/fun-research/TiTok) | Initialize BERT; frozen 1D generation at eval |
| `tokenizer_titok_l32.bin` | [fun-research/TiTok](https://huggingface.co/fun-research/TiTok) | Native 1D quantizer/tokenizer |
| `mot_latest.pt` | [sophiaa/MoT-1-checkpoints](https://huggingface.co/sophiaa/MoT-1-checkpoints), `latest.pt` | Frozen trained MoT EMA decoder and 2D codebook |
| `router/e117.pt` | [Chloeeeeeeee123/MoT-1](https://huggingface.co/Chloeeeeeeee123/MoT-1), `h20-titok-bert-20260910/router/e117.pt` | Inference region selection |
| ADM reference NPZ and Inception graph | [OpenAI guided-diffusion evaluation assets](https://github.com/openai/guided-diffusion/tree/main/evaluations) | Same reference/statistics as local FID |

No unpublished local experiment checkpoint, DetailFlow directory, standalone
original LlamaGen checkpoint, validation-code cache, or external evaluation
script is required. This launcher consumes the provided train cache; it does
not need to extract codes again or access ImageNet images during training.

## 3. Start 40 epochs

```bash
wandb login
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # only GPUs allocated to this job
bash scripts/launch_bert_sparse2d_40epoch.sh
```

Default output: `$MOTAR_RESULTS/bert_sparse2d_40epoch`.
Supported GPU counts: 1/2/4/8. GPU count is not permission to occupy other jobs.
Startup verifies assets, probes worst-case K=128 memory, runs distributed
save + resume checks, then resets to official initialization for formal training.
Failed checks stop launch. Smoke weights are deleted after successful verification;
their small audit JSONs remain in a separate startup directory.

Defaults:

- **40 packed-data epochs**. Each is `ceil(2,562,334 / global_batch)` optimizer
  updates. The cache has two views of each of 1,281,167 source images, so 40
  packed epochs are approximately 80 source-image-equivalent epochs. Bucket
  sampling drops incomplete batches; this is an update-based epoch convention.
  At global batch 448: 5,720 updates/epoch, **228,800 updates total**.
- Global batch **448**, bf16, first 10 BERT layers activation-recomputed.
  Auto-probe chooses the largest safe microbatch that divides global batch.
  With 8 GPUs the ceiling is 56/GPU; it will not silently increase the global
  batch merely to fill H20/H200 memory. To intentionally change the recipe, set
  e.g. `GLOBAL_BATCH=896`; this changes optimization and epoch update counts.
  `MICRO=0` is automatic; an explicit `MICRO` must divide the global batch/world.
- W&B online scalar curves, plus small JSON status/metrics; no `log.txt` or
  TensorBoard. Authenticate with `wandb login`; never put keys in Git.
- **Only `latest.pt`**, overwritten after each epoch and on completion or a
  graceful stop. No step/epoch/best checkpoint archive. Save includes raw model,
  Adam, per-rank RNG and data cursor, with strict readback verification. The
  adjacent `latest.json` contains metadata/hash, not a second checkpoint.
  Atomic saving briefly uses `latest.pt.tmp`; it is not a retained checkpoint.
  The `.pt` file uses the safetensors container, not `torch.load`.

Re-run the same command and output directory to resume its `latest.pt` to the
same **total 40 epochs**, not 40 more. World size, micro/global batch, LR, seed,
assets and model configuration must match. The launcher rechecks memory.
Do not substitute the older local 4k/8k probe directory into this formal launcher.
Use a new `OUTPUT` directory for an independent experiment.

## 4. Generation and FID

Evaluate a stable checkpoint (training finished or stopped); do not race the
epoch-level overwrite of `latest.pt`. All shards must load the same SHA-256 or
the merge fails. FID here is **free generation**, not reconstruction FID.

```bash
# GPU TensorFlow visibility must be nonempty:
python -c 'import tensorflow as tf; assert tf.config.list_physical_devices("GPU")'

# 5k screen, base and full direct-replacement refinement on paired 1D samples:
python -m bert2d.eval_sharded \
  --checkpoint "$MOTAR_RESULTS/bert_sparse2d_40epoch" \
  --output "$MOTAR_RESULTS/bert_sparse2d_40epoch_fid5k" \
  --assets-root-override "$MOTAR_ASSETS" \
  --gpus 0,1,2,3,4,5,6,7 --n 5000 --batch 8 --feature-batch 8 \
  --seed 20260914 --variants halton_fixed_margin4

# Full 50k (use a different, fresh output directory):
python -m bert2d.eval_sharded \
  --checkpoint "$MOTAR_RESULTS/bert_sparse2d_40epoch" \
  --output "$MOTAR_RESULTS/bert_sparse2d_40epoch_fid50k" \
  --assets-root-override "$MOTAR_ASSETS" \
  --gpus 0,1,2,3,4,5,6,7 --n 50000 --allow-50k \
  --batch 8 --feature-batch 8 --seed 20260914 --variants halton_fixed_margin4
```

Use `summary.json` for metrics and `manifest.json` for protocol/provenance.
Shards write features and sample coverage; one global FID is computed from the
merged features, **not** by averaging shard FIDs. Evaluation does not upload
images/features to W&B. Sampling settings remain the local experiment's defaults:
official TiTok 8 steps / CFG 4.5 linear / randomize temperature 9.5;
2D `halton_fixed_margin4` is specified in `bert2d/sampling.py`.

Compare runs with the same sample count, reference, seed, shard count, batch
size, sampling config and model state (raw). Changing shard count/batch changes
the random sampling stream. Never compare 5k directly to 50k, and do not label
local5k as the standard ImageNet50k benchmark.

## Evidence and limits

The unchanged short model trained successfully on four local GPUs. At matched
local5k settings (4 shards, batch 8, seed 20260914), 2,000 updates gave full-refine
FID 58.6813 and 4,000 updates gave 49.6225; paired pure-1D baseline was 10.8663.
This is **not yet a successful generation result** or evidence that 40 epochs
will outperform 1D. The long run tests that question.

Packaging checks are documented in `BERT_SPARSE2D_VALIDATION.md`. Remote H20/H200
GPU capacity and full-state startup checks run on the target server; CPU tests
alone are not evidence of a tested eight-GPU long run.
