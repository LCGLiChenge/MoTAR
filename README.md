# Current requested experiment: dense proxy MaskGIT, 6000 → 20000 steps

The 2026-09-16 request is a **continuation**, not a fresh40-epoch or unified run.
Convert frozen1D features to nearest2D codebook IDs; a single2D BERT sees
`[class | 256 grid tokens]` and generates **only Router-selected positions**.
Decode by **direct replacement**, retaining continuous base features elsewhere.

Read [the complete H20 guide](docs/DENSE_PROXY_GRID_H20_20000.md), then run
`scripts/launch_dense_proxy_grid_h20_20000.sh`: **8H20, global448, total20k steps,
5k FID at8k/12k/16k/20k, rolling latest.pt only**.
Weights: [Chloeeeeeeee123/MoT-1](https://huggingface.co/Chloeeeeeeee123/MoT-1/tree/main/checkpoints/dense_proxy_grid_maskgit_step6000_20260916).
The downloader can wait for publication without allocating GPUs, then pins
one HF commit and verifies SHA256. Source6000 fullFID5k50.34 is worse than
pure1D10.87; packaging is not a quality claim.

---

# Previous handoff: BERT sparse 2D-only, 40 epochs

The previous requested run was **TiTok-initialized BERT sparse 2D-only**, not the
older unified model below. It takes the complete frozen 1D 16×16 feature grid,
generates only Router-selected 2D tokens, and directly replaces selected regions.

**Batch/LR update (2026-09-15):** new runs use joint AdamW scaling from reference
batch448. [Exact rules and bounded H20 test commands](docs/BERT_BATCH_SCALING.md).
To compare on H20, run `scripts/test_bert_sparse2d_h20_scaling.sh` for B448 and
B3200 in separate outputs: each stops after **one packed epoch + one paired 5k FID**.
Old checkpoints require `BATCH_SCALING=legacy`; never silently change their
optimizer state. Implementation tests passed; FID improvement is not yet established.

Read [the complete training + 5k/50k FID guide](docs/BERT_SPARSE2D_40EPOCH.md).
It includes environment installation, all public weight/cache downloads,
W&B setup, automatic GPU microbatch probing and strict save/resume startup tests.

```bash
# After activating the compatible environment; choose your own data-disk paths:
export MOTAR_ASSETS=/mnt/data/YOUR_NAME/MoTAR/assets
export MOTAR_RESULTS=/mnt/data/YOUR_NAME/MoTAR/bert2d
python -m pip install -r requirements-bert2d-eval.txt
python -m bert2d.assets --root "$MOTAR_ASSETS" --fid
wandb login
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # explicitly allocated GPUs only
bash scripts/launch_bert_sparse2d_40epoch.sh
```

Default: **40 packed-data epochs**, global batch448, fresh official TiTok backbone
initialization; only **latest.pt**, overwritten each epoch. No intermediate
numbered/best checkpoints. Existing output resumes to the same total target.
40 packed epochs include two cached views/image; see the guide for the precise
update and source-image-equivalent counts. All downloads have hash checks.
No unpublished local checkpoint or validation cache is needed.

**Automatic evaluation:** paired 5k FID **every completed epoch (1–40)**,
**asynchronously while training continues**, on the same allocated GPUs. Results
are logged to the same W&B run with the separate `eval/epoch` axis.
Only latest.pt is retained: wait briefly for every shard to load/acknowledge the
checkpoint, then continue training during evaluation. No snapshot or trainer
restart per epoch. At most one eval; backpressure if it falls behind; final drain.
Auto-probe reserves 12 GiB/GPU for eval. See the guide for memory limits, failure
recovery, and safely stopping/updating/resuming an existing B448 job.

[Packaging validation and limitations](docs/BERT_SPARSE2D_VALIDATION.md).
This short-run model has not yet beaten the pure-1D baseline; publication makes
it reproducible and does not claim the long run is already successful.

---

# Legacy documentation — only when explicitly requested

The older 80-epoch unified/Halton instructions below are **not** the default for
this BERT 2D-only handoff. Do not combine their launchers or checkpoints with it.

# MoTAR Unified HaltonMix MaskGIT v2

This repository contains the MoTAR unified MaskGIT experiment for mixed TiTok-L32 1D codes and sparse LlamaGen VQ-16 2D refinement codes.

The current recommended model is `UnifiedHaltonMixMaskGIT v2`:

- 1D branch: official TiTok-L32 MaskGIT/ImageBERT initialization, generating exactly 32 TiTok 1D tokens;
- 2D branch: Halton/LlamaGen full-16x16-context MaskGIT initialization, generating only E117/Router-selected sparse 2D tokens;
- coupling: zero-initialized 1D-hidden to 2D-grid cross-attention residual;
- decode contract: direct replacement of selected 2D features into the 1D base feature grid.

It is a single PyTorch `nn.Module`, one optimizer/checkpoint namespace, one training command, and one free-generation/FID pipeline. The 1D and 2D token types use modality-specific expert structure instead of forcing both token spaces through the same transformer stack.

## Required large files

Assumption: the conda environment already has PyTorch/CUDA, Hugging Face Hub, safetensors, omegaconf, TensorFlow ADM-FID dependencies, and W&B installed.

Download the MoTAR packed codes, E117 routes, MoT/TiTok/Router weights declared in `configs/h20_assets.json`:

```bash
git clone https://github.com/LCGLiChenge/MoTAR.git
cd MoTAR
export PYTHONPATH="$PWD:$PYTHONPATH"

python -m h20.assets --root "$PWD/h20_local_assets" --profile joint
python -m h20.assets --root "$PWD/h20_local_assets" --profile joint --verify-only
```

Those assets are pulled from Hugging Face repo `Chloeeeeeeee123/MoT-1` at pinned revisions in the JSON manifest.

Download the Halton/LlamaGen MaskGIT checkpoint used to initialize the 2D branch:

```bash
mkdir -p external_weights/halton_maskgit
hf download llvictorll/Halton-MaskGIT ImageNet_256_base.pth   --local-dir external_weights/halton_maskgit
```

Expected file:

```text
external_weights/halton_maskgit/ImageNet_256_base.pth
sha256 7fe25cb80b05743e8b42dacc61d88792a9ab5217d6bd756bef28821e0a5fc68f
```

No local 100-update screening checkpoint is required for the formal run. The 80-epoch run starts from the official TiTok 1D MaskGIT and Halton/LlamaGen 2D MaskGIT initializations.

## Start 80-epoch training

Default H20/H200 launcher:

```bash
cd MoTAR
wandb login
export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MOTAR_ASSETS="$PWD/h20_local_assets"
export HALTON_CKPT="$PWD/external_weights/halton_maskgit/ImageNet_256_base.pth"
export MOTAR_MASKGIT_RESULT_ROOT=/mnt/data/heyefei/lichenge/Mixture-of-Tokenizer/AR/results/maskgit_optimization_sweep_20260910
export WANDB_PROJECT=motar-unified-haltonmix-maskgit
bash scripts/launch_unified_haltonmix_v2_h20_80epoch.sh
```

Default launcher settings:

```text
model              UnifiedHaltonMixMaskGIT v2
NPROC_PER_NODE     8
MICRO_PER_GPU      224
GLOBAL_BATCH       MICRO_PER_GPU * NPROC_PER_NODE = 1792
EPOCHS             80
UPDATES/EPOCH      ceil(2562334 / GLOBAL_BATCH) = 1430 with defaults
TOTAL UPDATES      114400 with defaults
SAVE_EVERY         one epoch, overwrite latest only
EVAL_EVERY         one epoch
EVAL_N             512
W&B                rank0 online logging when WANDB_PROJECT is set
```

For larger-memory H200, increase batch before launch, for example:

```bash
export MICRO_PER_GPU=320
export GLOBAL_BATCH=$((MICRO_PER_GPU * 8))
bash scripts/launch_unified_haltonmix_v2_h20_80epoch.sh
```

For H20/96GB, start from the default `MICRO_PER_GPU=224`; if OOM occurs, lower to 192. The local 32GB screen used `MICRO_PER_GPU=72` and peaked at 27.94 GiB/GPU, so the default is intended to use H20 memory aggressively while leaving safety margin.

The trainer writes only `latest.pt` plus JSON sidecars. Intermediate epoch saves overwrite `latest.pt`; no extra epoch checkpoint files are created. The file uses a safetensors container with a `.pt` filename for compatibility with the requested handoff convention.

## Evaluate free generation

5k local screen:

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.maskgit_optimization_sweep_20260910.unified_haltonmix_free_fid   --checkpoint /path/to/checkpoint_dir   --output /mnt/data/heyefei/lichenge/Mixture-of-Tokenizer/AR/results/maskgit_optimization_sweep_20260910/eval5k_unified_haltonmix   --state raw --n 5000 --batch 8 --feature-batch 8 --seed 20260914   --feature-chunk 4 --variants margin4
```

Full 50k requires an explicit flag:

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.maskgit_optimization_sweep_20260910.unified_haltonmix_free_fid   --checkpoint /path/to/checkpoint_dir   --output /mnt/data/heyefei/lichenge/Mixture-of-Tokenizer/AR/results/maskgit_optimization_sweep_20260910/eval50k_unified_haltonmix   --state raw --n 50000 --allow-50k --batch 8 --feature-batch 8 --seed 20260914   --feature-chunk 4 --variants margin4
```

The evaluator proves the full free-generation chain:

```text
checkpoint 1D branch generates TiTok-L32 tokens
→ frozen E117 route is computed from generated 1D only
→ checkpoint 2D branch generates selected sparse LlamaGen tokens
→ direct replacement decode
→ ADM pool/FID with GPU trace
```

## Local v2 screen

The short local screen was used only to validate the implementation before long training.

```text
training: 4 GPUs, micro=72, global_batch=288, 100 updates
checkpoint: /mnt/data/heyefei/lichenge/Mixture-of-Tokenizer/AR/results/maskgit_optimization_sweep_20260910/unified_haltonmix_v2_short4_u100_m72_gpu4567/latest.safetensors
note: that local screen was produced before the formal filename switch; new training runs save latest.pt.
sha256: 170229f41fe3ef6c0c0e0d5a1c4a790fefdd9fe0c88c71e3174de99daec2857d
peak_reserved: 27.94 GiB/GPU
dev nll2d: 8.5598 → 7.7066
```

Generated-prefix local5k screen from that 100-update checkpoint:

| endpoint | local5k FID | IS | note |
|---|---:|---:|---|
| 1D base only | 11.2328 | 138.53 | no 2D refinement |
| unified v2 + margin4 | 9.7463 | 235.85 | direct replacement sparse 2D refinement |
| unified v2 + greedy_margin4_cfg1 | 17.0274 | 108.61 | do not use as default |

Use `margin4` as the default sampler for this branch.

## Key files

- `scripts/launch_unified_haltonmix_v2_h20_80epoch.sh`: 8-GPU 80-epoch launcher.
- `experiments/maskgit_optimization_sweep_20260910/unified_fullctx_maskgit.py`: unified v1/v2 model definitions.
- `experiments/maskgit_optimization_sweep_20260910/unified_fullctx_trial.py`: unified trainer with `--epochs` support and latest-only checkpoints.
- `experiments/maskgit_optimization_sweep_20260910/halton_sparse2d_adapter.py`: Halton sparse/full-context 2D adapters and load accounting.
- `experiments/maskgit_optimization_sweep_20260910/unified_haltonmix_free_fid.py`: unified free-generation ADM-FID evaluator.
- `experiments/maskgit_optimization_sweep_20260910/test_unified_halton_mix.py`: zero-init/coupling structure tests.

## Smoke checks

Before a long run on a new server:

```bash
python -m py_compile   experiments/maskgit_optimization_sweep_20260910/unified_fullctx_maskgit.py   experiments/maskgit_optimization_sweep_20260910/unified_fullctx_trial.py   experiments/maskgit_optimization_sweep_20260910/halton_sparse2d_adapter.py   experiments/maskgit_optimization_sweep_20260910/unified_haltonmix_free_fid.py   experiments/maskgit_optimization_sweep_20260910/test_unified_halton_mix.py

python -m experiments.maskgit_optimization_sweep_20260910.test_unified_halton_mix
```

A tiny DDP smoke can be run by overriding the launcher variables, but do not use the smoke checkpoint as scientific evidence.
