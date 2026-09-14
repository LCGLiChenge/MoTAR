# MoTAR selected sparse 2D-only MaskGIT

This AR handoff trains a selected sparse **2D-only** MaskGIT for MoTAR.

Contract:

- input: TiTok-L32 1D codes and frozen 16×16 1D-decoder feature map;
- route: E117/Router selected 16×16 cells only, K=64 or K=128;
- output/loss: only selected sparse LlamaGen VQ-16 2D tokens;
- non-selected cells are not generated;
- decode: direct replacement of selected 2D tokens into the 1D base feature grid.

The current handoff model uses `full-context / sparse-output`: the pretrained Halton/LlamaGen-token MaskGIT backbone sees a full 16×16 context internally, but `forward_2d` returns logits only for the selected K positions.

## Fresh H20/H200 setup

Assumption: the conda environment already has PyTorch, CUDA, Hugging Face Hub, safetensors, omegaconf, TensorFlow/ADM FID dependencies, and W&B installed.

```bash
git clone https://github.com/LCGLiChenge/MoTAR.git
cd MoTAR
export PYTHONPATH="$PWD/delivery_h20_20260910:$PWD:$PYTHONPATH"

# Download packed train/val codes, E117 route caches, and MoT/TiTok/Router weights.
python -m h20.assets --root "$PWD/h20_local_assets" --profile joint
python -m h20.assets --root "$PWD/h20_local_assets" --profile joint --verify-only

# Download Halton MaskGIT ImageNet-256 checkpoint used as the LlamaGen-token backbone.
mkdir -p external_weights/halton_maskgit
hf download llvictorll/Halton-MaskGIT ImageNet_256_base.pth \
  --local-dir external_weights/halton_maskgit
```

Expected Halton checkpoint:

```text
external_weights/halton_maskgit/ImageNet_256_base.pth
sha256 7fe25cb80b05743e8b42dacc61d88792a9ab5217d6bd756bef28821e0a5fc68f
```

All other large files are declared in `delivery_h20_20260910/configs/h20_assets.json` and downloaded from Hugging Face. The local 50-update workstation checkpoint is not needed for H20 long training.

## Start 8-GPU H20 training

```bash
cd MoTAR
wandb login
export PYTHONPATH="$PWD/delivery_h20_20260910:$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MOTAR_ASSETS="$PWD/h20_local_assets"
export HALTON_CKPT="$PWD/external_weights/halton_maskgit/ImageNet_256_base.pth"
export MOTAR_MASKGIT_RESULT_ROOT=/mnt/data/heyefei/lichenge/Mixture-of-Tokenizer/AR/results/maskgit_optimization_sweep_20260910
export WANDB_PROJECT=motar-selected2d-maskgit
bash scripts/launch_halton_fullctx_sparse2d_h20.sh
```

Default launcher settings:

```text
context_style      fullctx
NPROC_PER_NODE     8
MICRO_PER_GPU      320
GLOBAL_BATCH       2560
EPOCHS             80
UPDATES            EPOCHS * ceil(2562334 / GLOBAL_BATCH), 80080 with default batch
WARMUP             1000
SAVE_EVERY         ceil(2562334 / GLOBAL_BATCH), 1001 with default batch
EVAL_EVERY         SAVE_EVERY
EVAL_N             512
checkpoint policy  overwrite latest only
W&B                rank0 online logging when WANDB_PROJECT is set
```

To use more/less H20 memory, set `MICRO_PER_GPU` before launch:

```bash
export MICRO_PER_GPU=384
export GLOBAL_BATCH=$((MICRO_PER_GPU * 8))
bash scripts/launch_halton_fullctx_sparse2d_h20.sh
```

Use a fresh `RUN_NAME`, `MOTAR_OUTPUT`, or `EPOCHS` for each independent run.

## Current screened result

Local screening used generated 1D prefixes, E117 routes, selected sparse 2D generation only, and direct replacement decode. These are local5k screens, not final ImageNet FID:

| model | local5k FID | note |
|---|---:|---|
| 1D base only | 11.0116 | no sparse 2D generation |
| sparse-sequence 50u | 9.2655 | old selected-only best |
| full-context 25u | 9.2837 | selected-only output |
| full-context 50u | 9.1383 | current selected-only best screen |
| full-context 75u | 9.1773 | lower NLL but worse FID |

The H20 launcher is configured for 80 augmented ImageNet epochs by default.

Current local checkpoint, not required for H20 long training:

```text
/mnt/data/heyefei/lichenge/Mixture-of-Tokenizer/AR/results/maskgit_optimization_sweep_20260910/halton_fullctx_sparse2d_u50_m112_g448_v1/latest.safetensors
```

## Evaluation

5k screen:

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.maskgit_optimization_sweep_20260910.halton_sparse2d_free_fid \
  --checkpoint /path/to/checkpoint_dir \
  --output /mnt/data/heyefei/lichenge/Mixture-of-Tokenizer/AR/results/maskgit_optimization_sweep_20260910/eval5k_name \
  --state raw --n 5000 --batch 8 --feature-batch 8 --seed 20260914 \
  --feature-chunk 4 --variants halton_margin4
```

Full 50k intentionally requires `--allow-50k`:

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.maskgit_optimization_sweep_20260910.halton_sparse2d_free_fid \
  --checkpoint /path/to/checkpoint_dir \
  --output /mnt/data/heyefei/lichenge/Mixture-of-Tokenizer/AR/results/maskgit_optimization_sweep_20260910/eval50k_name \
  --state raw --n 50000 --allow-50k --batch 8 --feature-batch 8 --seed 20260914 \
  --feature-chunk 4 --variants halton_margin4
```

## Key files

- `scripts/launch_halton_fullctx_sparse2d_h20.sh`: 8-GPU H20 launcher.
- `experiments/maskgit_optimization_sweep_20260910/halton_sparse2d_adapter.py`: selected-only sparse/full-context adapters.
- `experiments/maskgit_optimization_sweep_20260910/halton_sparse2d_trial.py`: trainer with latest-only checkpointing and W&B rank0 logging.
- `experiments/maskgit_optimization_sweep_20260910/halton_sparse2d_free_fid.py`: generated-prefix selected-only FID evaluator.
- `experiments/maskgit_optimization_sweep_20260910/HALTON_SPARSE2D_ONLY_LOCAL5K_RESULT.md`: local screening record.
