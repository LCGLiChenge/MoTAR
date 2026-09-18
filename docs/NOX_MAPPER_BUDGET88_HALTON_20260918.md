# Calibrated no-`x_base` Router + mapper-conditioned Halton (2026-09-18)

This release is the tested 8-H20, 4,000-update **2D-only** sparse MaskGIT
fine-tune. Frozen 1D features feed a no-`x_base` Router and native-codebook
8D proxy mapper. MaskGIT generates only Router-selected 2D IDs; MoT EMA
decoding directly replaces selected 16×16 features, while other locations
retain continuous 1D features. It is **not** a unified 1D+2D generator.

## Exact assets

All paths below are on the original H20, relative to
`/root/data/heyuanyu/yefei/lichenge/`. The checkpoints are **not committed to
GitHub** and are not claimed to be on Hugging Face. Transfer separately when
reproducing elsewhere, and verify SHA256 before use.

| Asset | Relative path | SHA256 |
|---|---|---|
| MoT EMA step 199440 | `MoTAR_assets/weights/mot_latest.pt` | `86c8f9da5e61261ab93066c73d7719203e8c00b69f05b805c5937e6b7319b446` |
| Halton mixed-prefix 20k init | `MoTAR_halton_proxy_20260916/results/halton_rgb_proxy_mixedprefix20k_seed0/latest.pt` | `d77fabe3420b4bd94680868b5c5ccbc36f1f5f1f63c6a2f1d23a6f71286ae6bc` |
| Native-codebook 8D mapper 20k | `MoTAR_proxy_codebook8_40k_20260918/results/proxy_converter_codebook8_40k_b512_seed20260918/step020000.pt` | `6f0edaf78325b07312c9974458a09cf80d354bd3f5bafae06d5bd270abbcb25d` |
| Calibrated no-`x_base` Router | `MoTAR_nox_mapper_ft_20260918/results/router_budget88_calibrated/latest.pt` | `888a5093b5aa9c7c63657b7a0ed1871d605341d3fe38ac6d5adbc6a1caaf2741` |

The original Halton base is `llvictorll/Halton-MaskGIT/ImageNet_256_base.pth`.
Frozen assets and packed ImageNet codes follow the main README's H20 asset
manifest. The small MoT/Halton codebook compatibility audit JSON is included
with this source release. The calibrated Router preserved feature-proxy
weights and added a threshold of `3.764075051320672`. It targets mean K=88
on train; generated 5k calibration averaged 96.448 2D tokens.

## Reproduce on the original H20

From the MoTAR checkout in the documented environment, set persistent paths
outside Git:

```bash
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export MOTAR_ASSETS=/root/data/heyuanyu/yefei/lichenge/MoTAR_assets
export MOTAR_RESULTS=/root/data/heyuanyu/yefei/lichenge/MoTAR_nox_mapper_ft_20260918/results
export HALTON_ROOT=/root/data/heyuanyu/yefei/lichenge/MoTAR_halton_proxy_20260916
export MAP_CKPT=/root/data/heyuanyu/yefei/lichenge/MoTAR_proxy_codebook8_40k_20260918/results/proxy_converter_codebook8_40k_b512_seed20260918/step020000.pt
export ROUTER_CKPT="$MOTAR_RESULTS/router_budget88_calibrated/latest.pt"
export INIT_CKPT="$HALTON_ROOT/results/halton_rgb_proxy_mixedprefix20k_seed0/latest.pt"
export CACHE_ROOT="$MOTAR_RESULTS/cache_train_budget88_full"
export RUN_ROOT="$MOTAR_RESULTS/nox_mapper_budget88_ft_m128_lr5e6_4k_seed0"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MOTAR_EVAL_ALLOWED_GPUS="$CUDA_VISIBLE_DEVICES"
export MOTAR_EVAL_WORKERS=8
```

If the full cache is absent, regenerate it from the packed ImageNet train
codes, frozen mapper and Router:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  experiments/feature_to_token_20260916/cache_nox_mapper_train.py \
  --router-proxy-checkpoint "$ROUTER_CKPT" \
  --mapper-checkpoint "$MAP_CKPT" --output "$CACHE_ROOT" --batch 16
```

Require `status=complete`, `complete_coverage=true`, and 2,562,334 views
in the cache summary. The tested cache averaged 88.4987 selected 2D tokens.
Run training into a **new** output directory:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  experiments/feature_to_token_20260916/train_halton_proxy_h20.py \
  --upstream-root "$HALTON_ROOT/upstream" \
  --pretrained "$HALTON_ROOT/weights/ImageNet_256_base.pth" \
  --mot "$MOTAR_ASSETS/weights/mot_latest.pt" \
  --codebook-audit experiments/feature_to_token_20260916/halton_codebook_audit_mot199440.json \
  --proxy-root "$CACHE_ROOT" --route-root "$CACHE_ROOT" \
  --init-model "$INIT_CKPT" \
  --eval-mapper-checkpoint "$MAP_CKPT" \
  --eval-router-proxy-checkpoint "$ROUTER_CKPT" \
  --eval-router-mode no-xbase --eval-stage2-steps 8 \
  --mask-objective mixed-prefix --eval-cfg-w 2.75 --eval-gpu auto \
  --epochs 50 --micro 128 --seed 0 --lr 5e-6 --warmup 200 \
  --output "$RUN_ROOT" --stop-after 4000 \
  --lr-schedule budget-cosine --lr-hold-until 1000 \
  --lr-decay-until 4000 --min-lr 5e-7 \
  --extra-eval-steps 1000,2000,4000 --log-every 50
```

This is a fresh-optimizer fine-tune from the 20k task model, global batch
1024, **4k updates (~1.6 cache passes), not 50 completed epochs**. A
three-update smoke passed first. The trainer kept a rolling `latest.pt`,
logged to W&B, and removed temporary immutable FID snapshots. The 5k FIDs
at updates 1k/2k/4k were 7.6765/7.6275/7.6651.

Paired 50k evaluation, fixed seed 20260914, CFG 2.75, 8+8 sampling:

```bash
python -c 'from experiments.feature_to_token_20260916.halton_parallel_eval import cli; raise SystemExit(cli())' \
  --checkpoint "$RUN_ROOT/latest.pt" \
  --mapper-checkpoint "$MAP_CKPT" \
  --router-mode no-xbase --router-proxy-checkpoint "$ROUTER_CKPT" \
  --output "$MOTAR_RESULTS/nox_mapper_budget88_ft4k_eval50k_seed20260914" \
  --n 50000 --seed 20260914 --cfg-w 2.75 --stage2-steps 8 \
  --workers 8 --allowed-gpus 0,1,2,3,4,5,6,7
```

## Observed result

| Same 50k cohort | Base FID | Full-refine FID | Mean generated 2D |
|---|---:|---:|---:|
| Earlier no-`x_base` Router + mapper 4k combination | 4.7597 | 1.9296 | 103.7619 |
| Calibrated budget-88 Router + mapper 4k combination | 4.7597 | **1.9072** | **96.3904** |

The completed evaluation covered all 50,000 samples, checked unchanged
anchors, and counted mean 32 1D + 96.3904 generated 2D tokens. Evaluated
checkpoint SHA256:
`0979b33c86e957b2fc42ad18b4d9ae3ad47886e8e78d9a87f1363a4b2c5a53ad`.
The 0.0224 FID numerical gain is small. This comparison changes both Router
calibration and the paired fine-tune; it does **not** isolate a causal training
gain or demonstrate statistical significance. Large checkpoints, cache
arrays, generated images and FID features are not committed to GitHub.
