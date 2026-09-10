# TiTok official MaskGIT / current model comparison (2026-09-10)

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-10
- Verification Status: UNVERIFIED (not an independent full replication)
- Version Label: titok_official_current_decoder_attribution_v2

## Question and controls

Does the current model's almost unchanged 1D masked NLL imply preserved image
generation? Separate generator weights from the image decoder. All new FID
endpoints use 50,000 sample IDs with 50 labels per ImageNet class, the same
seed (20260925), generation batch (64), 1D sampler and ADM reference as the
preceding completed evaluation. No real prefix enters free generation.

| Generator | Decoder | Source |
|---|---|---|
| Official TiTok-L32 MaskGIT | Original TiTok | New: official_native |
| Official TiTok-L32 MaskGIT | Frozen MoT base | New: official_mot_base |
| Current step-10938 EMA | Original TiTok | New: current_native, using prior saved generated codes |
| Current step-10938 EMA | Frozen MoT base | Previous complete 50k run, base |
| Current step-10938 EMA | MoT with generated sparse 2D | Previous complete 50k run, mixed |

The current generator's old codes are reused, not regenerated with a different
seed. The old base endpoint was **not** the official TiTok decoder. A
generator-only comparison must hold the decoder fixed.

1D sampling: upstream `ImageBert.generate`, 8 iterations, CFG 4.5 with linear
guidance, randomization temperature 9.5, no softmax-temperature annealing.
Generator forward is BF16; CFG/Gumbel sampling is FP32 for both generators;
image decoders are FP32, TF32 disabled. Native uint8 postprocessing follows
TiTok's clamp-[0,1], multiply-255, truncate; MoT retains its original
clamp-[-1,1], scale-and-round procedure. Native decoding sub-batch is 32.

Official README reports TiTok-L32 generation FID 2.77:
[official model zoo](https://github.com/bytedance/1d-tokenizer/blob/main/README_TiTok.md).
That is a published reference, not a measured result of this run. Their default
benchmark uses a different seed/batch and FP32 with TF32 enabled; this run uses
the controlled common-precision protocol above. Do not call it a bitwise
replication of the published number.

## NLL measurement (completed)

`compare_official_nll1d.py` evaluated the exact 1,024-image validation development
cohort saved during training, using the same masks (seed 20260922), K grouping
and batch size 32 as `baseline.evaluate`. This is not train loss and not a new
held-out generalization set. Hard-label NLL is computed only at masked positions,
without label smoothing; conditional and null-class paths are recorded separately.

| Masked 1D positions | Official conditional NLL | Current EMA conditional NLL | Top-1 prediction agreement |
|---|---:|---:|---:|
| 8 / 32 | 7.020416 | 7.030381 | 70.50% |
| 16 / 32 | 7.415346 | 7.421182 | 73.29% |
| 32 / 32 | 8.057971 | 8.056044 | 82.62% |
| Equal mean of the three regimes | 7.497911 | 7.502535 | not pooled |

Difference in mean NLL is +0.004624. Full numbers, null-class results,
per-sample values and source hashes are under
`official_current_nll1d_20260910/`.
W&B: https://wandb.ai/lcg-li-chenge-zhejiang-university/motar-maskgit/runs/4yiqbvsv

This controls the validation definition; it does not imply identical sampling.
NLL scores the probability of ground-truth tokens given masked real tokens.
The sampler uses its own generated context, class guidance, token ranking and
repeated remasking. Decoder changes do not enter token NLL at all. The resulting
FID differences must be measured, not deduced from NLL alone.

## Precision-confounded attempt: excluded

The first new evaluator (`eval_official50k_gpu.py`, protocol v1) inadvertently
left the official BF16 output logits as BF16, while the current model's
`forward_1d` returns FP32. The upstream Gumbel implementation creates random
noise with the logits' dtype, so these were different sampling procedures.
This problem did not affect the previous current-model 50k run or the NLL
comparison (both NLL paths cast logits to FP32).

The v1 full run was stopped deliberately with SIGINT at approximately 5,440
feature-complete samples (W&B 8bnvutno). Partial files and failure records are
retained, but no FID from that run is used. The original v1 script is retained.

V2 adds an official-forward adapter that converts logits to FP32 before calling
the same upstream sampling routine. Its 256-image four-GPU smoke completed
successfully; token agreement with current cached samples was 55.10%.
Smoke mode does not report small-sample FID as a 50k score. The v2 full run
uses a fresh output directory and includes source-file integrity checks.

## Commands

Run from the AR repository root with the already installed local dependencies.

```bash
CUDA_VISIBLE_DEVICES=3 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
python3 -u experiments/titok_bert_migration_20260909/compare_official_nll1d.py \
  --output experiments/titok_bert_migration_20260909/official_current_nll1d_20260910

OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 OMP_NUM_THREADS=4 \
python3 -u experiments/titok_bert_migration_20260909/eval_official50k_gpu_v2.py \
  --output experiments/titok_bert_migration_20260909/official_current_decoder50k_v2_seed20260925 \
  --gpus 0,1,2,3 --n 50000 --batch 64 --decode-batch 32 --feature-batch 32
```

Output directories must be fresh. To smoke-test, choose a fresh directory and
add `--n 256 --smoke`. Do not run NLL on GPU 3 concurrently with the full
four-GPU run. No training/checkpoint mutation occurs; W&B is online, no
`log.txt` is created. Local dependency paths remain server-specific.

## FID result status

Read `official_current_decoder50k_v2_seed20260925/summary.json` only after
`status=complete`, `n=50000`, `smoke=false`, `complete_coverage=true`,
`gpu_features_verified=true`. Six ADM feature arrays plus generated codes are
retained (about 2.5 GB), not 150,000 image files. Before handoff, verify all
sample IDs/classes, input hashes, W&B upload, normal exit and GPU release.
A single seed does not estimate FID uncertainty or establish statistical
significance. No new training or configuration sweep is authorized by this
evaluation alone.

## Completed results

V2 completed normally (exit code 0), runtime 652.110 seconds. All 50,000 sample
IDs were independently checked: no duplicates/missing IDs, exactly 50 per class,
all cached current tokens exactly match the prior complete run. Input hashes
remained unchanged, no failure records exist, and all four GPUs were released.
Outputs occupy 2,455,453,734 bytes at audit time. No checkpoint was changed.

| Generator / output | FID | sFID | IS |
|---|---:|---:|---:|
| Official + native TiTok decoder | 2.881062 | 12.626429 | 178.122284 |
| Current EMA + native TiTok decoder | 2.900411 | 12.508803 | 177.597382 |
| Official + MoT base decoder | 4.719759 | 8.354706 | 142.378159 |
| Current EMA + MoT base decoder (prior run) | 4.715998 | 8.337731 | 142.018433 |
| Current EMA + generated sparse 2D refinement (prior run) | 23.147936 | 14.134065 | 70.028702 |

At fixed native decoder, changing generator weights changes FID by +0.019349.
At fixed MoT base decoder, the observed change is -0.003762. These tiny
single-seed differences are not evidence of a significant regression or gain.
At fixed official generator, replacing native decoding with MoT base raises
FID by +1.838697. The current generator is therefore not the source of the
large native-versus-MoT FID gap in this controlled evaluation.

The 1D conditional NLL change (+0.004624) and nearly unchanged same-decoder FID
are consistent, not contradictory. NLL contains no decoder/refinement terms;
it cannot detect the MoT base decoding gap or the much larger additional
degradation of the current sparse-2D pipeline. This experiment does not
separately attribute the latter to Router, 2D predictor, or 2D sampler.

Final generated-token agreement under shared randomness was 54.0142%; many
discrete token choices changed without a large change in aggregate FID.
NLL/top-1/token agreement should not replace image-level evaluation. A larger
1D loss weight is not supported as a remedy for the observed decoder/2D gap
by these results alone.

Results and sampling audit are online in
[W&B](https://wandb.ai/lcg-li-chenge-zhejiang-university/motar-maskgit/runs/nzlh6oxv).
The NLL run is linked above. The earlier v1 run is excluded for its documented
precision confound, not because of a favorable or unfavorable FID result.
