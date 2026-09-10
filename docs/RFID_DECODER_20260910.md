# MoT decoder weight audit and 50k reconstruction FID

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-10
- Verification Status: UNVERIFIED (not an independent full replication)
- Version Label: mot_ema_reconstruction_rfid50k_v1

## Purpose

Verify the actual decoder weights and reconstruct real ImageNet validation
images. This evaluation does not use MaskGIT, generated prefixes, generated
2D codes or a generation FID reference. All real images are center-cropped
with the same transform used to extract the packed codes.

The checkpoint is MoT `latest.pt`, state `model_ema`, step 199440:
`/home/heyefei/lichenge/MoT/weights/sophiaa_root_latest/latest.pt`.
SHA-256:
`86c8f9da5e61261ab93066c73d7719203e8c00b69f05b805c5937e6b7319b446`.

Remote identity checked on 2026-09-10 using the public Hugging Face tree API:
`sophiaa/MoT-1-checkpoints/latest.pt`, size 6,400,628,829 bytes, LFS SHA-256
exactly matches the local hash above. Repository file commit:
`fec1d064e44060e0591d9421a7f34c58ce80404d` (2026-08-07). Thus the loaded file
matches the user-specified remote checkpoint, not merely a same-named local file.

## Endpoints

| Name | Input / reconstruction |
|---|---|
| real | Original 50k ImageNet validation crops; same-sample rFID reference |
| base_1d | Real 32-token 1D code, MoT base decoder, no refinement |
| mix96 | Real 1D + real 2D, original checkpoint's f2d-aware Router, exactly 96/256 cells (37.5%) |
| mix_e117 | Same real codes, content-only E117 Router, 64 or 128 selected cells |
| titok_native | Real 1D code decoded by the original TiTok decoder, independent control |

The original f2d-aware Router is legitimate for tokenizer reconstruction
because real-image 2D features exist. It is not available from 1D alone during
generation. Do not relabel `mix96` as the deployed E117 generation pipeline.

## rFID definition

Use the same torchmetrics / torch-fidelity feature family as
`MoT/eval_titok_llamagen_mix_metrics_router_f2d_e2e_dynamic.py`, with
`FrechetInceptionDistance(feature=2048, normalize=False)`.
All feature extraction is on GPU from uint8 NCHW images; mean/covariance
aggregation and torchmetrics' `_compute_fid` use FP64 on CPU after workers
finish. Images are rounded to uint8 using the original MoT convention.
Reference features come from the exact same 50,000 real validation crops.
No ADM/VIRTUAL reference features are mixed with torch-fidelity features.

Frozen reconstruction is FP32 with TF32 off, batch32. Fresh-code audit alone
temporarily uses the cache extraction settings (FP32, TF32 on, batch32), then
restores evaluation flags. Thus this matches the original metric family,
preprocessing and real-image reference, not every original command's precision
default (the original script defaults to BF16 and enables TF32).

PSNR is an additional paired check: mean per-image PSNR on float images clamped
to [0,1], using the original MoT epsilon 1e-8. It is not the quantized PSNR from
some earlier generation diagnostics.

## Mandatory checks passed before full evaluation

The 128-image four-GPU smoke completed successfully (exit code 0).

- Each rank compared all 692 stored EMA tensors against the actual loaded
  model, including every required latent-decoder, LlamaGen and Router parameter.
  No mismatch, missing parameter or unexpected checkpoint tensor.
- Frozen checkpoint step199440 and fixed min/max 96 verified.
- Each rank freshly re-encoded 32 real images: all 1D and 2D IDs exactly equal
  the corresponding packed cache (128 audited images total).
- Original MoT base decode and the E117 adapter's base decode: max error 0.
- Their intermediate 1D feature tensors: max error 0.
- Sparse E117 decode and dense replacement of the same selected cells: max
  pixel error 0.
- Peak reserved memory approximately 21.02 GiB per GPU.
- No small-sample FID is reported as a 50k result.

Evidence: `rfid_decoder_smoke_20260910/weights_and_cache_audit_rank*.json`.
The full run repeats these checks; no pass is inherited without execution.

## Command

From the AR repository root, with the existing local environment/assets:

```bash
OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 OMP_NUM_THREADS=4 \
python3 -u experiments/titok_bert_migration_20260909/eval_decoder_rfid50k.py \
  --output experiments/titok_bert_migration_20260909/rfid_decoder50k_ema199440_20260910 \
  --gpus 0,1,2,3 --n 50000 --batch 32
```

Fresh output directory required. Smoke: use a separate output directory and
`--n 128 --smoke`. Four GPUs maximum. W&B online, project `motar-maskgit`,
group `mot-decoder-rfid`; no `log.txt` or checkpoint write.

## Outputs / completion

`manifest.json` records source/weight/cache hashes, versions and metric
definition. `progress.json` records completion counts; `summary.json`
must report complete, n50000, non-smoke and full coverage. Five feature arrays,
per-image PSNR and selected K are retained, plus only eight-image previews.
All worker audit files, failures (if any) and timings are retained. A single
evaluation does not establish uncertainty across seeds or fix the generator.

## Completed 50k results

The full evaluation exited successfully (code 0), with 50,000 / 50,000 unique
images covered, no failure files, and unchanged input/source hashes. Runtime
was 964.47 seconds (about 16.1 minutes) on four RTX 5090 GPUs, batch32 per GPU.
All four GPUs were released after completion. Retained output is approximately
2.06 GB; no checkpoint was modified and no training was started.

| Reconstruction | rFID, lower is better | Mean PSNR, dB |
|---|---:|---:|
| MoT pure 1D, 32 real tokens | 2.9816565061 | 15.5928245903 |
| MoT mix96, original Router, fixed 37.5% | 1.3069101203 | 19.5066683017 |
| MoT mix with content-only E117 Router | 1.4179407999 | 18.6236809873 |
| Original TiTok decoder, same 32 real tokens | 2.2189351830 | 15.8956869416 |

E117 selected 64 cells for 23,677 images and 128 cells for 26,323 images:
mean 97.69344 cells, or 38.1615%. Its budget is close to, but not exactly, the
fixed 96-cell/37.5% arm. Their difference is not a budget-matched Router ablation.

All four full-run audits repeated and passed: 692 EMA tensors exactly equal
the loaded model; no missing required parameters; 128 freshly encoded images
have zero 1D/2D cache mismatches; base/latent adapter and sparse/dense decode
maximum errors are zero. An independent post-run read verified five finite
feature arrays of shape (50000, 2048), no all-zero feature rows, finite PSNR,
valid E117 budgets, and the complete written bitmap.

These results support correct loading of the user-specified MoT EMA decoder
and demonstrate good reconstruction with real codes (mix96 rFID about 1.3).
They do not establish that generated 2D codes have the same quality, or fully
identify the source of the earlier generation-FID degradation. MoT's base
decoder is intentionally not the original TiTok decoder; under the same real
codes it has higher rFID here. Reconstruction FID in this report must not be
numerically conflated with the earlier ADM/VIRTUAL generation FID.

The earlier controlled generator/decoder comparison is documented separately
in `OFFICIAL_COMPARE_20260910.md`; it holds the decoder fixed when comparing
official and current generators. Together these checks distinguish decoder
choice, decoder loading, and generated-code quality.

Full numerical result: `rfid_decoder50k_ema199440_20260910/summary.json`.
W&B (synced successfully):
https://wandb.ai/lcg-li-chenge-zhejiang-university/motar-maskgit/runs/xrnb2bdm
