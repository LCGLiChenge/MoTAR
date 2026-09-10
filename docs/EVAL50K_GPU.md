# Frozen 50k GPU evaluation

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-10
- Verification Status: UNVERIFIED (one evaluation run, not an independent reproducibility rerun)
- Version Label: titok_bert_free_generation_adm_v1

## Scope

Evaluate the EMA state in `formal_bs128_warmup600_50000/latest.safetensors`.
The training summary records **step 10,938**, approximately **17.485
source-equivalent epochs**, stopped at the existing three-hour budget. The
directory's `50000` is the requested training-step target, not achieved steps.
The checkpoint SHA-256 is
`1d680436cb401d05a2dd5de45174140e16db0d72670417f526073f5bf194b28c`.

This is class-conditional **free generation**, not reconstruction and not
2D generation conditioned on ground-truth 1D codes. No real images, packed
validation codes, oracle routes, or reference-image features enter generation.

For every sample, generate 32 1D tokens using the upstream TiTok
`ImageBert.generate` through `OfficialSamplingView`. The frozen E117 router
then selects 64 or 128 spatial locations from generated 1D information only.
Generate sparse 2D tokens conditioned on these 1D tokens and locations.

Two paired endpoints share exactly the same generated 1D prefix:

- `base`: the MoT base decoder's `x_base`, without 2D refinement. This is **not**
  the original TiTok image decoder, so it is not an official TiTok baseline.
- `mixed`: replace only selected latent cells with generated 2D codes and use
  the frozen MoT/LlamaGen image decoder. No invented tokens at unselected cells.

Both endpoints use the same 50,000 global sample IDs, 50 per ImageNet class.
Class ID is `sample_id % 1000`. Seed is 20260925. Random streams are seeded by
global batch start, so changing GPU count preserves batch assignment semantics;
changing batch size changes the random stream and requires a new protocol run.

## Fixed sampling / metrics

- 1D: 8 iterations, CFG 4.5, linear guidance decay, randomization temperature 9.5.
- 2D: 8 iterations, CFG 4.5, constant guidance, standard CFG formula,
  randomization temperature 1.0.
- No softmax-temperature annealing or sample rejection/reranking.
- Generation BF16; frozen router and image decoder FP32; TF32 disabled.
- Clamp decoded images to [-1,1], map by `(x + 1) * 127.5`, round to uint8 NHWC.
- Use the existing **ADM TensorFlow frozen Inception graph** and matching
  `VIRTUAL_imagenet256_labeled.npz` reference statistics. Do not substitute
  torch-fidelity features while retaining this reference.
- Inception convolution runs on a dedicated GPU, verified from TensorFlow's
  actual execution trace. The final covariance / matrix square root uses the
  original ADM NumPy/SciPy implementation on CPU.
- Report FID, sFID and Inception Score (IS split size 5,000) separately for
  `base` and `mixed`. One seed is not an uncertainty estimate or proof of
  statistically significant improvement.

## Commands (this server)

Run from the AR repository root. The script currently uses the local dependency
and weight paths recorded in its constants; this document does not claim that
these paths or assets are already available on a new server.

```bash
# Fresh output directory required. Three generation GPUs + one ADM GPU.
OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 OMP_NUM_THREADS=4 \
python3 -u experiments/titok_bert_migration_20260909/eval50k_gpu.py \
  --output experiments/titok_bert_migration_20260909/fid50k_gpu_ema_step10938_seed20260925 \
  --gpus 0,1,2,3 --n 50000 --batch 64 --feature-batch 32
```

Smoke test: use a different fresh output path and add `--n 256 --smoke`.
Smoke mode tests sample coverage and GPU feature extraction but intentionally
does not report small-sample FID as 50k FID. The four-GPU smoke run completed
successfully with generation peak reservation about 25.55 GiB per GPU.

W&B is online in project `motar-maskgit`, group `titok-bert-fid50k`.
Existing local W&B credentials are used; no credential belongs in this file.
Do not redirect console output to `log.txt`. W&B's own internal run files are
normal. The full run URL is stored in `wandb_run.json`.

## Outputs and completion checks

- `manifest.json`: sampling settings and SHA-256 of checkpoint, router, MoT,
  Inception graph, evaluator, reference and evaluation script.
- `progress.json`: current number of feature-complete samples and process PID.
- `codes_*.npz`: generated 1D/2D tokens, labels, routes and sample IDs.
- `base_pool.npy`, `mixed_pool.npy`, `base_spatial.npy`, `mixed_spatial.npy`:
  ADM features indexed by global sample ID, approximately 1.63 GB total.
- `written.npy`: per-sample completion bitmap; `route_counts.npy`: selected K.
- `base_first8.png`, `mixed_first8.png`: paired visual sanity checks only.
- `adm_gpu_trace.json`: evidence that Inception convolution used GPU.
- `summary.json`: final metrics, complete-coverage check, input-integrity check,
  feature hashes and runtime. This must say `status=complete`, `n=50000`,
  `smoke=false`, `complete_coverage=true` and `gpu_features_verified=true`.
- `failure.json` / `generator_*_failure.json`: preserve any runtime failures;
  a partial feature file or W&B run alone is not a completed evaluation.

No checkpoint is written or modified. Only tokens, features, metadata and a few
previews are stored; the 50k image arrays are streamed rather than retained.
The 2D endpoint doubles evaluated images to 100k, but there are 50k paired seeds.

The script is additive: no training code, model architecture, checkpoint,
dataset, or reference statistics were changed for this evaluation. Runtime
monitoring checks process health, output growth and finite/valid samples. The
user authorized a multi-hour evaluation; do not substitute a short training
budget or terminate a healthy 50k run at an arbitrary 30-minute limit.

## Numerical checks during this run

The smoke and full runs produced exactly equal 1D tokens, 2D tokens, routes and
labels for the shared first 256 samples. Both saved first-eight uint8 preview
grids were pixel-identical. Feature arrays were not bitwise identical: relative
L2 differences ranged from 7.05e-5 to 1.16e-4. Bitwise GPU feature reproducibility
is therefore not claimed.

A CPU-only execution of the original ADM graph on eight saved images per
endpoint was compared with this run's GPU features. All four arrays passed the
pre-set relative-L2 tolerance of 1e-3: base pool 6.16e-5, base spatial 1.06e-4,
mixed pool 5.66e-5, mixed spatial 9.33e-5. This is a numerical compatibility
check, not an independent 50k replication or an estimate of FID variability.
The main 50k feature extraction remains GPU.

## Completed result

- Status: completed; process exit code 0; runtime 581.781 seconds.
- Output: `fid50k_gpu_ema_step10938_seed20260925/summary.json`.
- W&B: https://wandb.ai/lcg-li-chenge-zhejiang-university/motar-maskgit/runs/iji6q5vf

| Endpoint | FID (lower) | sFID (lower) | IS (higher) |
|---|---:|---:|---:|
| Generated 1D + MoT base decoder | 4.715998 | 8.337731 | 142.018433 |
| Same generated 1D + generated sparse 2D | 23.147936 | 14.134065 | 70.028702 |

At this checkpoint and fixed sampler, adding the sparse refinement pipeline
worsens FID by 18.431938. This isolates the net effect of the combined routing,
2D generation and refinement path; it does not isolate which individual
component causes the degradation. It is not evidence that sparse generation
is fundamentally infeasible, nor that training longer alone will fix it.

An independent read-through of all 782 saved code chunks confirmed 50,000
unique sample IDs, exactly 50 examples per class and no missing/duplicate
samples. The completion bitmap is full, all input hashes remained unchanged,
and no failure records exist. Mean selected 2D budget: 103.59424 cells.
Outputs occupy approximately 1.65 GB. All four evaluation GPUs were released
after successful completion; no training or checkpoint write was performed.

Environment: PyTorch 2.10.0, Transformers 4.50.0.dev0, NumPy 1.26.2,
SciPy 1.12.0, TensorFlow 2.21.0, W&B 0.25.0, Safetensors 0.7.0;
four NVIDIA GeForce RTX 5090 GPUs. The fixed-pair, input-hash and coverage
checks follow the experiment skill's traceability requirements. A single
seed still does not provide confidence intervals or an independent replication.
