# BERT sparse2D: reference-batch AdamW scaling

This is an optimization change, not a model or sampling change. New runs default
to `--batch-scaling adamw-sde`. The reference batch remains **448**. It is a
theory-guided candidate, **not a demonstrated FID improvement**.

## Rules and evidence

For `k = global_batch / 448`:

- Both peak learning rates multiply by `sqrt(k)`; their ratio stays 10:1.
- `beta_i = 1 - k * (1 - beta_i_reference)`.
- `eps = 1e-8 / sqrt(k)`.
- AdamW weight decay is `.03 * sqrt(k)`. This preserves the first-order
  cumulative decoupled shrinkage per processed sample, not exactly the entire
  optimization trajectory. It is not ordinary L2 regularization.
- Freeze and warmup use the sample clock `step * k`, preserving the reference
  sample budgets: fresh warmup 22,400 examples; pretrained freeze 8,960 examples,
  followed by a 44,800-example ramp. Boundaries are quantized to whole updates.
- Peak rates remain constant after warmup. We do not add an unrelated cosine
  schedule, new objective, loss weighting, parameter grouping, or optimizer.
- Gradient clipping stays 1.0. Its frequency is logged; the theory does not
  establish a universal clipping-threshold scaling law for this task.

Sources:

1. [Malladi et al., On the SDEs and Scaling Rules for Adaptive Gradient Algorithms,
   NeurIPS 2022](https://arxiv.org/abs/2205.10287): joint Adam scaling, conditional
   on the stochastic approximation regime. This is not a theorem that FID will match.
2. [How to Scale Hyperparameters as Batch Size Increases, author's explanation](https://sadhikamalladi.github.io/blog/2024/01/22/SDEs-ScalingRules/):
   explicit formulas and limits.
3. [How to Scale Your EMA,
   NeurIPS 2023, Appendix C.2 Eq. 13](https://arxiv.org/abs/2307.13813):
   small-learning-rate weight-decay scaling. The original Adam SDE result used
   zero weight decay; the decay rule is a separate approximation.

| Setting | Reference B=448 | B=3200 |
|---|---:|---:|
| Fresh peak LR | 0.0001 | 0.00026726124 |
| Pretrained peak LR | 0.00001 | 0.000026726124 |
| beta1 | 0.9 | 0.28571429 |
| beta2 | 0.96 | 0.71428571 |
| epsilon | 1e-8 | 3.7416574e-9 |
| weight decay | 0.03 | 0.080178373 |
| Fresh reaches peak | update 50 | update 7 |
| Pretrained first nonzero LR | update 21 | update 3 |
| Pretrained reaches peak | update 120 | update 17 |

The very short warmup and low beta1 at B=3200 are consequences of this reference
recipe, not independently validated best choices. Scaling can leave its useful
regime before betas become invalid. Invalid betas cause a hard error; no clamp or
automatic fallback is used. Sparse K=64/128 and random mask counts also make
image-batch size an imperfect proxy for the number of supervised tokens.

## Inspect and start a NEW run

Use the environment/assets instructions in [the 40-epoch guide](BERT_SPARSE2D_40EPOCH.md).
The preview is read-only and requires no GPU, weights, or W&B login:

```bash
python -m bert2d.optimization --global-batch 3200
python -m bert2d.launch --global-batch 3200 --print-optimization
```

Only on an allocated 8-GPU H20 server, after choosing a **fresh output**:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export GLOBAL_BATCH=3200
export MICRO=0
export BATCH_SCALING=adamw-sde
export OUTPUT="$MOTAR_RESULTS/bert_sparse2d_40epoch_b3200_sde_v1"
bash scripts/launch_bert_sparse2d_40epoch.sh
```

`MICRO=0` probes divisors of the requested global batch. At 8 GPUs it can select
400/GPU if safe; it can select smaller microbatches with accumulation, without
changing the effective batch or applying scaling twice. The probe now loads the
official initialization and uses the same parameter groups and resolved optimizer
at post-warmup rates. Its three updates are capacity checks, not quality evidence.

`LR_NEW` and `LR_PRETRAINED`, or their CLI equivalents, are **reference-batch**
rates, not already-scaled rates. Leave them at 1e-4/1e-5 for the table above.
The launcher forwards the same recipe to capacity, save/resume smoke, and training.

## Bounded H20 comparison (run these, not the full-training command, for testing)

Update the checkout with `git pull --ff-only`, activate the existing environment,
set `MOTAR_ASSETS`/`MOTAR_RESULTS`, and log into W&B as in the main guide.
Use the **same eight allocated H20 GPUs** for both runs, sequentially:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MICRO=0
export BATCH_SCALING=adamw-sde
export LR_NEW=1e-4 LR_PRETRAINED=1e-5
export OUTPUT="$MOTAR_RESULTS/bert_h20_b448_sde_control_seed0"
GLOBAL_BATCH=448 bash scripts/test_bert_sparse2d_h20_scaling.sh
export OUTPUT="$MOTAR_RESULTS/bert_h20_b3200_sde_test_seed0"
GLOBAL_BATCH=3200 bash scripts/test_bert_sparse2d_h20_scaling.sh
```

Each command starts from official TiTok weights with seed 0, runs **one packed
training epoch**, performs one paired full-refine **5k FID**, logs to W&B, and exits.
It does **not** automatically continue to 40 epochs. The trainer metadata retains
its full target of 40; the launcher/test pipeline is explicitly capped at 1.
B448 processes 2,562,560 examples (5,720 updates); B3200 processes 2,563,200
(801 updates). The 640-example difference is epoch rounding, about 0.025%.
The B448 SDE recipe is numerically identical to the old B448 baseline.

Auto-probe preserves these global batches. No claim that 400/GPU fits a particular
H20 server is made before its on-device probe. Keep the same eval batch/seed/GPU
count in both runs. Do not reuse a previous unrelated training output. If an
existing output matches this test, rerunning resumes/reuses it rather than starting
an independent seed. To later continue one test to the original 40-epoch target,
use the ordinary launcher with the **same** output and optimizer/batch settings;
choose the desired evaluation cadence explicitly.

Report both W&B run links and each output's
`evaluations/epoch001_attempt01/summary.json` (or the successful retry directory),
plus `metrics.jsonl` and startup `launch.json`. Compare **full-refine FID at this
matched sample budget** and training throughput on H20. One-epoch seed-0 tests
are screening evidence, not proof of long-run superiority. The existing B3200
legacy run is only a matched control if initialization, sample budget, eval
sharding and settings are confirmed identical.

## Checkpoints and reporting

- Still 40 packed epochs, latest-only each epoch, paired full-refine 5k FID every
  two epochs, W&B online by default. No change to evaluator or model weights format.
- `config.json`, `latest.json`, startup `launch.json`, and W&B config include the
  full resolved `optimization` dictionary, not just the reference LR inputs.
- Curves include both effective LRs, betas, epsilon, WD, clip fraction, processed
  examples, reference updates, and samples/second. W&B defaults retain step axes;
  select `samples_seen` when comparing different batches. FID stays primary.
- New checkpoints resume only with identical recipe/world/micro/global batch.
  The optimizer groups are checked before restoration, so stale saved settings
  cannot silently overwrite the new ones.
- **Old checkpoints require `BATCH_SCALING=legacy` to continue their old recipe.**
  This does NOT apply the new rules to them. To try the new recipe use a new
  output and official TiTok initialization. Existing checkpoints are never edited.
  No optimizer-state migration or warm-start conversion is implemented here.

At B=3200 there are 801 updates/packed epoch and 32,040 updates in 40 epochs.
Compare full-refine FID at matched samples and GPU cost; neither occupancy nor
same-step loss establishes sample efficiency. Further long-run/FID validation
is required before claiming the theory-guided recipe improves generation.

## Implementation checks

```bash
USE_TF=0 python -m unittest bert2d.test_model bert2d.test_resume \
  bert2d.test_periodic_eval bert2d.test_optimization
```

Tests cover formulas, sample-clock boundaries, legacy behavior, invalid inputs,
CLI transport, strict recipe guards, bit-exact B448 baseline updates and exact
scaled-optimizer resume on a deterministic CPU fixture. These are implementation
tests, not evidence that large-batch learning efficiency or FID improved.
