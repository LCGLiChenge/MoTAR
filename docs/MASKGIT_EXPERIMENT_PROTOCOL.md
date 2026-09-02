# E117 unified MaskGIT experiment protocol

## Scope

This protocol separates three claims:

1. optimization feasibility: both token branches reduce fixed-eval CE and use
   their intended conditioning;
2. generative quality: decoded samples improve distributional metrics;
3. efficiency: E117 K=64/128 sparse refinement is faster than dense 256-token
   MaskGIT under matched hardware and sampling settings.

Passing one claim does not establish the others.

## Frozen method identity

- generator starts from random initialization;
- shared 24-layer, width-768 pre-norm bidirectional transformer;
- 32 TiTok-L32 codes followed by E117-routed K in {64,128} LlamaGen VQ codes;
- E117 route checkpoint SHA256
  `a5b84689d2b29f579d2442da7594ac093292b6386760867a0668ca02f82e6156`;
- arccos mask schedule, class dropout 0.1, label smoothing 0.1,
  visible-token loss weight 0.1;
- loss weights 1D=1.5 and 2D=1.0;
- AdamW LR 1e-4, betas 0.9/0.96, weight decay 0.03, EMA 0.999;
- target exposure 1,024,000,000 examples.

The H200 micro-batch is a measured systems parameter, not a tuned quality
hyperparameter. Record the chosen value, global batch, rounded max steps,
throughput, peak allocated/reserved memory, GPU model, software versions, and
wall-clock duration.

## Optimization feasibility

Use one immutable validation cache and mask seed. Report at minimum:

- 1D and 2D masked CE, top-1, and top-5;
- 1D CE after shuffling visible 1D context;
- 1D CE after shuffling class labels;
- 2D CE after shuffling visible 2D context;
- 2D CE after shuffling the completed 1D prefix;
- 2D CE after shuffling class labels.

The local 40k checkpoint passed the registered thresholds once. Treat this as
permission to continue, not checkpoint selection. Formal training uses the
terminal EMA state rather than choosing the best validation/FID checkpoint.

## Sampling hyperparameters

Before examining final FID, freeze a validation-only sweep:

- 1D MaskGIT steps: 8, 16, 32;
- sparse 2D MaskGIT steps: 8, 16, 32;
- CFG scales and temperature schedules on a small, fixed validation generation
  set;
- one deterministic tie-breaking/random seed list shared by all comparisons.

Choose one setting using only the validation generation set. Apply it unchanged
to all test seeds and the final 50k generation. Report the full sweep, including
failed settings, so the terminal choice is auditable.

## Final generative evaluation

- generate exactly 50,000 class-balanced ImageNet samples per method and seed;
- use identical class order, decoder, preprocessing, evaluator implementation,
  real-statistics file, numeric precision, and image format;
- evaluate at least seed 0 plus two independent frozen seeds;
- report per-seed FID and mean/standard deviation (or a justified confidence
  interval);
- report Inception Score only as a secondary metric;
- retain generation manifests containing code revision, checkpoint SHA256,
  config SHA256, sample seed, class schedule, CFG, temperatures, step counts,
  decoder identity, and evaluator identity.

Do not compare train-set metrics with validation-set metrics, and do not reuse
samples or tune on the final 50k outputs.

## Efficiency comparison

Compare against dense 256-token MaskGIT and relevant AR baselines using:

- the same H200 model, batch size where feasible, precision, CFG batching, and
  decoder timing boundary;
- synchronized CUDA timings after warmup;
- separate 1D generation, routing, 2D generation, and decode times;
- throughput, latency per image, peak allocated/reserved memory, and generated
  token-forward count;
- route-density buckets K=64 and K=128, plus their observed mixture.

Sparse speedup is a measured end-to-end claim. Sequence-length arithmetic alone
is not sufficient evidence.

## Statistical and reporting rules

- preregister seeds and all stopping/exclusion rules;
- never drop failed seeds without reporting the failure and reason;
- keep W&B histories and immutable run/generation manifests;
- disclose that E117 consumes generated 1D codes at deployment while training
  routes were cached from ground-truth 1D codes;
- include an ablation for generated-1D route distribution shift if final
  quality is sensitive;
- label the current 40k numbers as preliminary feasibility diagnostics.
