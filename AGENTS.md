# AGENTS.md

## Current objective

Run the E117-routed unified TiTok-L32 1D + MoT/LlamaGen VQ-16 sparse-2D
MaskGIT experiment from random initialization on exactly eight NVIDIA H200s.
The formal budget is 1,024,000,000 examples, matching TiTok's 500k-step,
global-batch-2048 generator training.

Do not substitute the older 150-epoch causal-AR launcher. Its documentation is
kept only in `docs/LEGACY_CAUSAL_AR.md`.

## Authoritative launch

Read `README.md` and `docs/DATA_AND_CHECKPOINTS.md` first. Then use:

```bash
PACKED_CODE_ROOT=/persistent/data/train-codes \
EVAL_PACKED_CODE_ROOT="$PWD/data/imagenet-val-titok_l32-mot199440ema-none-256_packed" \
E117_ROUTE_CACHE=/persistent/data/train-e117-routes \
E117_EVAL_ROUTE_CACHE=/persistent/data/val-e117-routes \
RESULTS_DIR=/persistent/results/motar \
WANDB_PROJECT=motar-maskgit \
WANDB_ENTITY=your-entity \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash scripts/launch_e117_maskgit_h200.sh
```

## Invariants

1. Formal training uses exactly eight H200 processes and bf16.
2. Model is 24 layers, width 768, pre-norm, explicitly bidirectional.
3. Start the generator from scratch; never import an RTX 5090/AR/RandAR
   generator checkpoint. Native resume may use only this run's `latest.pt`.
4. Keep TiTok arccos masking, label smoothing 0.1, visible-token weight 0.1,
   class dropout 0.1, loss weights 1D=1.5 and 2D=1.0.
5. Keep AdamW LR 1e-4, betas 0.9/0.96, weight decay 0.03, and full-model EMA
   0.999. The launcher adjusts only batch-dependent step counts.
6. Preserve 1,024,000,000 target exposures. Rounding error may be at most half
   one selected global batch.
7. Use the exact registered E117 route caches. Their metadata checkpoint hash
   must be
   `a5b84689d2b29f579d2442da7594ac093292b6386760867a0668ca02f82e6156`.
8. W&B is required unless offline mode is deliberately selected. Do not create
   `train.log` or TensorBoard logs.
9. Save only `latest.pt`, atomically, after every completed data epoch.
   No step, best, epoch, or final checkpoints.
10. Never edit packed arrays or route caches and never commit weights, train
    data, generated images, or result directories. The immutable validation
    cache already tracked under `data/` is the sole data exception.
11. Never signal another user's process. On an abnormality, SIGINT only the
    owned launcher parent after identifying it.

## H200 batch procedure

The launcher probes the exact K=128 two-branch train step on one H200. It keeps
the largest isolated-process candidate below 90% peak reserved memory, then
launches eight ranks. Do not bypass the probe. With real DDP OOM or co-tenancy,
stop cleanly and repeat with `H200_MEMORY_FRACTION=0.85`.

`run_plan.json` and `resolved_config.yaml` are immutable once a run
starts. Use a new `EXP_NAME` for a changed experiment.

## Monitoring

Monitor W&B total/1D/2D loss, top-1/top-5, both mask ratios, gradient norm, LR,
throughput, source-equivalent epoch, and fixed-eval context/prefix deltas.
Also monitor all eight ranks, memory, temperature, and unexpected co-tenancy.

Stop the owned run for NaN/Inf, an unrecoverable OOM after the probe, missing
ranks, a sustained collective stall, corrupted data, or a persistent regression
confirmed by fixed evaluation. Do not stop on a noisy single training batch.

After each completed data epoch, validate:

```bash
python scripts/validate_e117_maskgit_latest.py \
  "${RESULTS_DIR}/${EXP_NAME}" --expected-world-size 8
```

## External route caches

The registered train and validation E117 route caches are published in
`Chloeeeeeeee123/MoT-1`. Download the pinned revision and verify both
SHA256 manifests exactly as shown in `README.md` before launch. Packed codes
can be regenerated with the two extraction launchers. Do not silently generate
routes with another E117 checkpoint.

## Evidence discipline

The step-40k local result is a feasibility result, not final generative quality.
Follow `docs/MASKGIT_EXPERIMENT_PROTOCOL.md` for final sampling sweeps,
multi-seed evaluation, 50k FID, throughput, and memory claims.
