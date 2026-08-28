# Paper experiment protocol

This document separates the registered experiment from exploratory pilot
evidence. Following it makes the MoTAR result reproducible and suitable for a
paper table; it does not by itself establish novelty or a state-of-the-art
claim.

## Method under test

Let the unified trunk be `f_theta`, the disjoint 1D sidecar be `g_phi`, TiTok
targets be `z_1d`, and LlamaGen targets be `z_2d`. The trunk objective is

```text
L_main(theta) = 1.5 CE(f_theta^1d, z_1d) + CE(f_theta^2d, z_2d).
```

The sidecar receives the same class and causal TiTok prefix embeddings, but the
input and trunk logits are detached. Its objective is

```text
L_side(phi) = CE(stopgrad(f_theta^1d) + g_phi(stopgrad(prefix)), z_1d).
```

The two models have separate DDP wrappers, AdamW states, backward passes,
gradient clipping, and schedulers. Consequently `L_side` cannot update
`theta`. The sidecar output matrix is zero-initialized, so the method is
exactly the matched trunk at initialization. During 1D decoding, trunk and
sidecar logits are added; the sidecar is not used for 2D logits.

The registered method is depth 4, width 1024, sidecar LR `2e-4`, trunk
base/new LRs `1e-4`/`2e-4`, 1D/2D loss weights `1.5`/`1.0`, bf16, and
150 epochs. Do not tune these values using the reported test set.

## Primary hypothesis and endpoints

The primary hypothesis is that the disjoint sidecar reduces TiTok 1D
teacher-forced cross-entropy without degrading 2D cross-entropy under an
otherwise matched trunk update.

Primary endpoints:

1. TiTok 1D cross-entropy on a held-out ImageNet split.
2. LlamaGen 2D cross-entropy on the same held-out images with fixed random order.
3. Class-conditional AR generation FID-50k, computed with the same decoder,
   guidance, sampling schedule, seed policy, and reference statistics.

Secondary endpoints are 1D/2D top-1 and top-5 accuracy, per-position CE,
training throughput, sampling latency, peak memory, trainable parameters, and
FLOPs or a clearly specified measured-compute proxy.

The fixed first 8,192 packed training samples are a development diagnostic, not
the paper test set. Tokenizer reconstruction FID `1.3032` is not AR generation
FID and must never be reported as such.

## Matched control

Use `configs/h200_8gpu_150epoch_matched_baseline.yaml`. It matches every main
model, data, optimizer, scheduler, seed, precision, and epoch field in the
disjoint configuration. Use the exact per-GPU micro-batch selected for the
sidecar run:

```bash
export MATCHED_MICRO_BATCH_SIZE="$(<"$RESULTS_DIR/motar_titok_l32_mot199440_disjoint_150ep_h200/h200_micro_batch_size.txt")"
bash scripts/launch_h200_matched_baseline.sh
```

Do not use `configs/h200_8gpu_150epoch.yaml` as the paper control: it preserves
an older LR-scaling experiment and is not optimizer-matched to the sidecar run.

Because sidecar construction restores the CPU RNG and sidecar training is in a
forked RNG context, the main model initialization and stochastic stream are
designed to match the control. Record nondeterministic kernel settings and
verify the main-only curves empirically rather than assuming bitwise identity
across H200 runs.

## Replication and reporting

- Run at least three predeclared seeds for the method and matched control.
- Report mean, standard deviation, every individual seed, and the exact sample
  count. Use paired seed differences when initialization and data order match.
- Select checkpoints by a validation rule declared before viewing test metrics;
  do not select using test FID. The current latest-only 150-epoch run is valid
  only if epoch 150 is the predeclared endpoint.
- Preserve `resolved_config.yaml`, `run_plan.json`, checkpoint metadata, W&B
  run ID, Git commit, packed-data manifest, package versions, GPU model, and
  batch size for every run.
- Report failed and neutral ablations, including shared and partially isolated
  sidecars, not only the selected depth-4 result.
- Clearly label the 2,000-step, first-8,192-sample numbers as pilot/development
  evidence. They are not an independent test result.

## Required ablations and fairness checks

At minimum compare:

1. matched 24-layer unified trunk;
2. 24-layer trunk plus the depth-4 disjoint sidecar;
3. a parameter- or compute-matched stronger trunk/control;
4. sidecar depth and 1D-weight ablations selected only on validation data;
5. sidecar residual disabled at inference, isolating whether the gain comes
   from the new path;
6. equal sampling settings and equal decoded-token budget for generation.

The sidecar adds parameters and compute, so comparison to the 24-layer trunk
alone is insufficient for an efficiency or architectural-superiority claim.
Report trainable parameters, training tokens/s, peak H200 memory, sampling
images/s, and end-to-end latency.

## Held-out code extraction

The packer and validator accept arbitrary nonempty ImageFolder sample counts;
they are not hard-coded to the 1,281,167-image training split. Reuse the same
verified tokenizer assets and extract ImageNet validation codes into a separate
directory:

```bash
export ASSET_ROOT=/persistent/assets/motar-extraction
export IMAGENET_TRAIN=/persistent/datasets/imagenet/val
export PACKED_CODE_ROOT=/persistent/data/imagenet-val-titok_l32-mot199440ema-adm-256_packed
bash scripts/launch_extract_h200.sh
```

Keep this directory read-only after its manifest is recorded. Use augmentation
0 for the primary held-out report. Do not use held-out metrics to tune sidecar
depth, loss weights, learning rates, sampling temperature, guidance, or
checkpoint choice.

## Fixed development diagnostics

Run both checkpoints with `augmentation=0`, `num_samples=8192`, and the same
2D order seed. The 1D commands are in the README. For 2D:

```bash
torchrun --standalone --nproc_per_node=8 evaluate_2d.py \
  --checkpoint-kind disjoint \
  --config configs/h200_8gpu_150epoch_disjoint.yaml \
  --checkpoint-dir "$RESULTS_DIR/$EXP_NAME/checkpoints/latest" \
  --packed-code-root "$PACKED_CODE_ROOT" \
  --output-json "$RESULTS_DIR/$EXP_NAME/eval/diagnostic_2d_8192.json" \
  --num-samples 8192 --augmentation 0 --order-seed 20260828
```

`evaluate_2d.py` records overall, generation-step, and 16x16 spatial-position
CE/top-1/top-5 metrics. It validates the appropriate checkpoint format and puts
the checkpoint metadata into the JSON report.

## Claim boundary

Until the matched multi-seed runs and FID-50k evaluation are complete, the
supported statement is only: "a depth-4 disjoint residual sidecar improved
early fixed-subset 1D teacher-forced metrics without worsening the observed 2D
teacher-forced metric in a 2,000-step pilot." Do not claim better generation,
generalization, efficiency, statistical significance, novelty, or SOTA from
the pilot.
