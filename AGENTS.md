# AGENTS.md

## Objective

Run the unified TiTok-L32 1D + MoT/LlamaGen VQ-16 2D AR experiment on exactly
8 H200 GPUs until the checkpoint metadata reports `completed_epochs: 150`.

## Authoritative launch

```bash
PACKED_CODE_ROOT=/persistent/data/codes \
RESULTS_DIR=/persistent/results/motar \
bash scripts/launch_h200.sh
```

The launcher validates data and devices, probes H200 memory, selects the batch,
and resumes `checkpoints/latest` automatically after the first H200 epoch.

## Invariants

1. Production training uses exactly 8 H200 processes.
2. Total training target is 150 epochs.
3. Keep `loss_1d_weight: 1.5` and `loss_2d_weight: 1.0`.
4. Keep bf16, gradient checkpointing, and the 24-layer/1024-wide model.
5. Do not load any prior AR or RandAR initialization checkpoint. The first H200
   launch is random initialization; only its own later `latest` may be resumed.
6. Do not create versioned, best, step, or final checkpoints.
7. Update only `checkpoints/latest` after every epoch.
8. Do not edit or overwrite the packed arrays.
9. Do not commit large data, model, optimizer, or generated result files.
10. Never signal another user's process.

## Codex operating procedure

1. Read `README.md` and `docs/DATA_AND_CHECKPOINTS.md`.
2. Locate or request the packed-code path. Never invent it, and never request
   an old AR checkpoint for this run.
3. Confirm persistent `RESULTS_DIR` has enough free space for one model and
   optimizer checkpoint plus a transactional copy during save.
4. Run the validation command before launch.
5. Start `scripts/launch_h200.sh` in a persistent terminal/session.
6. Record the selected micro/global batch and scaled learning rates from
   `run_plan.json`.
7. Monitor loss, both modality losses/accuracies, gradient norm, throughput,
   eight GPU processes, memory, temperature, and co-tenancy.
8. On a real abnormality, send SIGINT only to the owned torchrun parent and
   inspect the last complete `latest`.
9. After every epoch, run or inspect `scripts/validate_latest.py`; confirm no
   non-hidden checkpoint sibling exists.
10. Completion requires `metadata.json` to show `completed_epochs: 150` and
    the latest checkpoint to pass validation.

## Batch policy

Default batch selection is an actual forward/backward/AdamW probe capped at 90%
peak reserved memory. If DDP overhead still causes OOM, stop cleanly and rerun
with `H200_MEMORY_FRACTION=0.85`. Use a manually fixed
`MICRO_BATCH_SIZE` only when supported by a successful probe.

Changing batch changes the learning rate through the documented square-root
rule. Do not silently change the rule during a run.

## Checkpoint recovery

Stable state:

```text
checkpoints/latest
```

The hidden `.latest-next` and `.latest-previous` directories are
transactional only. Do not treat them as experiment checkpoints. The trainer's
recovery helper resolves an interrupted rotation before resume.
