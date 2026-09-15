# Per-epoch asynchronous evaluation: validation scope

Change: the 40-packed-epoch BERT sparse2D launcher defaults to one background
paired 5k FID per epoch. Global batch448, model, optimizer recipe, sampling,
FID/IS calculations and latest-only checkpoint format are unchanged.

## Mechanism

- Save latest at the epoch boundary and start one shard on each allocated GPU.
- Require every shard to acknowledge the exact SHA-256, step and raw state after
  copying its model into independent storage. Only then resume training.
- Reference-asset hash verification remains mandatory, but runs after this
  acknowledgment instead of extending the training pause.
- One evaluation at a time, with backpressure, failure propagation to all
  trainer ranks, final drain, and termination of owned subprocesses on failure.
- Trainer rank0 alone writes W&B. Delayed FID uses eval/epoch and
  eval/checkpoint_step, not the current training step/epoch.

## Checks

All 32 CPU tests passed in the clean release checkout. Shell syntax and diff
whitespace checks passed. The CPU suite tests model/resume behavior and evaluator lifecycle: all-shard
acknowledgment, checkpoint identity after latest changes, wrong-hash rejection,
timeouts/failure, result recovery, duplicate suppression, no snapshots, W&B axes,
and per-epoch cadence. W&B calls are mocked; these tests do not prove a cloud
upload or eight-H20 concurrency.

Run from a clean checkout in the installed environment:

```bash
USE_TF=0 python -m unittest bert2d.test_model bert2d.test_resume \
  bert2d.test_periodic_eval bert2d.test_optimization bert2d.test_async_eval
bash -n scripts/launch_bert_sparse2d_40epoch.sh
```

The first local same-GPU smoke timed out during startup before shard loading.
No quality metric was produced and no formal checkpoint was modified. Inspection
found FID reference validation ahead of the loading handshake; that validation
was moved after acknowledgment without disabling it. The machine also had a
load average above 100 and processes waiting on file reads. This first failure
is retained rather than reported as a passed concurrency test.

## Target-server limits

The default 12 GiB/GPU reservation is headroom, not a guaranteed measured H20
peak. Startup still checks actual training capacity. Same-GPU eval competes for
compute; asynchronous execution does not imply zero training slowdown. Defaults
allow 600 seconds to load and 1800 seconds total per eval; slower servers must
explicitly increase these limits. No new local long training was started.

Use EVAL_MODE=serial only as an explicit non-overlapping fallback. Existing
processes must exit before pulling/resuming the same output. Older overwritten
latest checkpoints cannot be evaluated retroactively.

## Local GPU follow-up (partial verification, not a full pass)

One RTX5090 held the full trainable BERT, frozen feature provider and Adam while
an independent evaluator loaded step8000. After its acknowledgment, ONLY the
smoke's disposable latest links were changed to step6000. The evaluator still
reported step8000 and completed all32 images with GPU-verified ADM features.
The test executed eight finite-loss optimizer updates and reached its final
worker wait. This verifies model residency and fixed checkpoint identity;
it is not a measured H20 throughput benchmark.

The test was interrupted during the global merge after slow disk loading and
TensorFlow PTX compilation warnings on CC12.0. No final merged summary/FID or
real W&B upload was completed; do not call it an end-to-end concurrency pass.
All owned processes exited. Smoke arrays and disposable links were deleted after
recording hashes; source checkpoints were untouched. Local audit:
`results/bert_async_eval_20260915/verification.json` under the approved data root.
Eight-H20 concurrent capacity and cloud logging remain target-server checks.
