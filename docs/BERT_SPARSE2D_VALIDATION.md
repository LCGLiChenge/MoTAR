# BERT sparse2D delivery validation — 2026-09-15

Scope: packaging the existing TiTok-initialized BERT **2D-only** experiment, not
a new unified architecture, new FID result, or an eight-GPU long-training claim.

- Seven CPU unit tests: padding invariance, sparse permutation equivariance,
  full-grid feature dependence, CFG/gradient flow, recomputation, exact next
  Adam update after restore, and data-cursor continuity.
- Two-rank Gloo synthetic trainer: fresh updates 1–2, save, restore updates 3–4.
  `latest.pt` strict safetensors readback passed; 180 tensors, 13,698,520 bytes.
  The second save replaced the first; no numbered/best weights were created.
  Both CPU-smoke checkpoints were deleted after verification (27,397,040 bytes);
  small hash/config/resume audit records remain. They are not retained assets.
- Comparison against the unchanged local experiment: matching tiny-model state
  names/values, exact forward outputs and parameter gradients; sampling
  `generate` AST unchanged. Tiny-model parity is an implementation check, not
  a substitute for evaluating the full trained model.
- ADM `Evaluator`, `FIDStatistics`, and `DistanceBlock` definitions match the
  upstream OpenAI implementation structurally. Vendored local CLI and unused
  precision/recall helper differences are documented in `bert2d/vendor/SOURCE.md`.
- Clean-checkout full model initialization loaded 389 official TiTok tensors
  (170,880,768 parameters) exactly; total trainable parameters 196,459,264.
  Frozen assets loaded strictly and CPU feature extraction returned finite
  `[1,256,16,16]` features. All 23 HF transport files/parts across four pinned
  repo revisions were found publicly with matching advertised sizes.
- Clean MoTAR checkout is tested separately from the local experimental source
  tree. Only the allowlisted package/docs/launcher/dependency files are published.

On the remote machine, the launch command still must pass asset hashes,
worst-case K128 GPU memory probing and actual distributed checkpoint + resume
smokes before starting fresh formal training. GPU driver/runtime availability
and eight-GPU memory headroom cannot be certified by CPU checks on this machine.

The local 4k→8k continuation was not modified or restarted by this publication.
There are no new weights to upload: the fresh initialization and frozen assets
are already public and listed in the download manifest.

## Periodic FID update

- Four added CPU orchestration tests cover exactly20 evaluations at epochs2..40,
  latest-only retention, recovery after failed eval, reuse of completed results,
  invalid-metric rejection and same-W&B-run upload without a duplicate explicit
  history step. W&B calls are mocked; this is not evidence of a real cloud upload.
- Existing seven model/resume tests are retained. Real GPU FID computation is
  unchanged; the sharded summary now records its checkpoint SHA-256.
- Target-server startup checks FID assets and TensorFlow GPU availability before
  training. No additional local GPU experiment is started for this packaging.
- Two-rank Gloo integration: continuous8 updates versus4 updates + full-state
  resume to8. All180 saved model/Adam/per-rank RNG tensors match exactly; both
  checkpoints have SHA-256 `60b0c19f78fce655bbcc136ba51faa8491e0ed30486f0261add440f2ddcef73a`.
  Smoke weights were removed after verification; metadata and cleanup manifest remain.

## Per-epoch asynchronous update

See [asynchronous validation](BERT_ASYNC_EVAL_VALIDATION.md) for the current
per-epoch concurrent implementation. The serialized-every-two-epochs section
above describes the earlier release, not the current launcher default.
