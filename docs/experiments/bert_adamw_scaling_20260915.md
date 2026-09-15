# AdamW reference-batch scaling implementation audit

## Material Passport

- Date: 2026-09-15
- Origin: user-authorized code change; academic-research-suite used to check
  scaling assumptions and distinguish implementation checks from FID evidence.
- Version: adamw-sde-v1, reference batch 448
- Verification: implementation checks passed; H20 quality/efficiency UNVERIFIED
- AI-assisted implementation and source checking; no new FID claim.

## Scope

Added joint LR/betas/epsilon/decoupled-decay scaling and a sample-based warmup
clock. No model, loss, dataset, frozen decoder, direct-replacement, or FID
sampling changes. `legacy` explicitly retains the historical optimizer recipe.
No existing experimental checkpoint was modified, migrated, or deleted.

See [the protocol and exact H20 commands](../BERT_BATCH_SCALING.md).

## Checks executed locally

- 20 CPU unit tests: model, old resume, periodic FID, bounded one-epoch stop,
  theoretical formulas, invalid beta handling, reference LR transport,
  B448 bit-exact old-AdamW parity over 124 updates, and scaled Adam resume.
- Actual trainer CPU/Gloo integration: uninterrupted four updates versus two
  updates + full-state resume to four. All **178 tensors** matched exactly,
  including raw weights, Adam states and RNG. Cursor and optimizer groups matched.
  Tiny synthetic fixture (B8), not full-size training quality evidence.
- Real 196M BERT + official TiTok initialization + frozen feature provider,
  GPU0 RTX5090, bf16, K128, micro2, three post-warmup updates using the **B3200
  resolved optimizer**: finite loss/gradient checks passed, peak torch reserved
  **5,324 MiB**. This is a micro2 numerical smoke, NOT a measured B3200 training
  run and NOT proof that micro400 fits H20.
- Read-only launcher optimization preview and shell syntax checks passed.
- External DeepSeek helper was attempted as requested by repository guidance,
  but unavailable (`DEEPSEEK_API_KEY` unset); verification was performed locally.

Local evidence (not a runtime dependency of the published code):

```text
/mnt/data/heyefei/lichenge/Mixture-of-Tokenizer/AR/results/bert_adamw_scaling_20260915/
  gpu_probe_b3200_micro2.json
  cpu_resume_p__fq0bq/verification.json
```

The two CPU smoke checkpoints (13,693,312 bytes each) were removed only after
exact comparison and a cleanup manifest with hashes was saved. The GPU smoke
did not write a checkpoint. Small configurations/metrics/audit records remain.

## Not yet established

- Better full-refine 5k/50k FID, better sample efficiency or GPU-hour efficiency.
- Long-run stability of beta1~0.286 and the seven-update fresh warmup at B3200.
- Full-scale H20 multi-GPU capacity or numerical behavior.
- Optimizer-state migration from an old large-batch run into this new recipe.

Published test commands must be run on allocated H20 GPUs. The test wrapper is
bounded to one packed epoch plus one paired 5k FID, using separate new outputs
for B448 and B3200. It never automatically starts 40 epochs.
