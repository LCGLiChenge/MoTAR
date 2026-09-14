# Latest handoff override — BERT sparse 2D-only, 40 epochs (2026-09-15)

This section supersedes all older default-launch instructions below for the
current user request. Read README.md, docs/BERT_SPARSE2D_40EPOCH.md and
 docs/BERT_SPARSE2D_VALIDATION.md before acting.

- Entrypoint: scripts/launch_bert_sparse2d_40epoch.sh. Default 40 PACKED epochs,
  global batch448. Do not launch the older spatial-joint/unified/80-epoch scripts.
- One BERT generating only sparse2D; full256 frozen base-feature context;
  official TiTok encoder/class initialization, fresh2D embedding/head.
  Keep direct replacement; no half blending, fusion, Halton initialization,
  or local probe checkpoint resume unless explicitly requested.
- Large assets/results stay outside the Git checkout, on the user's data disk:
  MOTAR_ASSETS and MOTAR_RESULTS. Downloads: python -m bert2d.assets --fid.
  No unpublished files or val cache are required; do not download legacy resume.
- Explicit GPU allocation only; launch supports1/2/4/8, but hardware support is
  not authorization. Packaging does not authorize a new local long training run.
- Preserve global batch while auto-probing microbatch; explain any explicit
  global-batch change. Run the launcher memory and save/resume gates before
  formal training. No silent NaN/OOM bypass. Keep20GiB free startup headroom.
- W&B online scalar training curves; no credentials in Git and no log.txt.
- Keep only latest.pt per formal run, atomically updated every epoch/end/signal,
  plus small JSON metadata. No numbered/best checkpoints. Verified smoke weights
  are deleted after saving audit metadata. Source assets must not be modified.
- Evaluate stable latest with bert2d.eval_sharded, same 5k/50k protocol, raw state,
  full direct replacement and frozen official TiTok prefix. Never average shard
  FIDs or compare different sample counts as a controlled improvement.
- This is NOT a unified model and NOT proven to outperform pure1D. Do not infer
  paper-ready success from packaging tests. See the validation limitations.

---

# Legacy agent instructions — apply only to explicitly requested legacy runs

# Codex handoff — spatial MaskGIT + online fusion on H20

## Current default: spatial joint v1

Read README.md, docs/H20_SPATIAL_JOINT.md, configs/h20_assets.json and
docs/H20_JOINT_VALIDATION.md completely before launching this version.
The new authoritative entry is scripts/launch_spatial_joint_h20.sh.
Everything under the legacy heading below applies ONLY when the user explicitly
requests the old no-spatial baseline. Never use its resume default for the new run.

- Default: official TiTok initialization, fresh2D/spatial/fusion, --memory local,
  --epochs80. Not all-random scratch and not a local pilot/legacy resume.
- Assets: python -m h20.assets --root "$MOTAR_ASSETS" --profile joint.
  This includes train+inference; old step10938 resume assets are NOT needed.
- Explicit GPU allocation, supported1/2/4/8; never occupy another user's GPU.
  No local long training is authorized by packaging or smoke instructions.
- Use the launcher, not a direct formal h20_joint.train invocation: it verifies
  versions/assets/data/devices, probes actual memory, then performs a4-update
  DDP save smoke and a resume-to5 smoke. All must pass before formal training.
  The smoke starts independently; formal parameters are reset to official init.
- CE global batch2048 and fusion global batch16 have INDEPENDENT accumulation.
  Adjust only micro/accumulation to use memory, default ceiling92%. Do not change
  global batch or objective weights to fill memory. Never silently skip OOM/NaN.
- W&B online required, no secrets in code/config/git; no log.txt or TensorBoard.
- Only each run's latest.safetensors/latest.json, every source-equivalent epoch
  and finish/signal. Smoke is a separate run with its own latest. Keep10GiB save
  headroom; launcher requires20GiB before capacity/smoke. Never change source
  checkpoints/caches or weaken the guard to fit a local disk.
- --in-memory-smoke is a bounded developer test; it saves NO checkpoint and
  cannot satisfy the H20 launcher's save/resume gate. Do not call it a full-state
  resume test. Local validation used this because local disk was insufficient.
- CE2D still uses real packed1D; generated1D/2D feed online fusion MSE only.
  Hard sampling means the image loss does NOT backpropagate into MaskGIT.
  Frozen nativeTiTok/MoT/E117 stay frozen;1D forward never reads2D features.
- --memory full is a separate unproven ablation, not the5.440 architecture.
  Do not substitute it silently or restore a local-memory checkpoint into it.
- Published5.440 was worse than paired half5.231 and pure1D4.803. New joint
  long training has NO FID result yet. Preserve negative controls and distinguish
  teacher MSE, reconstruction rFID and generation FID. Smoke is not quality proof.

## Legacy no-spatial baseline ONLY

Read README_BASELINE_H20.md completely, then configs/h20_assets.json and
docs/H20_HANDOFF_VALIDATION.md. The legacy baseline launcher is
scripts/launch_titok_bert_h20.sh. Older H200/LLaMA scratch and causal AR launchers
are historical and must not be substituted.

## Start-up

1. Check the user's GPU allocation; set CUDA_VISIBLE_DEVICES explicitly.
   Supported world sizes are 1, 2, 4, 8. Never claim unallocated GPUs.
2. Create a separate environment with scripts/install_h20_environment.sh,
   or --active-env in a dedicated Python 3.10 environment. Do not alter system drivers.
   For separate shell calls, prefix commands with
   `conda run --no-capture-output -n motar-h20`; do not rely on prior-shell activation.
3. Set MOTAR_ASSETS, download with `python -m h20.assets --root "$MOTAR_ASSETS" --profile resume`.
   Use --profile all if decoder/tokenizer/E117 weights are also wanted.
4. Authenticate W&B interactively; never write credentials into files or commits.
5. Run `python -m h20.preflight --assets "$MOTAR_ASSETS"`.
6. Set a fresh MOTAR_OUTPUT and launch with `--init resume --steps 50000` unless
   the user explicitly chooses official initialization. Re-running an existing
   run resumes that output's own latest. Changed layout requires a new output.

The published checkpoint is step10938, global batch2048, warmup600, micro128,
four ranks, accumulation4. The model is initialized from the official TiTok
generator, NOT from the old random LLaMA checkpoint. Keep the current clean-1D
prefix objective, shared BERT, LR groups and EMA unchanged during migration.
The initial total target is 50000 steps, not 50000 additional steps or the
legacy 800-epoch budget. Longer budgets must be explicitly selected.

## Memory and safety

Always run the real microbatch probe; preserve effective global batch2048 via
accumulation. The default reserved-memory ceiling is92%, and eight GPUs cap
microbatch at256. Do not increase global batch just to fill memory.
Do not bypass failed hash, data, version or device checks. No NaN/OOM retries
that skip a batch or silently change the experiment. Never kill another job.
No long local training is part of packaging/testing. SIGINT/SIGTERM of an
owned run is handled at optimizer boundaries with latest saving.

## Evidence and checkpoints

W&B online required. No log.txt/TensorBoard. Only latest.safetensors and its
latest.json metadata; save per source-equivalent epoch and on normal finish.
Do not edit source checkpoints or caches. Preserve at least10GiB free disk.
Resume restores raw/EMA/AdamW, not only EMA. Changed DDP layout starts the next
packed pass with new rank RNG; disclose non-bitwise migration. Even unchanged
layout cannot promise bitwise identical updates on a different GPU architecture.

No ImageNet raw images are needed for training. All training data, route caches
and checkpoints are published/pinned in configs/h20_assets.json. Do not infer
an old server path in metadata is a dependency. Real-image reconstruction FID
is not generation FID, and a successful smoke is not a quality claim.
