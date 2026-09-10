# Codex handoff — current TiTok-BERT unified MaskGIT on H20

Read README.md completely, then configs/h20_assets.json and
docs/H20_HANDOFF_VALIDATION.md. The current authoritative launcher is
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
