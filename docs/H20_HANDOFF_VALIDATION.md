# H20 handoff validation — 2026-09-10

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Verification Status: UNVERIFIED on target H20; local tests below completed
- Version Label: titok-bert-h20-handoff-v1

## Scope

Portable training handoff, not a model-quality improvement experiment. The
model equations, official TiTok parameter mapping, arccos masking, per-sample
objective, 1D weight1.5, optimizer LR groups and EMA were retained. Source files
are independently located in h20/; no existing training code/checkpoint/cache
was edited. Old launchers and documentation remain archived.

Changes are asset-relative paths, pinned dependencies and checksums, variable
explicit GPU allocation, global-batch-preserving capacity probing, tested
resume, W&B and durable latest output. The pilot's three-hour deadline and
NLL-only retention stop are not enabled in the formal handoff.

## Completed local checks

Eight CPU unit tests passed (11.35 seconds):

- official 1D logits and load accounting, eager and SDPA, conditional/null;
- invalid official checkpoints rejected without mutating the target;
- official sampler RNG adapter;
- sparse padding and strict 1D isolation from 2D inputs/parameters;
- incompatible architecture rejection;
- candidate batches preserve global batch2048 for 1/2/4/8 ranks;
- same-layout cursor versus disclosed changed-layout continuation;
- full raw/EMA/AdamW round trip, including exact equality after the next AdamW update.

Four-GPU startup smoke completed from the new standalone checkout:

- automatic real K128 memory probe passed;
- four steps, two new-2D-only warmup plus two joint steps;
- 32 source/augmentation pairs verified disjoint across all ranks/updates;
- no missing/nonfinite gradients, checkpoint readback exact;
- 1,998 saved tensors, 3,241,625,288 bytes;
- runtime24.84s for the distributed training portion, excluding preflight/probe;
- [W&B ginl1qmt](https://wandb.ai/lcg-li-chenge-zhejiang-university/motar-maskgit/runs/ginl1qmt).

Real-checkpoint continuation smoke completed:

- restored the published step10938 checkpoint, raw/EMA/AdamW/RNG and saved cursor;
- same four-rank layout, micro128, accumulation4, global batch2048;
- exactly one optimizer update to10939; 2,048 disjoint source/augmentation pairs;
- finite loss and gradient; 1,998 tensors saved/read back exactly;
- peak GPU reserved memory29.87GiB; runtime52.93s;
- [W&B mgn5dl9x](https://wandb.ai/lcg-li-chenge-zhejiang-university/motar-maskgit/runs/mgn5dl9x).

These are smoke checkpoints, not new published training results. The asset
to resume remains the original step10938 checkpoint, SHA256
`1d680436cb401d05a2dd5de45174140e16db0d72670417f526073f5bf194b28c`.
Both test jobs exited0 and all four GPUs were released. No formal long run
was started. Tests used RTX5090, not H20.

Environment preflight passed with torch2.10.0+cu128 and the specified
Transformers Git commit. A 66-package transitive dependency constraint file is
included. The existing broad local environment has an unrelated ninja wheel
platform warning from pip check; ninja is not part of the new environment's
runtime closure. No claim is made that the old broad environment itself is a
clean install. Shell syntax and all new Python modules were checked.
The pinned Transformers commit was fetched successfully and the full
requirements/constraints pip dry-run resolved without dependency errors. This
used the existing environment, not an independently installed H20 environment.

## Remaining target-side checks

No H20 host credentials were provided, so target hardware capacity and a
complete clean Conda installation on that host cannot be verified locally.
The README commands explicitly install/check versions, verify every asset,
check allocated devices are free, and run the real capacity probe before
training. Do not label the RTX5090 smoke as H20 validation.

The cross-layout path restores training state but intentionally starts the
next packed pass with fresh rank RNG. It is not bitwise replay. Same-layout
RNG/cursor restoration also does not guarantee identical GPU kernels across
architectures. Full multi-seed generation quality is outside this handoff.

The project-requested DeepSeek helper was attempted with a non-sensitive
checklist; DEEPSEEK_API_KEY was not configured. No external review is claimed;
the main agent performed checks and verified tests directly.

## Publication gate

All assets must have immutable revisions in configs/h20_assets.json before
the GitHub push. A stalled Xet upload was terminated by its identified own
PID and retried through the standard Hugging Face LFS upload path. No source
file or checkpoint was changed to retry the upload. Remote object sizes and
hashes, download path and final GitHub commit are recorded after completion.
Because ordinary monolithic LFS transfer was also very slow, the two largest
files were transport-sharded into128MiB pieces with12 bounded upload threads.
The downloader validates each piece and the reconstructed full-file SHA256;
the model/checkpoint bytes are not converted. A dedicated CPU test verifies
byte-exact assembly and rejection of a wrong final checksum.

## Completed remote asset verification

Asset publication succeeded at Hugging Face commit
`1af3500c0ef4542884c5c9d7aeadedbb33584d3e` in `Chloeeeeeeee123/MoT-1`.
All 27 logical assets (60 physical objects including transport pieces) were
checked anonymously against pinned revisions: object sizes and LFS SHA256,
or freshly downloaded SHA256 for regular small files. All repositories are
public and non-gated. No old-host Hugging Face credential is needed.

Two non-metadata objects were force-downloaded anonymously and hashed:
validation TiTok codes (3,200,128 bytes) and checkpoint part00024
(20,399,816 bytes). Both match their registered SHA256. All27 final local
logical files also passed the published downloader's --verify-only path.
This verifies publication and sample download, not a second full15GB download.

The final CPU suite, including transport assembly, passed all9 tests in1.61s.
The final source-only staging check found no credential patterns, accidental
large files, or original-server absolute dependencies in h20/. Imports were
confirmed to resolve to bundled TiTok/RandAR code. Owned code passes diff
whitespace checks; original upstream whitespace in vendored TiTok is retained.
