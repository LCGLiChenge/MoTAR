# Dense proxy MaskGIT:8H20,6000→20000

## Scope

2D-only, NOT unified or the older sparse continuous-context BERT. Frozen official
TiTok generates32 codes; frozen MoT EMA decodes16x16 features. Nearest projected
2D codewords become fixed anchors. One24-layer768-wide BERT sees
`[class | grid0..255]`, generating only E117-selected positions(K64/128).
Generated2D features directly replace those positions. Unselected features stay
continuous for final decoding. No half blend or learned/GAN converter.

Source6000 checkpoint includes raw model, Adam, per-rank RNG and data cursor;
196260352 trainable parameters. Full5k FID:2000=62.7794,3000=55.1031,
4000=54.3367,6000=50.3377; paired base≈10.8671. Not yet successful generation;
no claim20k will beat base, and no50k result.

## Run on the allocated8-H20 machine

Activate the existing compatible environment. Known versions: Python3.10,
torch2.10.0+cu128, transformers4.50.0.dev0(pinned existing requirements).
GPU TensorFlow is required by ADM-FID. Do not modify drivers/system Python.

```bash
git clone https://github.com/LCGLiChenge/MoTAR.git
cd MoTAR
# Only if missing in the intended environment:
# python -m pip install -r requirements-bert2d-eval.txt
export MOTAR_ASSETS=/YOUR/DATA/DISK/MoTAR_assets
export MOTAR_RESULTS=/YOUR/DATA/DISK/dense_proxy_grid_h20
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Existing verified assets can be reused; otherwise download:
python -m bert2d.assets --root "$MOTAR_ASSETS" --fid

# Run inside tmux; no log.txt redirect.
bash scripts/launch_dense_proxy_grid_h20_20000.sh
```

The launcher runs CPU layout/sampling tests, verifies assets and downloads the
source6000 release. It can wait up to6h for HF publication without CUDA/GPU
allocation. Resolve one HF commit once, download all3 files at that revision,
record revision in download.json and verify the pinned SHA256. Never use
torch.load: latest.pt contains safetensors. Source old local paths in metadata
are provenance, not required external files; the loader remaps assets.

New dependency: `Chloeeeeeeee123/MoT-1`, directory
`checkpoints/dense_proxy_grid_maskgit_step6000_20260916/` containing
latest.pt/latest.json/config.json. Public download needs no credentials.
latest.pt bytes2355340260, SHA256:
`0a3acbe2e185b138f242a436650285684be8d244911a14f5eabc61ada39c67d1`.

Other dependencies use existing pinned manifests: fulltrain packed codes/E117
routes from Chloeeeeeeee123/MoT-1, native TiTok generator/tokenizer from
fun-research/TiTok, MoT EMA from sophiaa/MoT-1-checkpoints, ADM reference/graph
from the official guided-diffusion assets. `python -m bert2d.assets --fid --list`
shows all URLs/revisions/checksums. No raw ImageNet, val-code cache, original
standalone LlamaGen checkpoint or unpublished local file required. Large assets,
results and HF caches remain outside Git; HF_HOME defaults to MOTAR_RESULTS/.hf.

## Fixed protocol

- 8 physical H20 GPUs0..7, micro56, accumulation1, global448. Permission must be
explicit; no preemption. Requires40GiB free/card and checkpoint disk headroom.
- Restore all weights/Adam moments and step. LRnew1e-4, pretrained1e-5,
betas(.9,.96),WD.03,clip1; no new warmup. Same loss and full packed train data.
- Preserve ordered global448 K-buckets and next-batch cursor across4→8 ranks.
Old ranks retain RNG; new ranks get distinct sourcehash-derived RNG. Disable
activation recomputation for H20 speed; no model layer is removed. Hardware
and layout change mean NOT bitwise replay. Do not inflate batch to fill RAM.
- 2 real updates to6002, checkpoint readback and8-image generation smoke, then
strictly resume the SAME checkpoint. No repeated/discarded test updates.
- Train to8000,12000,16000,20000 TOTAL updates. At each endpoint stop training,
evaluate stable latest on4GPUs, then resume optimizer+cursor. No snapshots.
- Paired5k ADM FID:4logical shards,batch8,seed20260914; official1D8steps,
CFG4.5 linear,temp9.5;2D16steps,cosine categorical/confidence sampling,CFG4.5.
Merge all5000 features, never average shardFID. full_refine is primary; base
is diagnostic. No all-discrete endpoint and no50k evaluation.

## Files and failure behavior

Only `run_seed0_g448/train/latest.pt` rolls forward, every500updates/end.
No numbered/best/eval-snapshot ckpt. Atomic save briefly uses latest.pt.tmp.
After verified8k save+resume+eval, delete downloaded input6000 copy; preserve
HF/5090 original. Delete verified smoke feature arrays; retain small audits.
Keep formal5k features, previews and JSON. Local JSON records, no W&B/log.txt
for this bounded probe; older40-epoch W&B behavior is unchanged.

```bash
cat "$MOTAR_RESULTS/input6000/download.json"
cat "$MOTAR_RESULTS/run_seed0_g448/pipeline.json"
cat "$MOTAR_RESULTS/run_seed0_g448/train/train_rank0.json"
cat "$MOTAR_RESULTS/run_seed0_g448/fid_trend.json"
```

Stop on a failed stage; no silent OOM retry/skipped eval. Training stage4h limit,
eval65min limit. Use a fresh output. After interruption inspect exact saved step
and failures before preparing a continuation; don't blindly rerun the fresh
pipeline over existing files or substitute an unrelated checkpoint.

## Verification limits

CPU tests passed locally and on H20: exact4x112 vs8x56 rows/order across3epochs,
cursor offsets, independent deterministic new-rank RNG, and AST-identical
generation loop before decoding. Original model/sampler/sourceweight hashes
unchanged. Clean-checkout checks precede publication. Actual8-GPU smoke and
20k quality remain pending until corresponding run records confirm completion.
