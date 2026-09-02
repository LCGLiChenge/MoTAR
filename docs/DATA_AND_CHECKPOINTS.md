# Data and checkpoint handoff

Neither the packed ImageNet codes nor checkpoints belong in Git.

## Current E117 MaskGIT handoff

The current MaskGIT trainer requires four directories:

1. full ImageNet train packed TiTok/LlamaGen codes (1,281,167 sources, two
   augmentations);
2. ImageNet val packed codes (50,000 sources, one augmentation);
3. full-train E117 K=64/128 route cache;
4. validation E117 K=64/128 route cache.

Packed codes can be regenerated using `scripts/launch_extract_h200.sh` and
`scripts/launch_extract_val_h200.sh`. Exact known manifests are:

```text
docs/mot199440_packed_manifest.sha256
docs/mot199440_val_packed_manifest.sha256
```

The route caches cannot currently be downloaded from Hugging Face. Transfer
the exact current-server directories listed in `README.md`. Their
manifests are:

```text
docs/e117_routes_train_manifest.sha256
docs/e117_routes_val_manifest.sha256
```

Both caches must report E117 checkpoint SHA256
`a5b84689d2b29f579d2442da7594ac093292b6386760867a0668ca02f82e6156`.
The formal launcher validates this before probing or training.

The 138,821,223-byte E117 checkpoint itself is not needed for training from
precomputed routes. It is needed only to regenerate routes or perform
end-to-end generation, and it has not yet been published to Hugging Face.

## Packed code dataset

Expected files:

```text
titok_codes.npy     [1281167, 2, 32]  uint16
llamagen_codes.npy  [1281167, 2, 256] uint16
labels.npy          [1281167]         uint16
written.npy         [1281167]         uint8
meta.json
manifest.sha256
```

The known source dataset is:

```text
/mnt/data/heyefei/lichenge/Mixture-of-Tokenizer/AR/data/imagenet-titok_l32-mot199440ema-adm-256_packed
```

Copy it to persistent storage on the H200 host, then run:

```bash
cd "$PACKED_CODE_ROOT"
sha256sum -c manifest.sha256
```

The expected hashes are also recorded in
`docs/mot199440_packed_manifest.sha256`.

## Regenerate packed codes on H200

The MoT checkpoint is downloaded from
`sophiaa/MoT-1-checkpoints/latest.pt`. It contains all 343 adapted LlamaGen VQ
parameters, but no TiTok encoder/codebook. Extraction therefore also downloads
the original TiTok-L32 checkpoint. The original LlamaGen VQ checkpoint is not
needed; the only omitted LlamaGen state is the eval-irrelevant
`quantize.codebook_used` usage buffer.

Required persistent storage is approximately 9.0 GB for extraction weights plus
1.4 GiB for packed codes, excluding ImageNet itself.

```bash
export ASSET_ROOT=/persistent/assets/motar-extraction
export IMAGENET_TRAIN=/persistent/datasets/imagenet/train
export PACKED_CODE_ROOT=/persistent/data/imagenet-titok_l32-mot199440ema-adm-256_packed

bash scripts/launch_extract_h200.sh
```

The launcher pins both source repositories, validates both checkpoints'
sizes and SHA256 hashes, then runs 8-rank FP32 extraction. The default extraction
batch is 64 images per H200 before the original/flip augmentation doubles the
encoder input. Set `EXTRACT_BATCH_SIZE=32` if the target host has co-tenancy.
An interrupted run resumes from `written.npy`; completed output includes a new
`manifest.sha256`.

This extraction batch is independent of AR training. AR training performs its
own real forward/backward/AdamW H200 probe and does not use tokenizer weights.

## Legacy causal-AR H200 training state

Do not transfer a prior AR model or optimizer checkpoint to this experiment.
The first H200 launch starts from random initialization and consumes only the
packed code dataset.

After each completed H200 epoch, the trainer atomically replaces:

```text
$RESULTS_DIR/$EXP_NAME/checkpoints/latest
```

That native checkpoint is used only to recover the same H200 experiment after
an interruption. If a completely new run is intended, use a new, empty
`RESULTS_DIR`/`EXP_NAME` combination.
