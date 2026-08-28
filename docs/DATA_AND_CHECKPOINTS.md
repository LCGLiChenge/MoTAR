# Data and checkpoint handoff

Neither the packed ImageNet codes nor checkpoints belong in Git.

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
`sophiaa/MoT-1-checkpoints/latest.pt`. It contains the adapted LlamaGen VQ,
decoder, and router state but no TiTok encoder/codebook and no original
LlamaGen base buffers. Extraction therefore also downloads the original
TiTok-L32 and LlamaGen VQ-16 checkpoints.

Required persistent storage is approximately 9.3 GB for extraction weights plus
1.4 GiB for packed codes, excluding ImageNet itself.

```bash
export ASSET_ROOT=/persistent/assets/motar-extraction
export IMAGENET_TRAIN=/persistent/datasets/imagenet/train
export PACKED_CODE_ROOT=/persistent/data/imagenet-titok_l32-mot199440ema-adm-256_packed

bash scripts/launch_extract_h200.sh
```

The launcher pins both source repositories, validates all three checkpoint
sizes and SHA256 hashes, then runs 8-rank FP32 extraction. The default extraction
batch is 64 images per H200 before the original/flip augmentation doubles the
encoder input. Set `EXTRACT_BATCH_SIZE=32` if the target host has co-tenancy.
An interrupted run resumes from `written.npy`; completed output includes a new
`manifest.sha256`.

This extraction batch is independent of AR training. AR training performs its
own real forward/backward/AdamW H200 probe and does not use tokenizer weights.

## H200 training state

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
