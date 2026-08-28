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
