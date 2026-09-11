# Joint handoff validation — 2026-09-11

This report distinguishes actual tests from the mandatory tests still to be run
on the destination H20. No new image-quality/FID claim is made.

## Passed locally

- 15 CPU unit tests: all9 existing baseline tests plus6 joint tests. Official
  BERT arithmetic/load accounting,1D isolation, zero spatial/full-cross residual,
  nonzero spatial gradient, alpha=.5 initialization, no image-gradient leak into
  generator/frozen conditions, independent fusion accumulation, optimizer phase
  membership, sampler repeatability/padding, joint tensor restore and exact next
  Adam update on a tiny fixture, corruption and wrong-format rejection.
- Two-rank CPU Gloo test: CE and fusion each accumulated over two microbatches,
  followed by one DDP synchronization. Compared with one-process global-batch
  gradients: maximum absolute error **4.470348358154297e-08**.
- Real RTX5090 single-GPU in-memory capacity/gradient test: local spatial branch,
  CE micro1, fusion micro1, two Adam/EMA updates and up-to32-image development
  forward. Peak reserved **9.2578125GiB**. See
  [machine-readable report](joint_local_gpu_validation.json).
- Real **two-GPU NCCL training-controller smoke**,3 optimizer updates, CE global4
  (micro1/rank, accumulation2), fusion global4 (micro1/rank, accumulation2).
  Actual packed TRAIN data, frozen native tokenizer/MoT/E117, warmup→joint,
  online generation, differentiable frozen image decoding, CE and fusion updates,
  raw/EMA token/image development, and **W&B online** all completed.
  Rank0's duplicate audit saw12 disjoint global(source,augmentation) pairs.
  Peak reserved observed on rank0 **9.146484375GiB**; process ended normally.
  [W&B smoke run](https://wandb.ai/lcg-li-chenge-zhejiang-university/motar-maskgit/runs/vpbs8g3e).

The real loader verified SHA256 of frozen artifacts, then copied and compared
**692 nativeTiTok tensors,661 MoT EMA tensors and387 E117 EMA tensors**.
Official generator migration accounted for391 source tensors /393 loaded target
tensors. MoT state is explicitly model_ema at step199440. No original LlamaGen
checkpoint was used. The MoT state does not contain the nativeTiTok quantizer,
so that dependency is intentionally still downloaded from fun-research/TiTok.

An initial real-weight test correctly rejected a missing `quantize.codebook_used`
buffer: the upstream VQ default registers training-only usage history, absent
from MoT EMA. The adapter now explicitly constructs VQ with
`codebook_show_usage=False`; **strict learned-tensor loading remains enabled**.
The real GPU tests above are after that fix. Vendored upstream source was not
silently changed to ignore arbitrary checkpoint keys.

All24 joint-profile assets passed size/SHA256 verification. Six ported sampler,
fusion and routing definitions were AST-identical to the experiment sources.
Vendored provenance records both original and delivered hashes: three upstream
files differ only in terminal newlines, verified byte-for-byte after stripping
terminal LF characters; no executable code differences were hidden.

## Not tested locally / must not be claimed

- No H20 hardware,8-GPU capacity result, formal2048/16 global-batch run,80-epoch
  run, or new50k FID result was tested here.
- No multi-GB real joint checkpoint was saved/restored locally: the local
  filesystem had only about1.1GiB free. The two-GPU smoke used the explicitly
  bounded `--in-memory-smoke` developer mode. Its summary has
  `checkpoint=null` and `full_checkpoint_tested=false`.
- Tiny CPU checkpoint tests used temporary fixtures inside the repository and
  a mocked free-space query. They do not weaken the production10GiB guard and
  are NOT evidence that the real full state was saved locally.
- The `full` spatial-memory ablation has CPU structural tests only, not a real
  GPU quality/capacity result. Default `local` is the real-GPU-tested path.
- The preview renderer was exercised in development diagnostics; the new
  checkpoint-to-PNG CLI still requires a stable real joint checkpoint on H20.

## Required destination gate

`scripts/launch_spatial_joint_h20.sh` performs, in order:

1. Pinned versions, full asset hashes, dataset provenance and GPU ownership.
2. Actual-model capacity probe including EMA/Adam/frozen assets/online fusion.
3. Actual chosen-micro/world DDP smoke with four updates and full save/readback.
4. Fresh process loading the saved raw/EMA/Adam/RNG/cursor, updating to step5,
   saving and verifying again.
5. Only then formal training from official initialization, never smoke weights.

No in-memory receipt can pass this gate. The launcher checks the actual
checkpoint format/hash/size, saved step, readback result, and resume start/end.
If the destination gate fails, stop and report the failure; do not bypass it or
call this package verified for that machine. A full-state roundtrip is necessary
but still does not prove long-run model quality.

## Re-run small tests

```bash
python -m unittest h20.test_model h20.test_portability h20_joint.test_joint -v
CUDA_VISIBLE_DEVICES='' python -m torch.distributed.run \
  --standalone --nproc_per_node=2 -m h20_joint.ddp_test
bash -n scripts/launch_spatial_joint_h20.sh
```

The small tests need no ImageNet, large checkpoints, private paths or GPU.
