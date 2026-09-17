# 1D-feature to RGB-reencoded 2D proxy mapper

This experiment maps the frozen MoT/TiTok 16x16 base feature grid to the 256
LlamaGen VQ IDs obtained by decoding the same 1D representation to RGB and
re-encoding it with the frozen MoT EMA LlamaGen encoder.  The target is not the
original-image VQ grid.  At final mixed decoding, unselected cells still use the
original continuous 1D features; only Router-selected cells are generated and
directly replaced by sparse 2D features.

## Verified full mapper

The verified model is `MODEL=full`, with 4,686,336 parameters.  It was trained
for 1,000 updates on one H20 with batch 512, AdamW, peak reserved memory
47.996 GiB, and a source-disjoint held-out set.

Held-out proxy-token metrics:

| update | NLL | exact token accuracy |
|---:|---:|---:|
| 0 | 6.913196 | 0.140999 |
| 1000 | 3.728841 | 0.197769 |

Paired generation uses the same 20k mixed-prefix Halton checkpoint, generated
sample IDs, E117 Router, decoder, direct-replacement rule, CFG 2.75, and 8+8
MaskGIT sampling steps.

| proxy condition | FID-5k | FID-50k |
|---|---:|---:|
| exact RGB -> frozen encoder | 7.668903 | 1.939918 |
| learned full mapper | 7.683637 | 1.976907 |

The 50k mapper gap is +0.036989 FID.  The mapper checkpoint SHA256 is
`70a15986a31bd076d393a487cbe84ee7a1883b71e53e56773a9a86c04550c88e`.
The evaluated Halton checkpoint SHA256 is
`d77fabe3420b4bd94680868b5c5ccbc36f1f5f1f63c6a2f1d23a6f71286ae6bc`.

## Proxy cache

The cache contract is `(1281167 * 2, 256)` `uint16`, with both ADM views of a
source kept in the same train/eval split.  A resumable snapshot is in
`Chloeeeeeeee123/MoT-1` at:

```text
proxy_cache/mot199440_1d_base_rgb_reencoded_train_adm_20260916/partial_world8
```

Download that directory, then resume all missing rows and finalize it with
`scripts/launch_rgb_proxy_cache_8gpu_h20_resume.sh`.  Training refuses a cache
without a complete `summary.json` and full-coverage audit.

## Train the verified mapper

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
export MOTAR_ASSETS=/persistent/assets/motar
export MOTAR_RESULTS=/persistent/results/motar
export PROXY_ROOT=/persistent/data/rgb_proxy_train
export PYTHON_BIN=python
export CUDA_VISIBLE_DEVICES=0

MODEL=full MAPPER_RANK=12 BATCH=512 STEPS=1000 \
RUN_NAME=proxy_converter_full_1k_seed20260917 \
bash scripts/launch_proxy_converter_h20.sh
```

The `MAPPER_RANK` value is ignored by `MODEL=full`.  W&B project defaults to
`motar-proxy-converter`.  The trainer saves only `latest.pt` plus JSON audit
sidecars and refuses to overwrite an existing output directory.

## Low-rank candidate

`MODEL=lowrank MAPPER_RANK=12` factorizes the 256-to-16384 classifier.  It has 691,712
parameters, an 85.2% reduction from the full mapper.  The frozen LlamaGen
codebook originates in 8 dimensions before the affine post-quant projection;
rank 12 therefore preserves the initial nearest-codeword geometry with numeric
headroom.  This candidate must not be described as quality-equivalent until its
paired FID has been measured.

```bash
MODEL=lowrank MAPPER_RANK=12 BATCH=512 STEPS=1000 \
RUN_NAME=proxy_converter_lowrank12_1k_seed20260917 \
bash scripts/launch_proxy_converter_h20.sh
```

## Paired FID

```bash
python -c "from experiments.feature_to_token_20260916.halton_parallel_eval import cli; raise SystemExit(cli())" \
  --checkpoint /persistent/results/halton/latest.pt \
  --mapper-checkpoint /persistent/results/motar/proxy_converter_full_1k_seed20260917/latest.pt \
  --output /persistent/results/motar/proxy_converter_full_eval5k \
  --n 5000 --seed 20260914 --cfg-w 2.75 --stage2-steps 8 \
  --router-mode e117 --workers 8 --allowed-gpus 0,1,2,3,4,5,6,7
```

The evaluator records source/checkpoint hashes, exact sample coverage, frozen
asset identities, unchanged anchors, peak memory, and both base/full-refine FID.
