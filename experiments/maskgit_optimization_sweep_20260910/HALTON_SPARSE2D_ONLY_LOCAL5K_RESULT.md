# Halton sparse 2D-only MaskGIT local5k result

## Material Passport

- Date: 2026-09-14
- Mode: experiment validation / short screening
- Scope: 2D-only sparse selected-token generation, not unified MaskGIT
- Target: generate only selected sparse 2D token positions conditioned on frozen/generated 1D features and route indices
- Best large output: `/mnt/data/heyefei/lichenge/Mixture-of-Tokenizer/AR/results/maskgit_optimization_sweep_20260910/halton_fullctx_sparse2d_u50_halton5k_v1`
- Best checkpoint evaluated: `/mnt/data/heyefei/lichenge/Mixture-of-Tokenizer/AR/results/maskgit_optimization_sweep_20260910/halton_fullctx_sparse2d_u50_m112_g448_v1/latest.safetensors`
- Best checkpoint SHA256: `fddfd00faedb135b91bdfb9506ab470b8e531c26f6af22359faf1592ef898e7d`
- External init: `llvictorll/Halton-MaskGIT` `ImageNet_256_base.pth`, SHA256 `7fe25cb80b05743e8b42dacc61d88792a9ab5217d6bd756bef28821e0a5fc68f`

## Definition

This experiment is intentionally not the unified 1D+2D MaskGIT objective.

The model receives:

- class condition;
- full 16x16 frozen 1D feature memory;
- selected sparse route indices from generated 1D / E117-style routing.

The model predicts logits only for the selected sparse 2D positions. It does not generate 1D tokens and does not predict unselected 16x16 2D-grid tokens. Evaluation decodes by direct replacement of the selected 2D tokens.

## Model and training snapshot

Model/trainer:

- Adapter: `experiments.maskgit_optimization_sweep_20260910.halton_sparse2d_adapter.HaltonSparse2DAdapter`
- Trainer: `experiments.maskgit_optimization_sweep_20260910.halton_sparse2d_trial`
- Initialization: pretrained LlamaGen-token MaskGIT weights from Halton-MaskGIT, plus fresh zero-init 1D-feature conditioning branches
- Objective: masked CE/NLL on selected sparse 2D positions only

100-update screen:

- Output: `/mnt/data/heyefei/lichenge/Mixture-of-Tokenizer/AR/results/maskgit_optimization_sweep_20260910/halton_sparse2d_u100_m192_g768_v1`
- GPUs: 4
- Micro/global batch: 192 per GPU / 768 global
- Peak memory: about 28.13 GiB
- Final dev raw NLL2D: 8.0168
- Final dev EMA NLL2D: 9.1883

50-update screen:

- Output: `/mnt/data/heyefei/lichenge/Mixture-of-Tokenizer/AR/results/maskgit_optimization_sweep_20260910/halton_sparse2d_u50_m192_g768_v1`
- Checkpoint SHA256: `a696f4e55fb9797067beb767c3a413f661202c6167002d08dfc3e5906365fd75`
- GPUs: 4
- Micro/global batch: 192 per GPU / 768 global
- Final dev raw NLL2D: 8.2628

75-update screen:

- Output: `/mnt/data/heyefei/lichenge/Mixture-of-Tokenizer/AR/results/maskgit_optimization_sweep_20260910/halton_sparse2d_u75_m192_g768_v1`
- Checkpoint SHA256: `c12f07cabbff965f448e729b24229fada181a023d07561b1eb4435b7a942be8a`
- GPUs: 4
- Micro/global batch: 192 per GPU / 768 global
- Peak memory: about 30.17 GiB
- Final dev raw NLL2D: 8.1056
- Final dev EMA NLL2D: 9.2417

Full-context sparse-output 25-update screen:

- Output: `/mnt/data/heyefei/lichenge/Mixture-of-Tokenizer/AR/results/maskgit_optimization_sweep_20260910/halton_fullctx_sparse2d_u25_m112_g448_v1`
- Checkpoint SHA256: `90b19d941d123f07563ad3f6e328473a8f48818bff137b11d5ac79f49ba3cd46`
- GPUs: 4
- Micro/global batch: 112 per GPU / 448 global
- Peak memory: about 30.44 GiB
- Final dev raw NLL2D: 8.0354
- Final dev EMA NLL2D: 8.6184

Full-context sparse-output 50-update screen:

- Output: `/mnt/data/heyefei/lichenge/Mixture-of-Tokenizer/AR/results/maskgit_optimization_sweep_20260910/halton_fullctx_sparse2d_u50_m112_g448_v1`
- Checkpoint SHA256: `fddfd00faedb135b91bdfb9506ab470b8e531c26f6af22359faf1592ef898e7d`
- GPUs: 4
- Micro/global batch: 112 per GPU / 448 global
- Peak memory: about 30.45 GiB
- Final dev raw NLL2D: 7.8761
- Final dev EMA NLL2D: 8.5985

Full-context sparse-output 75-update screen:

- Output: `/mnt/data/heyefei/lichenge/Mixture-of-Tokenizer/AR/results/maskgit_optimization_sweep_20260910/halton_fullctx_sparse2d_u75_m112_g448_v1`
- Checkpoint SHA256: `a97fa35bb3b57a6bba2c24bb5ad5d590b99d55a12355fb5220f26cfcc2098a2d`
- GPUs: 4
- Micro/global batch: 112 per GPU / 448 global
- Peak memory: about 30.45 GiB
- Final dev raw NLL2D: 7.7882
- Final dev EMA NLL2D: 8.5748

500-update continuation from the 100u raw checkpoint:

- Output: `/mnt/data/heyefei/lichenge/Mixture-of-Tokenizer/AR/results/maskgit_optimization_sweep_20260910/halton_sparse2d_total500_fromu100_m192_g768_v1`
- Checkpoint SHA256: `72f1889b64b58c52c0ed0a5666b310fa001afef4dba1649f2930df1e3a0bf5f7`
- GPUs: 4
- Micro/global batch: 192 per GPU / 768 global
- Peak memory: about 30.21 GiB
- Final dev raw NLL2D: 7.7818
- Final dev EMA NLL2D: 8.4155

## Free-generation local5k results

All rows use generated 1D prefix, selected sparse 2D token generation, and direct replacement decode.

| checkpoint | seed | sampler | local5k FID | IS | mean K | output |
|---|---:|---|---:|---:|---:|---|
| base 1D only | 20260914 | none | 11.0116 | 139.3324 | 103.19 | `halton_sparse2d_freefid5k_v1` |
| 50u raw | 20260914 | halton_margin4 | **9.2655** | 230.7767 | 103.19 | `halton_sparse2d_u50_halton5k_v1` |
| 75u raw | 20260914 | halton_margin4 | 9.2687 | 231.4442 | 103.19 | `halton_sparse2d_u75_halton5k_v1` |
| fullctx 25u raw | 20260914 | halton_margin4 | 9.2837 | 272.3539 | 103.19 | `halton_fullctx_sparse2d_u25_halton5k_v1` |
| fullctx 50u raw | 20260914 | halton_margin4 | **9.1383** | 257.8070 | 103.19 | `halton_fullctx_sparse2d_u50_halton5k_v1` |
| fullctx 75u raw | 20260914 | halton_margin4 | 9.1773 | 249.7147 | 103.19 | `halton_fullctx_sparse2d_u75_halton5k_v1` |
| 100u raw | 20260914 | margin4 | 9.8445 | 218.6402 | 103.19 | `halton_sparse2d_freefid5k_v1` |
| 100u raw | 20260914 | halton_margin4 | **9.4266** | 219.6512 | 103.19 | `halton_sparse2d_u100_sampler5k_v1` |
| 100u raw | 20260914 | categorical05_margin4 | 9.5550 | 217.3268 | 103.19 | `halton_sparse2d_u100_sampler5k_v1` |
| 100u raw | 20260914 | halton_fixed_margin4 | 9.4680 | 222.6199 | 103.19 | `halton_sparse2d_u100_directsampler5k_v1` |
| 100u raw | 20260914 | baseline | 9.8377 | 215.1252 | 103.19 | `halton_sparse2d_u100_directsampler5k_v1` |
| 100u raw | 20260914 | margin | 9.9950 | 220.7551 | 103.19 | `halton_sparse2d_u100_directsampler5k_v1` |
| 100u raw | 20260914 | greedy_margin4 | 10.2466 | 210.4979 | 103.19 | `halton_sparse2d_u100_directsampler5k_v1` |
| 100u raw | 20260914 | greedy_margin8 | 10.2964 | 213.2954 | 103.19 | `halton_sparse2d_u100_directsampler5k_v1` |
| 500u raw | 20260914 | margin4 | 10.2429 | 201.1472 | 103.19 | `halton_sparse2d_total500_freefid5k_v1` |
| 500u raw | 20260914 | halton_margin4 | 9.4356 | 214.1472 | 103.19 | `halton_sparse2d_total500_sampler5k_v1` |
| 500u raw | 20260914 | categorical05_margin4 | 9.7131 | 206.3612 | 103.19 | `halton_sparse2d_total500_sampler5k_v1` |
| 100u raw | 20260915 | base 1D only | 10.8132 | 142.7209 | 104.22 | `halton_sparse2d_u100_halton_margin4_seed20260915_5k_v1` |
| 100u raw | 20260915 | halton_margin4 | **9.3360** | 225.6006 | 104.22 | `halton_sparse2d_u100_halton_margin4_seed20260915_5k_v1` |


Additional bounded controls after the first result:

| checkpoint | train condition | pretrained LR | dev raw NLL2D | sampler | seed | local5k FID | output |
|---|---|---:|---:|---|---:|---:|---|
| feature-only 500u | real GT 1D | 0 | 7.9894 | halton_margin4 | 20260914 | 10.2766 | `halton_sparse2d_featureonly_u500_halton5k_v1` |
| lowlr 500u | real GT 1D | 2e-6 | 7.8794 | halton_margin4 | 20260914 | 10.0520 | `halton_sparse2d_lowlr2e6_u500_halton5k_v1` |
| teacher-generated 100u | generated 1D cache | 1e-5 | 6.8147 | halton_margin4 | 20260914 | 11.5024 | `halton_sparse2d_teacher_generated_u100_halton5k_v1` |
| teacher-generated 100u | generated 1D cache | 1e-5 | 6.8147 | margin4 | 20260914 | 11.9036 | `halton_sparse2d_teacher_generated_u100_halton5k_v1` |
| halton-mask 100u | real GT 1D | 1e-5 | 8.0143 | halton_margin4 | 20260914 | 9.4631 | `halton_sparse2d_haltonmask_u100_halton5k_v1` |
| halton-mask + rollin25 | real GT 1D | 1e-5 | 9.3634 at step50 | not run | 20260914 | n/a | `halton_sparse2d_haltonmask_rollin25_u100_m192_g768_v1` stopped at step50 |

The teacher-generated cache control uses `/mnt/data/heyefei/lichenge/Mixture-of-Tokenizer/AR/results/generated_prefix_teacher_20260910/cache100k_seed20261020` and generated-prefix dev from `experiments/generated_prefix_teacher_20260910/paired2000_seed20261011`. It trains and validates on generated 1D prefixes, but its free-generation FID is worse than the base 1D endpoint. This strengthens the conclusion that lower teacher-forced NLL is not sufficient: the target/sampler distribution must match FID-improving direct replacement, not merely teacher-token likelihood under generated prefixes.

## Interpretation

The selected sparse 2D-only path is viable in local5k screening. The best direct-replacement result so far is `fullctx 50u raw + halton_margin4`, which improves local5k FID over the generated-1D base by about 1.87 on seed 20260914. The full-context sparse-output change preserves the selected-token-only contract but lets the pretrained MaskGIT backbone see its original 16x16 context layout. This improves over the sparse-sequence 50u result; fullctx 25u and 75u are both worse than fullctx 50u despite the 75u checkpoint having lower dev NLL.

The most important observation is that teacher-forced NLL and free-generation FID are not monotonic here: NLL improves from 50u to 75u to 100u to 500u, and full-context lowers NLL further, but local5k FID still peaks early. Continuing from 100u to 500u lowers dev raw NLL2D from 8.0168 to 7.7818, but the default margin4 FID worsens from 9.8445 to 10.2429. With the stronger Halton-order sampler, 500u recovers to 9.4356 but still does not beat the early 50--75u checkpoints.

The additional controls make this sharper. Freezing the pretrained backbone is too weak for generation FID; a lower pretrained LR is still worse than the 100u baseline; and training directly on a generated-prefix teacher cache drives generated-prefix dev NLL down to 6.8147 but makes FID worse than base. Matching the training mask order to Halton alone also does not help, and sampled-context roll-in at 25% is harmful enough to stop early at step50.

Therefore, current checkpoint selection should be FID/sampler aware rather than based on masked NLL alone. Blind long training is not justified unless paired with intermediate local5k FID checkpoints or a training objective/schedule that better matches free-generation direct replacement. The best confirmed local5k setting is now the full-context sparse-output 50u checkpoint evaluated with the Halton-order margin4 sampler.

## Current best setting

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m experiments.maskgit_optimization_sweep_20260910.halton_sparse2d_free_fid \
  --checkpoint /mnt/data/heyefei/lichenge/Mixture-of-Tokenizer/AR/results/maskgit_optimization_sweep_20260910/halton_sparse2d_u100_m192_g768_v1 \
  --output /mnt/data/heyefei/lichenge/Mixture-of-Tokenizer/AR/results/maskgit_optimization_sweep_20260910/halton_sparse2d_u100_sampler5k_v1 \
  --state raw \
  --n 5000 \
  --batch 8 \
  --feature-batch 8 \
  --seed 20260914 \
  --feature-chunk 4 \
  --variants halton_margin4
```

This result should not be reported as a final ImageNet FID. It is a local5k screening result with generated 1D prefix and selected sparse 2D generation only.
