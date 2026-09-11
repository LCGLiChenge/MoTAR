# 历史基线：TiTok-BERT unified MaskGIT — H20 handoff

本文仅适用于旧的无空间输入baseline。当前新联合方案见 [README.md](README.md)。

**当前入口是 `scripts/launch_titok_bert_h20.sh`，不是旧 H200 scratch launcher。**
本版本移植的是 2026-09-10 已通过检查的 TiTok-BERT 方案：官方 TiTok-L32
MaskGIT 初始化，共享 24 层、width 768、16 heads 的 BERT；class → 32 个 1D token；
冻结 E117 从完整 1D 决定 K=64/128；class + 1D + 坐标 → 稀疏 2D token。
1D forward 不接收 2D。训练阶段的 2D 条件仍然是真实 1D code，**尚未改成生成前缀训练**。

旧的从零训练 pre-norm LLaMA 版本见 [旧版说明](docs/LEGACY_SCRATCH_MASKGIT.md)，
旧 causal AR 见 [历史 AR](docs/LEGACY_CAUSAL_AR.md)。不要混用权重或命令。

## 1. 新服务器安装环境

要求：Linux x86_64、Conda/Miniforge（或 Python 3.10 venv）、git，以及能够运行
CUDA 12.8 PyTorch 的 NVIDIA 驱动。建议至少 80 GB 可用磁盘、64 GB 主机内存。
脚本不会修改系统驱动，也不会把旧服务器的账号配置复制过来。

```bash
git clone https://github.com/LCGLiChenge/MoTAR.git
cd MoTAR
bash scripts/install_h20_environment.sh
conda activate motar-h20
wandb login
```

已有独立 Python 3.10 环境时：`bash scripts/install_h20_environment.sh --active-env`。
不要在别的正在使用的环境中覆盖安装。
非交互执行时可以在每条 Python/启动命令前加
`conda run --no-capture-output -n motar-h20`，不依赖前一个 shell 的激活状态。

环境文件都在 GitHub：`environment-h20.yml`、`requirements-h20.txt`、
`constraints-h20.txt`。核心是 PyTorch 2.10.0 / torchvision 0.25.0 / CUDA 12.8，
Transformers 固定到 `a957b7911a758d54597914b4479fe6e81424d64f`，不是随便安装最新版。
安装方法对应 [PyTorch 官方版本说明](https://pytorch.org/get-started/previous-versions/)。
这是可重建环境，不是包含私人账号和无关包的原机器环境镜像。

## 2. 下载资产：不需要再从 ImageNet 提取 code

所有下载地址、不可变 revision、文件大小和 SHA256 以
[`configs/h20_assets.json`](configs/h20_assets.json) 为准。

```bash
export MOTAR_ASSETS="$PWD/assets/h20"
python -m h20.assets --root "$MOTAR_ASSETS" --profile resume
```

`resume` 下载完整训练数据和当前完整续训状态（约 5.8 GB）；`train` 只下载从官方
TiTok 初始化所需内容（约 2.6 GB）；`all` 额外下载 decoder、tokenizer、E117
权重，资产合计约 14.9 GB。Hugging Face 缓存和本地资产副本可能额外占用一份空间。
重复执行会先检查已有文件，不覆盖哈希不符的文件。
如果网络环境导致 Xet 下载停滞，可在命令前加 `HF_HUB_DISABLE_XET=1` 使用普通下载通道。
两个最大的文件以 `.parts/` 传输块存放，下载脚本逐块校验后自动拼接，再验证完整
文件 SHA256。它们不是模型格式转换，不需要手工拼接；训练仍读取普通完整文件。

| 资产 | Hugging Face 位置 | 用途 |
|---|---|---|
| 全量 train codes、完整 val codes | `Chloeeeeeeee123/MoT-1/h20-titok-bert-20260910/codes/` | 训练/固定验证，无原图依赖 |
| train/val E117 routes | 同仓库 `e117_routes_full_train_e116/`、`e117_routes_imagenet_val_e116/` | 冻结路由 |
| step 10938 完整 checkpoint | 同仓库 `h20-titok-bert-20260910/resume/` | raw、EMA、AdamW、RNG、进度 |
| E117 checkpoint | 同仓库 `h20-titok-bert-20260910/router/e117.pt` | 未来生成时的路由，训练不加载 |
| 官方 generator | `fun-research/TiTok/generator_titok_l32.bin` | 官方 1D 初始化 |
| 官方 TiTok tokenizer | `fun-research/TiTok/tokenizer_titok_l32.bin` | 提取/解码控制组，训练不加载 |
| MoT decoder EMA step 199440 | `sophiaa/MoT-1-checkpoints/latest.pt` | 图像重建/生成解码，训练不加载 |

Hugging Face 新资产目录：
https://huggingface.co/Chloeeeeeeee123/MoT-1/tree/main/h20-titok-bert-20260910

没有上传 ImageNet 原图或任何登录凭据。仅训练 unified MaskGIT 时不需要本地
ImageNet 原图、旧服务器目录、旧结果 manifest 或未发布的 Router 文件。
本仓库包含训练需要的最小 RandAR/TiTok 源文件和许可证。

## 3. 让 Codex 检查并开始训练

先明确指定获准使用的卡；支持 1、2、4、8 卡，不会自动占用服务器全部 GPU。
下面以四卡为例，八卡时显式改为 `0,1,2,3,4,5,6,7`。

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MOTAR_OUTPUT="$PWD/results/titok-bert-h20-seed0"
export WANDB_PROJECT=motar-maskgit
# 如需团队 workspace，设置你有写权限的 entity；否则使用当前登录账户。
# export WANDB_ENTITY=your-team

python -m h20.preflight --assets "$MOTAR_ASSETS"
bash scripts/launch_titok_bert_h20.sh --init resume --steps 50000
```

默认恢复发布的 step 10938，目标是**总计 50,000 optimizer steps**，不是再跑
50,000 steps。global batch 固定 2048，累计约 79.93 个源图像等效 epoch；
这不是旧 scratch 版本的约 800 epoch 预算。若决定更长训练，显式设置 `--steps`。

从官方 TiTok 重新开始时，使用一个新的输出目录：

```bash
export MOTAR_OUTPUT="$PWD/results/titok-bert-h20-official-seed0"
bash scripts/launch_titok_bert_h20.sh --init official --steps 50000
```

“official”不是整个模型随机初始化：官方 1D/BERT 参数加载 TiTok，新增 2D
embedding/head 随机初始化。前 600 步只更新新增 2D 参数，之后共享主干和 1D
参数也参与更新。新 2D LR=1e-4，预训练参数 LR=1e-5；1D/2D loss 权重=1.5/1；
AdamW betas=(0.9,0.96)、weight decay=0.03、EMA=0.999、bf16、TF32 off；
arccos masking、label smoothing=0.1、visible-token weight=0.1，per-sample reduction。

只检查配置不启动：在启动命令后添加 `--plan-only`。
短测试（单独目录，不是正式训练）：添加 `--smoke --max-micro 2`。

## 4. H20 显存与恢复语义

启动器首先在选中卡中显存最小的一张上运行真实 K=128 双阶段前向/反向、AdamW、EMA，
从大到小测试能整除 global batch 的 microbatch，默认峰值 reserved 不超过 92%。
选择结果写入 `capacity/` 和 `run_plan.json`，自动用梯度累积保持 global batch 2048。
实际 DDP 仍可能因外部占用等因素 OOM；遇到非有限值或 OOM 会报错，不会静默缩
batch、跳过训练样本或继续错误训练。请使用新的输出目录和较小 `--max-micro` 重试。

八卡时 microbatch 上限为 256；即使仍有显存，也不会为了填满显存擅自增大
有效 batch。不能把在 RTX 5090 上的显存结果冒充 H20 实测。

恢复一定读取 raw、EMA 和完整 AdamW，不是只加载 EMA。当卡数/microbatch/累积
布局相同时恢复保存的 RNG 和数据游标；布局改变则保留参数、优化器和全局步数，
从下一个确定性 packed pass 开始并重新设各 rank RNG，写明为非逐步一致的迁移。
跨 GPU 架构/内核也不承诺 bitwise 相同。

同一 H20 run 再启动相同命令，会使用该输出目录的 latest，继续原 W&B run。
更换布局用新输出目录和 `--resume /path/to/previous/run`。不要修改源 checkpoint。

## 5. 记录和保存

W&B online 必须启用；记录 1D/2D loss、masked NLL、LR、梯度范数、显存、吞吐、
源图像等效 epoch，以及固定 1024 张验证图像的 raw/EMA masked NLL。
这是 development 指标，不是生成 FID。不会创建 `log.txt` 或 TensorBoard 日志。

每跨过一个源图像等效 epoch 更新一次 `latest.safetensors` + `latest.json`，
结束或正常信号停止时也保存；不额外保存 step/best/final 权重。
保留 raw、EMA、AdamW、每 rank RNG、数据游标、配置和 SHA256，写入后逐张量检查。
至少保留 10 GiB 空闲磁盘用于原子写入。没有本机 pilot 的三小时自动停止限制，
也不再按单一 NLL 阈值自动停止。对故障运行先检查原因，不要盲目重启。

## 验收与证据

```bash
python -m unittest h20.test_model h20.test_portability -v
bash -n scripts/install_h20_environment.sh scripts/launch_titok_bert_h20.sh
```

已完成的本地核验见 [交付测试报告](docs/H20_HANDOFF_VALIDATION.md)。
包括四卡从官方初始化 smoke，以及从真实 step10938 checkpoint 完整恢复并更新一步。
尚未接入你的 H20，因此 H20 容量和目标机器全新安装应由上述脚本现场验证。
当前生成方案仍是实验方案，rFID 良好不等于生成 2D 已学好：
[decoder rFID](docs/RFID_DECODER_20260910.md)、
[官方生成器对照](docs/OFFICIAL_COMPARE_20260910.md)。这些历史报告里的本机路径仅作
原实验记录，不是本 handoff 的运行依赖。
