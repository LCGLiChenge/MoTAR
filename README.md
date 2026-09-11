# MoTAR：TiTok-BERT unified MaskGIT

当前新方案：**1D空间特征输入 + MaskGIT与融合头联合训练**。
入口是 `scripts/launch_spatial_joint_h20.sh`，完整流程见
[H20联合训练说明](docs/H20_SPATIAL_JOINT.md)。

默认从官方TiTok MaskGIT初始化一个新实验；新增2D/空间分支/融合头重新学习，
**不是整个模型随机初始化，也不resume本地短试验**。1D forward不读取2D。

本版是待验证的长训方案，尚无新版50k FID结果。此前学习融合5.440仍不如
同批半幅5.231和纯1D 4.803，不能把链路测试通过解释为质量已提升。

## 在新H20服务器启动

环境已装且符合固定版本时可跳过安装。只使用用户批准的GPU编号。

```bash
git clone https://github.com/LCGLiChenge/MoTAR.git
cd MoTAR
bash scripts/install_h20_environment.sh
conda activate motar-h20
wandb login

export MOTAR_ASSETS="$PWD/assets/h20"
python -m h20.assets --root "$MOTAR_ASSETS" --profile joint
python -m h20.preflight --environment-only
python -m unittest h20.test_model h20.test_portability h20_joint.test_joint -v

mkdir -p results
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_PROJECT=motar-maskgit
export MOTAR_OUTPUT="$PWD/results/spatial-joint-local-h20-seed0"
bash scripts/launch_spatial_joint_h20.sh \
  --assets "$MOTAR_ASSETS" --output "$MOTAR_OUTPUT" --epochs 80
```

启动器自动探测实际显存，保留CE全局batch2048、融合全局batch16，
再做实际卡数下的完整保存/恢复smoke；通过后才从官方权重开始正式训练。
默认80个源图像等效epoch，即50,046步。每跨1epoch更新自己的latest，
不保留epoch快照；W&B online记录，不创建log.txt。

全部必需大文件已经列在
[HF资产清单](configs/h20_assets.json)，`--profile joint`下载约11.672GB；
不需要旧机器目录、本地ImageNet原图、旧step10938状态或新上传的短试验权重。
原生TiTok tokenizer用于1D codebook/teacher，MoT EMA提供空间底图和VQ decoder；
不下载原版LlamaGen VQ权重。

在开始前请完整阅读：

- [训练目标、HF位置、初始化、显存、resume与采样命令](docs/H20_SPATIAL_JOINT.md)
- [实际验证记录及未验证范围](docs/H20_JOINT_VALIDATION.md)

旧TiTok-BERT无空间输入baseline说明保留在
[README_BASELINE_H20.md](README_BASELINE_H20.md)，其`h20/`训练实现不变。
旧H200 scratch和causal AR均是历史入口，不能与新版checkpoint混用。
