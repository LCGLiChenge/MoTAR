# H20：1D 空间特征输入 + MaskGIT / 融合头联合训练

当前新入口：`scripts/launch_spatial_joint_h20.sh`。旧 `h20/` baseline 保留，
不要混用旧 checkpoint、旧入口和新配置。新版本默认 **官方 TiTok 初始化的新实验**，
不是所有参数随机初始化，也不是恢复本地短试验。

## 已有证据与尚未验证的部分

此前同一批 50k 生成样本的比较：纯 1D（MoT decoder）FID **4.802982**，
固定半幅融合 **5.231282**，学习空间融合 **5.440041**，加入 Router score 的融合
**5.444212**。学习融合头仅训练 100 updates / global batch64；这些数据**没有证明**
学习融合优于半幅，更没有证明它优于纯 1D。此前全强度 2D 的 8+ FID 与降到5附近
主要反映抑制不可靠2D贡献的效果，不能归因于生成器长训。

本次提供的是待验证的联合长训方案，不是上述5.440模型的数值复现包：生成器继续
做 token CE，融合头使用当前 raw 生成器实时生成的1D/2D更新。不再只用固定离线
生成结果训练一个头。**这版还没有长训结果或50k FID；80 epoch不保证成功。**
不能把小模型单测、GPU smoke、teacher MSE下降当成生成质量结论。

学术报告应保留同一 checkpoint / raw或EMA / 同一seed和token的
纯1D、全幅2D、半幅、学习融合四组对照；先开发集选配置，再用新的固定seed做
50k paired FID。报告失败对照、采样步数、CFG、训练预算和额外decoder开销。
不要用重建rFID、4图teacher MSE或不同split的指标替代生成FID。

## 模型和梯度究竟是什么

1. 共享官方 TiTok-L32 ImageBert：24层、width768、16heads。
   1D只接收类别和32个1D位置，**不读取任何2D token/空间特征**。
2. 冻结 MoT199440 EMA 将完成的32个1D code展开为 `[B,256,16,16]` 特征。
   默认 `--memory local`：按E117选中位置取特征，LayerNorm后经**零初始化**线性层
   加到2D输入embedding。它是1D生成的空间底图，不是真实图片或真实2D特征。
3. `--memory full` 是独立、非默认消融：保留local输入分支，同时加一条零输出初始化的
   cross-attention分支，让2D输出查询完整256格1D特征。它不是5.440试验的结构，
   不能声称full优于local。切换此选项必须新开实验，不能resume另一种结构。
4. 1D生成后冻结E117独立决定K64/128和位置；无图像GT、无真实2D特征、无oracle路由。
   Router不参与训练，K不是固定37.5%：单张25%或50%，数据集平均比例不是逐图常数。
5. 融合头为16,897参数空间卷积头，score通道置零，初始选中格alpha=0.5，未选中格不改。
   `f = f_base + alpha * (f_mixed - f_base)`；alpha是隐空间贡献比例，不是像素透明度。
   E117仍用score选择位置，但本版**不把score作为融合头的有效输入**。

两个训练目标独立归一化、分别裁剪梯度：

| 目标 | 数据/条件 | 更新对象 |
|---|---|---|
| 1.5 × CE1D + CE2D | 全量TRAIN packed真实code；2D条件为真实完整1D及其冻结空间特征 | 共享BERT、1D头、2D参数、空间分支 |
| 图像MSE，权重1 | 当前raw模型生成1D→E117→生成2D；目标是冻结原生TiTok对**同一生成1D**的解码 | 仅融合头 |

**硬采样处不传梯度**。图像MSE不会更新MaskGIT；这叫联合训练中的交替更新，
不是通过离散采样的端到端图像监督。2D CE不会乘alpha，因此不能靠关闭2D逃避CE。
本版也**没有**新增“生成1D条件下的2D token CE”或straight-through估计。
生成前缀用于在线融合损失，不要把它误写为整个2D token训练已经解决暴露偏差。

CE沿用arccos mask、0.1 label smoothing、可见位置权重0.1和逐样本归一化。
前600步只更新新增2D/空间参数，融合lr=0；之后共享BERT和1D部分也更新。
新2D lr1e-4、共享/1D lr1e-5，AdamW betas(.9,.96)、wd.03；
融合lr3e-3、betas(.9,.999)、wd0。lr分别warm up50 /100 /50步。
EMA=.999。CE为bf16；冻结特征提取、图像解码和融合损失为fp32。

在线1D采样：官方 `ImageBert.generate`，8步、CFG4.5 linear、randomize temp9.5。
2D：4步margin confidence、CFG4.5 constant、arccos、退火Gumbel、已知token不重新mask。
训练期每个optimizer step全局融合16图，类别循环覆盖1000类；独立counter seed，
不消耗CE的mask/dropout RNG。模型更新后下一步重新生成，不复用固定teacher cache。

注意：native teacher偏好、alpha饱和/退化成纯1D、共享主干使1D退化仍是未解决风险。
W&B记录mean_alpha、pure1d/full2d/half/learned相对teacher的MSE和raw/EMA token NLL；
这些仅用于诊断，最终仍需同token图像对照和FID，不能只凭masked_nll2d判成功。

## 新服务器：完整命令

需要Linux、NVIDIA驱动支持CUDA12.8、git、Conda/Miniforge；建议128GB主机内存，
至少80GB可用磁盘。GPU仅使用用户批准的明确列表；支持1/2/4/8卡H20。
安装文件和旧baseline共用，环境已装且符合固定版本可跳过安装。

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
# 如需团队workspace，设置有写权限的WANDB_ENTITY；不写入任何密钥。
export MOTAR_OUTPUT="$PWD/results/spatial-joint-local-h20-seed0"
bash scripts/launch_spatial_joint_h20.sh \
  --assets "$MOTAR_ASSETS" --output "$MOTAR_OUTPUT" --epochs 80
```

不要将本机GPU编号照抄到另一个服务器；上面8卡仅在8卡已经获准时使用。
所有路径均相对clone/用户指定assets与output。不依赖旧服务器目录或ImageNet原图。
如果用分离的shell调用，使用 `conda run --no-capture-output -n motar-h20 ...`，
不要假定上一次shell的conda激活仍有效。

资产下载约11.672GB，不包括Hugging Face缓存副本。`joint`=train+inference，
**不下载旧step10938续训状态**。文件SHA256、size和不可变revision均在
[`configs/h20_assets.json`](../configs/h20_assets.json)。脚本校验后才加载，不覆写哈希不符文件。
大文件的.parts传输块自动逐块校验、合并为原文件，无需手动拼接。
网络下载慢可先设置 `HF_HUB_DISABLE_XET=1` 重试下载，不要绕过哈希检查。

| 权重/数据 | Hugging Face仓库及位置 | 联合训练用途 |
|---|---|---|
| train/val packed code | [Chloeeeeeeee123/MoT-1](https://huggingface.co/Chloeeeeeeee123/MoT-1/tree/main/h20-titok-bert-20260910/codes) | 全量训练、固定验证 |
| train/val E117 route、E117权重 | [Chloeeeeeeee123/MoT-1](https://huggingface.co/Chloeeeeeeee123/MoT-1/tree/main)；具体文件见manifest | 真实code路由缓存、生成时在线路由 |
| generator_titok_l32.bin | [fun-research/TiTok](https://huggingface.co/fun-research/TiTok/tree/main) | 官方MaskGIT初始化 |
| tokenizer_titok_l32.bin | 同TiTok仓库 | 原生1D codebook与teacher解码；MoT里不含这部分权重 |
| latest.pt，model_ema step199440 | [sophiaa/MoT-1-checkpoints](https://huggingface.co/sophiaa/MoT-1-checkpoints/tree/main) | MoT空间底图与LlamaGen VQ/codebook/decoder |

无需原版 `FoundationVision/LlamaGen/vq_ds16_c2i.pt`；VQ取MoT EMA。
也无需新的未发布本地空间/融合checkpoint：新增参数现场初始化。
这些公开资产的位置未改，本次不上传任何私人W&B登录信息或ImageNet原图。

## 显存测试、smoke和epoch含义

启动器先严格检查环境、全部资产、数据完整性和GPU空闲情况，然后按降序尝试
保持CE全局batch2048的micro候选。真实探测含最坏K128、Adam、EMA、冻结模型、
在线生成、融合图像反传，以及32图dev前向；reserved显存上限默认92%。
融合全局batch固定16，默认每卡fusion micro2，其余通过独立累积完成。
CE和融合micro改变不会无意中重标定两项loss。由于global batch约束和安全余量，
实际占用不保证恰好92%；不为填显存擅自增大全局batch。

容量探测允许隔离子进程中的OOM并尝试更小候选；正式训练不跳过OOM/NaN批次。
单卡探测不是DDP容量证明，随后**必须**在实际卡数和选定micro上运行：

1. 4步独立DDP smoke，覆盖warmup→joint切换并保存完整状态。
2. 读取刚保存的完整raw/EMA/Adam/RNG/cursor，恢复到第5步并再保存。
3. 两段均成功、SHA与readback通过，才开始全新正式run；**不继承smoke参数**。

使用 `--plan-only` 只检查/打印方案；`--smoke-only` 完成探测和两段smoke后停止，
不会正式训练。smoke-only的output已被使用，之后正式训练需另取一个新output。
成功凭据在 `capacity/verified.json`，测试run独立放在 `capacity/distributed_smoke/`。
两个smoke leg只保留其自己的latest；正式run也只保留自己的latest，不存epoch快照。

80 epoch指80个**ImageNet源图像等效epoch**：1,281,167源图、2种已打包augmentation，
global2048，目标ceil(80×1,281,167/2048)=**50,046 optimizer steps**。
完整packed数据有2,562,334项，80个源图等效epoch约为40次packed pass，而非800epoch。
这不是对另一台机器上正在跑的80epoch任务做resume或改配置。

正式输出每跨1个源图等效epoch更新 `latest.safetensors` 和 `latest.json`，结束或
收到SIGINT/SIGTERM时也保存；包含raw+EMA+Adam+rank RNG+数据游标。
不存额外历史checkpoint，不生成log.txt/TensorBoard；W&B online记录曲线。
W&B自身必要的本地同步缓存仍存在，不能将其解释为完全没有本地文件。
保留至少10GiB保存空间；启动前要求20GiB以覆盖测试状态和临时写入。
元数据和tensor文件分别原子替换；中途断电造成两者不匹配时会拒绝加载，而非静默续训。

明确恢复本版完整checkpoint时：

```bash
bash scripts/launch_spatial_joint_h20.sh \
  --assets "$MOTAR_ASSETS" --output "$MOTAR_OUTPUT" \
  --resume "$MOTAR_OUTPUT" --epochs 80
```

恢复前同样重新探测与smoke；epochs始终是总目标。相同布局恢复RNG/游标；
world/micro变化必须使用新output，并显式从旧output恢复，数据进入下一packed pass，
不声称bitwise连续。旧baseline/pilot不是本版格式，拒绝混用。

## 查看同token图像

训练独立退出或另有获准GPU时，使用一个新输出目录：

```bash
CUDA_VISIBLE_DEVICES=0 python -m h20_joint.sample \
  --assets "$MOTAR_ASSETS" --checkpoint "$MOTAR_OUTPUT" \
  --output "$PWD/results/joint-preview-ema" --num-images 16 --batch 2 --state ema
```

保存同一组token的pure1d、full2d、half、learned和native1d_teacher PNG及token档案。
这个命令**不计算FID**，也不要一边覆写latest一边读取；先稳定训练checkpoint。
如果需要正式50k FID，应另行固定评估协议与Inception参考统计，不能从预览估算FID。

本机验证记录见 [H20_JOINT_VALIDATION.md](H20_JOINT_VALIDATION.md)。
