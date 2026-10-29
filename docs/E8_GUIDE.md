# E8：DF2K、逐倍率训练与成本评估

E8 仅扩展数据、倍率和训练预算，复用 E4 选定的 uncertainty、local alpha、Depth encoder、confidence、loss 与 RGB 主干。原有配置及 ×4 数据集路径保留。新增脚本均含源码注释。

## 前置实测条件

没有 E4–E7 的真实报告和权重时可检查代码和准备清单，不能启动正式 E8。准备脚本核对 E6 独立训练 lineage、E6/E7 权重 SHA256 与 E4 选择记录。E4 五集平均 >0.03 dB、Urban100 ≥0、Manga109 >0；E5 完整五集、正确 Depth 优于零/打乱且打乱对照不落入 ±0.01 dB 近似平局；E6 五集为 strong、Urban100 ≥0、Manga109 >0。

附件没有给出 E7 的全局通过标志。这里采用显式的保守入口规则：E7 为 10,000 次 bootstrap，Urban100/Manga109 的 95% 区间下界 >0，其他数据集均值非负且区间上界 >0。其他集合跨零仍只称趋势。该入口规则是工程解释，不是附件新增的实验结论；每项判据写入计划，失败即停止。

## 数据准备

在原 CUDA 环境、项目根目录运行；不改动原有环境文件：

```bash
git clone https://github.com/KaiXu-HIT/v3.2-M3SR-MambaIRv2-E8.git
cd v3.2-M3SR-MambaIRv2-E8
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python scripts/udr/check_e8_scaling.py
python scripts/udr/prepare_e8_scaling.py --write-data-template options/E8_data.json
```

填写 `options/E8_data.json` 的空值：Flickr2K HR/LR/Depth，以及 ×2/×3 的 DIV2K、验证集、五测试集 LR/Depth。×4 原有路径和所有原有 HR 路径受保护。DIV2K 800 张 + Flickr2K 2650 张全部参与训练；验证仍为 DIV2K val。准备时读取所有配对样本，校验 LR/Depth 对齐及 HR 倍率。两个训练目录须使用相同文件名规则，例如 `000001.png` → `000001x4.png`；Depth 使用同名同扩展名。仅设置新目录，不得通过复制 DIV2K 冒充 Flickr2K。LR 为与原实验相同的 bicubic；Depth 必须由该倍率 LR RGB 经原有 estimator 产生，不能读取 HR。数据来源这一点需在生成缓存时保证，尺寸检查不能证明它。

## 训练与正式测试

默认每 seed 先训练 RGB 控制 480k，再训练 UDR A 30k + B 450k。三 seed 沿用 E6 的独立权重；RGB 和 UDR 各从对应祖先 warm-start，新增更新数相同。RGB LR=1e-5；UDR A 冻结 RGB，B RGB LR=1e-5、新模块 LR=1e-4；继承所选 lambda_U/lambda_A 和 L1，不增加损失。U3 教师固定为同倍率、同 seed 已完成的 DF2K RGB 控制。LR patch 固定 48×48，HR patch 随倍率变化。原 E0–E7 的固定 DIV2K baseline 继续保留为历史参照，E8 的数据升级效果以同数据/同倍率 DF2K RGB 控制衡量，不把两种 baseline 混用。

UDR 继承了更多上游训练，新增预算相同不等于总成本相同；论文需披露 E6 祖先训练和 E8 新增预算。只使用计划指定的最终权重，不依据测试集选择最优 checkpoint。

若实测报告在不同仓库，设置下列根路径；按实际存放位置修改：

```bash
E4ROOT=/home/BRAIN/xukai/code/v3.2-M3SR-MambaIRv2-E4
E5ROOT=/home/BRAIN/xukai/code/v3.2-M3SR-MambaIRv2-E5
E6ROOT=/home/BRAIN/xukai/code/v3.2-M3SR-MambaIRv2-E6
E7ROOT=/home/BRAIN/xukai/code/v3.2-M3SR-MambaIRv2-E7

python scripts/udr/prepare_e8_scaling.py --scale 4 \
  --data-manifest options/E8_data.json \
  --e4-report "$E4ROOT/results/E4_udrv2/ablation/E4_ablation.json" \
  --e5-report "$E5ROOT/results/E5_depth_causality/E5_causality.json" \
  --e6-plan "$E6ROOT/experiments/E6_plan/E6_plan.json" \
  --e6-report "$E6ROOT/results/E6_multiseed/E6_multiseed.json" \
  --e7-report "$E7ROOT/results/E7_image_statistics/E7_statistics.json"

python scripts/udr/run_e8_training.py --plan experiments/E8_scaling/x4/E8_plan.json --dry-run
CUDA_VISIBLE_DEVICES=0 python scripts/udr/run_e8_training.py --plan experiments/E8_scaling/x4/E8_plan.json
CUDA_VISIBLE_DEVICES=0 python scripts/udr/evaluate_e8_scaling.py --plan experiments/E8_scaling/x4/E8_plan.json
```

×4 测试生成 `E8_metrics.json` 后才可准备 ×2；×3 同样等待 ×2 完成。下一倍率通过后仍须审查报告，不把“脚本完成”解释为性能目标已达成。×2/×3 均从同 seed 的 E8 ×4 共享特征 warm-start，全部 `upsample.*` 重建层重新初始化，RGB/UDR 使用完全相同的新重建层，即使部分旧 tensor 形状相同也不复用。其余所有 tensor 必须严格匹配；不使用 `strict=False`。

对 ×2、×3 依次执行（上面的报告参数完整沿用）：

```bash
for SCALE in 2 3; do
  python scripts/udr/prepare_e8_scaling.py --scale "$SCALE" \
    --data-manifest options/E8_data.json \
    --e4-report "$E4ROOT/results/E4_udrv2/ablation/E4_ablation.json" \
    --e5-report "$E5ROOT/results/E5_depth_causality/E5_causality.json" \
    --e6-plan "$E6ROOT/experiments/E6_plan/E6_plan.json" \
    --e6-report "$E6ROOT/results/E6_multiseed/E6_multiseed.json" \
    --e7-report "$E7ROOT/results/E7_image_statistics/E7_statistics.json" && \
  CUDA_VISIBLE_DEVICES=0 python scripts/udr/run_e8_training.py --plan "experiments/E8_scaling/x${SCALE}/E8_plan.json" && \
  CUDA_VISIBLE_DEVICES=0 python scripts/udr/evaluate_e8_scaling.py --plan "experiments/E8_scaling/x${SCALE}/E8_plan.json" || break
done
```

中断恢复会核验已完成任务，按**当前阶段**最近的 optimizer/scheduler `.state` 恢复，禁止跨 Phase A/B 共用状态。例：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/udr/run_e8_training.py --plan experiments/E8_scaling/x4/E8_plan.json --resume
```

逐倍率输出 `results/E8_scaling/xN/E8_metrics.json`、`E8_per_image.csv`、`E8_summary.md`：相同 seed 的 PSNR/SSIM 与 ΔPSNR mean±std、五集等权平均、10,000 次逐图 bootstrap。保持原分块/重叠、排序、uint8 Y-channel、crop=scale。报告最低 +0.05、较强 +0.08 dB、Urban100/Manga109 正增益及 Manga109 置信区间；“clearly positive”没有虚构额外 dB 阈值。

## 复杂度与 Depth 单独成本

统一 batch=1、FP32、LR=160×160、TF32 关闭、预热20次、重复100次。Params 为完整 SR 模型参数；MACs/FLOPs 显式包含 Conv/Linear、attention、routing prompt、scan projection、实数 selective scan 递推及 Sobel。它们是注明覆盖范围的算术估计，activation/normalization/排序/索引等未计入算术总数；这些操作及原诊断统计均计入实测延迟，不能把 MACs/FLOPs 描述成完整硬件指令数。报告 mean/std/median/P95 latency、峰值 allocated/reserved VRAM。

预处理采用实际 P2/P98 normalization、RGB/Depth tensor conversion 和 H2D，独立计时，不含磁盘 IO。Depth estimator 单独加载，计参数、预处理、推理/后处理延迟和显存；必须使用产生实验 Depth 的同一个 estimator。无法提供它时，成本报告拒绝填充，而不是记为零。

内置 `e8_depth_adapter:build` 接收原 estimator 的 TorchScript 导出，导出图须包含原 resize/normalize/预测/恢复到 LR 的完整处理，输入 FP32 RGB `[0,1]` BCHW、输出 `[B,1,H,W]`。已有 eager 实现可提供 `module:function`：工厂接收 `checkpoint, device`，返回含 `.name`、`.model`、`.preprocess(uint8_RGB_HWC)`、`.predict(tensor)` 的对象，后者返回对齐 LR 的 FP32 depth tensor。不会选用未经验证的新 Depth estimator。

```bash
DEPTH_CKPT=/absolute/path/to/original_depth_estimator_with_prepost.ts
for SCALE in 4 2 3; do
  CUDA_VISIBLE_DEVICES=0 python scripts/udr/profile_e8_scaling.py \
    --plan "experiments/E8_scaling/x${SCALE}/E8_plan.json" \
    --depth-factory scripts.udr.e8_depth_adapter:build \
    --depth-checkpoint "$DEPTH_CKPT" --warmup 20 --repeats 100 || break
done
```

`E8_complexity.json/md` 将 estimator 成本单列；在线总延迟为各串行阶段单独实测值之和，明确不冒充流水端到端测量。SR 计时采用单块160×160网络前向；五集精度采用原分块测试，两个协议分别注明。

## 公开模型对比

```bash
python scripts/udr/compare_e8_public.py --write-template options/E8_public_models.json
# 填写并启用各模型的公开权重/统一复现条目，然后运行：
CUDA_VISIBLE_DEVICES=0 python scripts/udr/compare_e8_public.py \
  --manifest options/E8_public_models.json \
  --metrics results/E8_scaling/x4/E8_metrics.json
```

清单列出 MambaIR、MambaIRv2、SwinIR、HAT、SRFormer、RCAN、EDSR；可追加其他模型。每项必须提供准确训练数据 DIV2K/DF2K、倍率、公开来源 URL、checkpoint、原模型配置及 factory；权重/配置/adapter 均存 SHA256。内置 factory 仅加载本仓库真实注册架构，其他官方仓库需在原环境提供同签名的本地 adapter：`build(checkpoint, config, device) -> nn.Module`，完整严格加载原权重，输入/输出 RGB `[0,1]`，带 `.upscale`。不要为通过加载而忽略 tensor。

对比统一使用 E8 相同五集、原分块、uint8 Y/crop=scale 和 matched inference seeds；实际重新测量，不导入论文表格数值。DIV2K 与 DF2K 分表，只有 DF2K 对手可与 E8 DF2K 同条件比较。公共单 checkpoint 的重复推理 std 与本实验独立训练 seed std 含义不同，表中明确标注。缺少任一指定模型时报告 `complete_comparison=false` 并列出缺项，不称已完成全部对比。×2/×3 需各自对应公开权重及清单，命令替换相应 metrics 路径。

本机离线自检不代替 CUDA 训练、真实五集评估或真实成本测量；结果仅由正式命令生成。
