# 实验报告：DehazeMamba × SateHaze1k

## 任务配置

- 数据集：`/home/johnny/code/dataset/SateHaze1k`
- 覆盖强度：Haze1k thin / moderate / thick
- split：train 958 对、valid 105 对、test 135 对（中雾训练集损坏 2 对，跳过）
- 输入：去雾网络 RGB hazy + 单通道 SAR，监督 RGB target
- 模型：DehazeMamba DM-T（当前修正版）
- 训练：150 epochs，batch size 6，AdamW，初始 LR 2e-4，余弦退火到 1e-6，seed 2026
- 数据兼容：启动脚本创建统一为 512×512 的 PNG 规范化缓存；输入/目标按编号匹配，SAR 按官方 split 中排序后的相同位置配对。Pillow 对 RGB 使用 bicubic、对 SAR 使用 bilinear 插值，源文件不修改。中雾训练集 265、271 两张损坏 target 跳过。
- 验证集：保持在适配视图中，但现有训练入口不计算 validation 指标。

## 后台运行记录

启动时间：2026-11-16 18:18:12 +08:00；最终训练进程 PID：3228403，已正常结束。配置快照与日志见 `run_config.txt`、`train.pid`、`logs/train.log`。第一次启动因损坏文件、第二次启动因不同图像尺寸退出；适配脚本增加图像校验/过滤、统一尺寸，随后由 setsid 后台完成训练。

## 结果

150 epochs 已完成；最终训练和测试指标见下方完成结果。

## 复现限制

这是跨数据集实验，不直接等同于论文 MRSHaze/SEN12MS-CR 表格结果。该数据集未在本次运行中验证 SateHaze1k 的 SAR 与 RGB 同位置排序假设之外的元数据配准质量；另外该实现的通道数/SSM 等论文未披露参数存在已记录的实现假设，无法仅凭训练完成宣称作者指标复现。

## 实验完成结果（2026-11-17）

### 训练

- 150/150 epochs 完成，24,000 次优化更新（每轮 160 batches）。
- 最后一轮训练平均 loss：**2.141087**；第一轮：3.463376。
- 训练计时：**8.10 小时**（日志累计值）。
- 最终 checkpoint：`results/satehaze1k_all_levels/last.pt`（约 31.5 MB）。

### Test 结果

使用最终 checkpoint 对 135 张 test 图像计算指标；PSNR/SSIM 对逐图取算术平均，LPIPS 使用 AlexNet v0.1。

| 测试子集 | 图像数 | PSNR ↑ (dB) | SSIM ↑ | LPIPS ↓ |
| --- | ---: | ---: | ---: | ---: |
| Thin | 45 | 24.2514 | 0.9110 | 0.0622 |
| Moderate | 45 | 27.0222 | 0.9429 | 0.0635 |
| Thick | 45 | 21.7668 | 0.8382 | 0.1473 |
| **Overall** | **135** | **24.3468** | **0.8974** | **0.0910** |

逐图指标 CSV：`results/evaluation/metrics.csv`；预测图片：`results/evaluation/predictions/`。数据集各等级图像数量相同，因此总体均值是三组逐图均值的合并均值。测试结果仅说明该代码/配置在本地 SateHaze1k 适配集上的表现，不是论文原数据集结果或官方 leaderboard 对比。

训练日志：`logs/train.log`。本次评估完整跑完 135 张；LPIPS 使用环境中已缓存的 AlexNet 权重，包从临时验证依赖目录加载，未改动训练环境。
