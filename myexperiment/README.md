# SateHaze1k DehazeMamba 实验

本实验将 SateHaze1k 的薄雾、中雾、浓雾三个子集一起训练和测试。启动脚本在 `script/run_satehaze.sh`，它会按文件排序配对各 split 的 input/target/SAR，并建立 MRSHaze 目录格式的规范化缓存视图；原始数据不会复制或修改。

默认参数沿用项目论文设置：DM-T、150 epochs、batch size 6、seed 2026、6 workers。数据映射为 958 train / 105 valid / 135 test（中雾训练集 2 张 target 损坏并跳过）。当前训练入口只接收 train split，故 valid 保留在适配目录供后续检查，不参与梯度更新；训练结束后 test 评估需要单独启动 `evaluate_dehaze.py`。图像跨模态几何配对依赖 SateHaze1k 各 split 的排序一致性，启动前适配器会检查配对数量、解码文件并跳过损坏图像；将 RGB 以 bicubic、SAR 以 bilinear 缩放至 512×512，源数据不会修改。

## 后台任务状态

查看状态：

```bash
cat train.pid
ps -p "$(cat train.pid)" -o pid,stat,etime,cmd
 tail -f logs/train.log
```

停止任务：

```bash
kill "$(cat train.pid)"
```

训练权重每个 epoch 写入 `results/satehaze1k_all_levels/last.pt`。评估完成后将指标 CSV、汇总及预测保存到本目录下的 `results/evaluation/`。

启动配置、实际 PID、数据配对信息和训练结束后的评估结果追加记录在 `experiment_report.md`。

训练已完成 150 epochs，完整测试 135 张的平均结果为 PSNR 24.3468 dB、SSIM 0.897361、LPIPS 0.090995。各雾强度分项、训练 loss 与耗时见 `experiment_report.md`；逐图指标见 `results/evaluation/metrics.csv`。

## innovate_1：低频引导的小波输出细化

实现、检查点兼容方式及运行命令见 [innovate_1 实现说明](experiment/innovate_1_implementation.md)。使用已有规范化数据时运行 `bash myexperiment/script/run_innovate_1.sh`（从仓库根目录），默认写入独立的 `myexperiment/results/innovate_1`。原 `run_satehaze.sh` 仍运行 baseline。该方案尚未完成正式训练与性能验证。
