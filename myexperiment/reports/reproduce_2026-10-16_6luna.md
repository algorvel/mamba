# DehazeMamba / MRSHaze 本地复现实验记录

- **记录日期：** 2026-11-16
- **实验对象：** DehazeMamba，数据集 MRSHaze
- **实验状态：** 150 轮训练和完整测试集评估均已完成，训练与评估进程均正常退出
- **代码版本：** `modules/dehazeMamba`，分支 `dehazeMamba`，提交 `52257ae`（`Add DehazeMamba reproduction`）
- **记录范围：** 本地预检、实现验证、配置尝试、完整训练、完整测试评估及可核实结果；不混入其他论文/子模块实验。

## 1. 结果摘要

| 项目 | 本地结果 |
|---|---:|
| 训练集 | 6,000 对 |
| 测试集 | 2,000 对 |
| 训练轮数 / optimizer steps | 150 / 150,000 |
| 最终训练集 epoch mean loss | 0.0161133243 |
| 测试集 mean PSNR | 30.9914047 dB |
| 测试集 mean SSIM | 0.9304472 |
| LPIPS | 未计算 |
| 训练耗时 | 75.38 小时（训练进程报告值） |
| 全测试集评估耗时 | 约 4 分 39 秒 |

论文 Table II 的 DM-T 报告 PSNR 32.69 dB、SSIM 0.9507、LPIPS 0.1148。本地与论文数值有差异；本地网络参数为 15,631,827，论文 DM-T 报告 6.65M。由于论文没有充分披露通道宽度、状态维度和 VSS 的精确实现，本地实现是有记录假设的近似复现，不能描述成与作者实现完全相同。数据规模吻合，但论文未提供测试文件清单/划分种子，无法确认两边测试样本逐一相同。

## 2. 设备与软件环境

### 本地设备

| 项目 | 记录值 |
|---|---|
| 操作系统/主机 | Ubuntu，主机名 `ubuntu-3090` |
| GPU | NVIDIA GeForce RTX 3090，24,576 MiB 显存 |
| NVIDIA 驱动 | 535.309.01；当时 `nvidia-smi` 报告 CUDA 兼容版本 12.2 |
| 系统 CUDA 编译器 | `nvcc` 12.0 |
| CPU | 未记录型号，不能补写 |
| 系统内存 | 一次训练期间状态快照：总计约 62 GiB，可用约 45 GiB |
| 磁盘 | 一次训练期间状态快照：文件系统剩余约 542 GB |

训练期间抽查 GPU 状态：利用率约 99–100%，显存使用约 22.1/24 GiB，温度约 68–69°C，功耗约 343/350 W。这些是运行时抽查值，不是完整的温度/功耗历史曲线。

### Conda 环境

- 环境名：`mamba`
- Python：3.10
- PyTorch：`2.7.1+cu118`（PyTorch wheel 的 CUDA 版本为 11.8）
- Torchvision：`0.22.1+cu118`
- Triton：`3.3.1`
- Transformers：`4.51.3`
- 其他记录到的依赖：NumPy `<2`、Pillow、Ninja、einops、tqdm、huggingface-hub
- 训练使用 BF16 autocast（RTX 3090 支持 BF16）；评估脚本也使用 BF16 autocast，随后以 float32 计算指标。

**CUDA 扩展兼容性处理：** PyTorch wheel 为 CUDA 11.8，而本机 `nvcc` 为 CUDA 12.0。使用 `tools/build_selective_scan.py --allow-major-mismatch` 跳过 PyTorch 扩展构建中的版本守卫，并构建 `selective_scan_cuda`。这不是推荐的通用安装方式；本机执行了 selective-scan 前向/反向 smoke check。跨机器复现时建议匹配 PyTorch 与 CUDA Toolkit 版本。

## 3. 数据集与输入

- 压缩包：`/home/johnny/code/dataset/MRSHaze.zip`
- 解压目录：`/home/johnny/code/dataset/MRSHaze`
- 当前磁盘核查：解压目录约 8.8 GB，共 24,000 张 PNG（6,000 train × 3 模态 + 2,000 test × 3 模态）。记录中原压缩包路径为上项，但本次写报告时该 ZIP 已不在该路径；下载源 URL 和 ZIP 校验和未留存，因此这里不声称完成了来源校验。
- 使用目录结构：

  ~~~
  MRSHaze/
    train/{GT,hazy,SAR}/*.png
    test/{GT,hazy,SAR}/*.png
  ~~~

- 本地检查结果：训练集 GT/hazy/SAR 各 6,000 张且文件名完全匹配；测试集各 2,000 张且文件名完全匹配。测试集三个目录均检查到 512×512 图像；训练流程也成功读完 6,000 对训练样本。
- 输入含义：`hazy` 是有雾 RGB 输入，`SAR` 是灰度 SAR 引导图，`GT` 是清晰 RGB 目标图。数据读取按相同文件名配对，并将像素值缩放到 `[0,1]`。
- 数据划分数量与论文报告的 6,000/2,000 一致。论文只称随机划分，没有发布可核验的逐文件 manifest 或 seed，因此不声称测试集成员完全相同。

## 4. 模型实现和训练配置

### 模型配置

| 参数 | 本地值 |
|---|---|
| 模型 | DehazeMamba 近似实现，DM-T 阶段深度 |
| Stage depths | `[2, 2, 2, 1, 1]` |
| `base_channels` | 24，后续 stage 按实现递增 |
| `state_dim` | 8 |
| 总参数量 | 15,631,827 |
| 输入 | Hazy RGB + SAR 灰度图 |
| 输出 | RGB 去雾图，带输入残差路径 |

实现了双分支编码、HPDM/PFM 风格跨模态融合和 Mamba/VSS 特征块。由于论文未指定 DM-T 通道宽度、selective-scan 状态维度、精确 VSS 结构和部分融合实现，本地采用了文档记录的选择：宽度从 24 起、state size 8、共享双向轴向行/列扫描，并在最深两个编码尺度进行融合。实现假设见 `modules/dehazeMamba/DEHAZEMAMBA.md`。

### 训练参数

~~~
data_root=/home/johnny/code/dataset/MRSHaze
output_dir=outputs/dehazemamba
epochs=150
batch_size=6
base_channels=24
state_dim=8
workers=6
seed=2026
resume=None
max_steps=None
~~~

- 优化器：AdamW，初始学习率 `2e-4`；使用 PyTorch 默认 weight decay（论文未说明 weight decay）。
- 学习率：CosineAnnealingLR，150 轮，从 `2e-4` 到 `1e-6`。
- 损失：`L = L_spatial + 0.1 × L_frequency`。空间项为 RGB L1；频率项为正交归一化 `rFFT2` 的实部/虚部 L1。论文给出频率权重 0.1，本地 FFT normalization 是实现假设。
- 每轮：6,000 样本、batch 6、`drop_last=False`，共 1,000 个 optimizer steps；总计 150,000 steps。
- 数据顺序：训练 DataLoader shuffle；Python/NumPy/PyTorch seed 均设为 2026。启用了 `cudnn.benchmark=True` 与 TF32，因此不能将该运行描述为严格 bitwise deterministic。
- 未加载预训练权重；`resume=None`，本次从初始化权重开始训练。

完整训练命令（在 `modules/dehazeMamba` 目录运行）：

~~~
/home/johnny/miniconda3/envs/mamba/bin/python train_dehaze.py --data-root /home/johnny/code/dataset/MRSHaze --output-dir outputs/dehazemamba --epochs 150 --batch-size 6 --base-channels 24 --state-dim 8 --workers 6
~~~

## 5. 实验过程和尝试记录

训练进程启动时间在运行期间由系统进程信息记录为约 **2026-09-29 22:13:43（北京时间）**。训练脚本最终报告 `elapsed_hours=75.38`（约 75 小时 22 分 48 秒）；这个脚本计时是 epoch 循环耗时，不含预检/环境构建和之后的测试集评估。按启动时间与四舍五入后的时长推算，训练约于 **2026-10-03 01:36（北京时间）** 完成；完整测试评估约再用 4 分 39 秒。结束时刻是推算值，未单独保存系统级结束时间戳。训练及评估都正常退出。

| 阶段 | 实验内容 | 结果/处理 |
|---|---|---|
| 环境预检 | Python 编译检查、数据加载、文件配对、模型前向/反向、selective-scan CUDA 前向/反向 | 均通过；CUDA Toolkit 与 PyTorch CUDA 版本不匹配，通过上文的构建 helper 处理 |
| 显存试配 | `base_channels=32`、batch 6 的 512×512 训练 smoke run | 显存 OOM；切换到 `base_channels=24` 后 batch 6 前向/反向可运行 |
| 内存格式试验 | channels-last | 实测比默认格式慢，撤回 |
| 单样本评估 smoke | 用 smoke checkpoint 跑一个测试样本 | PSNR 22.318 dB、SSIM 0.728737；这是 smoke 结果，不是最终模型结果，不与论文表格比较 |
| 完整训练 | 6,000 对训练数据、150 轮、150,000 steps | 完成；最终 mean loss `0.0161133243`，训练报告用时 75.38 h |
| 完整测试 | 2,000 对测试数据，batch size 1，workers 4 | 完成；测试进度 `2000/2000`，耗时约 4:39，评估进程退出码 0 |

### 已核对的训练 loss 检查点

以下为运行期间逐次读取 checkpoint 时记录的 epoch mean loss。完整逐 epoch CSV/log 没有保存在产物目录，因此除列出的检查点外，不补造中间轮次数据。

| Epoch | Mean loss | Epoch | Mean loss | Epoch | Mean loss |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.07298 | 10 | 0.04484 | 19 | 0.03520 |
| 2 | 0.07298 | 11 | 0.04353 | 20 | 0.03400 |
| 3 | 0.06537 | 12 | 0.04191 | 21 | 0.03346 |
| 4 | 0.05994 | 13 | 0.04075 | 22 | 0.03267 |
| 5 | 0.05597 | 14 | 0.03951 | 23 | 0.03210 |
| 6 | 0.05295 | 15 | 0.03970 | 24 | 0.03184 |
| 7 | 0.05067 | 16 | 0.03723 | 25 | 0.0313230824 |
| 8 | 0.04854 | 17 | 0.03620 | 76 | 0.0207643147 |
| 9 | 0.04661 | 18 | 0.03527 | 150 | 0.0161133243 |

## 6. 最终评估方式和结果

- 测试集：`MRSHaze/test`，2,000 对；不打乱顺序。
- 预测先 clamp 到 `[0,1]`。PSNR 对每张 RGB 图计算 `[0,1]` 范围上的 MSE，再对 2,000 张逐图 PSNR 求算术平均。
- SSIM 使用本地 evaluator 中的 11×11 Gaussian window、sigma 1.5、`C1=0.01²`、`C2=0.03²`，逐图计算后取平均。
- LPIPS 未安装/未执行；论文的 LPIPS `0.1148` 与本地不可直接比较。
- evaluator 的 BF16 推理和具体 SSIM 实现未证明与论文官方评估代码完全一致，这是指标比较的一项限制。

| 指标 | 本地 | 论文 DM-T | 本地 − 论文 |
|---|---:|---:|---:|
| PSNR | 30.9914047 dB | 32.69 dB | −1.6985953 dB |
| SSIM | 0.9304472 | 0.9507 | −0.0202528 |
| LPIPS | 未计算 | 0.1148 | 不可比较 |
| 参数量 | 15.631827 M | 6.65 M | 本地约为论文的 2.35 倍 |

这些差异说明本地分数没有达到论文 DM-T 表格数值；由于网络参数规模、实现细节、可能的测试样本成员及指标实现并未完全对齐，不能单独据此归因于数据或论文结果。

## 7. 产物路径

| 产物 | 路径 |
|---|---|
| 最终检查点（模型/优化器/调度器状态和配置） | `modules/dehazeMamba/outputs/dehazemamba/last.pt`（约 188 MB，188,364,463 bytes） |
| 测试逐图指标 CSV | `modules/dehazeMamba/outputs/dehazemamba/evaluation/metrics.csv` |
| 预测图 | `modules/dehazeMamba/outputs/dehazemamba/evaluation/predictions/`（2,000 张，覆盖测试集文件名） |
| 数据加载实现 | `modules/dehazeMamba/dehazemamba/data.py` |
| 模型实现 | `modules/dehazeMamba/dehazemamba/model.py` |
| 损失实现 | `modules/dehazeMamba/dehazemamba/losses.py` |
| 训练脚本 | `modules/dehazeMamba/train_dehaze.py` |
| 评估脚本 | `modules/dehazeMamba/evaluate_dehaze.py` |
| 复现假设与环境说明 | `modules/dehazeMamba/DEHAZEMAMBA.md` |

## 8. 复现限制与日志完整性

1. 原论文未公布足够的架构细节，故本地实现是有记录假设的复现，不是作者官方实现的逐项重建。
2. 论文称随机划分 6,000/2,000，但没有随论文提供可核验的测试文件 manifest/seed；本地数量和配对结构匹配，样本身份不能确认完全一致。
3. 本地没有计算 LPIPS；指标代码、精度模式也未与论文官方评估脚本对齐。
4. 最终 checkpoint 保存了模型、优化器、调度器、epoch、global step、最终 mean loss 及配置。运行过程的完整逐 batch 日志及所有 150 轮的 loss CSV 没有保留；本记录只列出能从检查点/终端记录核实的数值。
5. 设备快照不是持续硬件监控数据；CPU 型号和运行期间温度/功耗峰值未记录。
