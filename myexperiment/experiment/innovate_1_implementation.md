# innovate_1 实现与运行

实现对应设计稿 P3–P16，网络开关为 `wavelet_refinement=True`，训练选择 `--experiment innovate-1`。默认训练模式仍为 baseline，便于已有脚本与检查点继续使用。

## 已实现

- `dehazemamba/wavelet.py`：按 P3 的 LL/LH/HL/HH 顺序进行固定 Haar 变换；低频残差、局部与全局低频门控、高频带符号残差、逆变换与联合细化。末层零初始化。
- `dehazemamba/model.py`：在最终 decoder 与 head 之间接入模块；输入原有 padding 和最终裁剪保持一致。
- `dehazemamba/losses.py`：保留原 `dehaze_loss`，新增 `InnovationLoss`。目标为 L1 + 1.0 MSE + 0.1 W-SSIM，2 层、r=0.4，使用 P14 固定尺度变换。SSIM 强制 float32，使用有效高斯窗口，剔除支持区域包含补齐像素的小波系数。
- `train_dehaze.py`：记录 l1/mse/w_ssim 分量、模型配置与损失配置；恢复时核对架构与损失；新实验默认独立输出目录，已有检查点不会被普通新训练覆盖。
- `evaluate_dehaze.py`：已有按 `model_config` 构建模型的入口可直接识别新检查点；评价指标与裁剪规则不变。

`architecture_version=2` 继续表示原主干实现版本，新增架构由检查点中显式保存的 `wavelet_refinement` 区分；旧检查点缺少该字段时默认为 False。模型默认初始化时，新增模块在 float32 浮点容差内为恒等映射；混合精度下仍存在舍入误差。

## 在 GPU 训练环境运行

以下命令仅为使用说明，本次没有启动正式训练。需先具备原 baseline 的 PyTorch/CUDA/Mamba 扩展环境及已核验配对的数据。新增模块不增加第三方运行依赖。

```bash
cd /path/to/dehazeMambaTry
python train_dehaze.py --experiment innovate-1 \
  --data-root /path/to/prepared_dataset \
  --output-dir myexperiment/results/innovate_1
```

项目内已经准备好 `myexperiment/data/satehaze1k_mrshaze` 时，可使用：

```bash
bash myexperiment/script/run_innovate_1.sh
```

脚本在前台运行，使用现有规范化数据，不重新运行配对/重采样。可用 `DEHAZEMAMBA_PYTHON`、`INNOVATE_DATA_ROOT`、`INNOVATE_OUTPUT_DIR` 指定环境和路径；额外 CLI 参数会继续传入训练入口。

从训练好的同规格 v2 baseline 初始化：

```bash
python train_dehaze.py --experiment innovate-1 \
  --data-root /path/to/prepared_dataset \
  --init-checkpoint /path/to/baseline/last.pt \
  --output-dir myexperiment/results/innovate_1_from_baseline
```

该方式保留 baseline 参数，新增分支采用零残差初始化，优化器重新创建；不能与 `--resume` 同用。公平比较从头训练的 baseline 与改进模型时，应使用相同初始化/预训练条件。

恢复训练时仍指定 `--experiment innovate-1`，并使用原训练的模型规格、`--mse-weight`、`--wavelet-weight` 和训练计划：

```bash
python train_dehaze.py --experiment innovate-1 \
  --data-root /path/to/prepared_dataset \
  --resume myexperiment/results/innovate_1/last.pt \
  --output-dir myexperiment/results/innovate_1
```

评价时明确指定新检查点和独立目录：

```bash
python evaluate_dehaze.py \
  --data-root /path/to/prepared_dataset \
  --checkpoint myexperiment/results/innovate_1/last.pt \
  --output-dir myexperiment/results/innovate_1/evaluation
```

## 验证边界

`tests/test_innovate_1.py` 检查 Haar 基底、能量与逆变换、自动微分、零初始化恒等性及门控学习信号、常量图像损失闭式值、奇数尺寸、越界预测梯度与 float32 SSIM、模型尺寸与配置恢复。模型布线检查替代了 Mamba 的 scan 运算，以便在无 CUDA 扩展的 CPU 主机运行；它不证明 CUDA scan 集成已经通过。

```bash
python -m pytest -q tests/test_innovate_1.py
```

正式训练、真实数据配准审计、GPU 混合精度端到端运行以及设计稿中的完整消融仍需后续执行。设计稿保留为理论方案，不将本次代码验证写成性能提升结果。

本次本机验证结果：临时 CPU 环境 PyTorch 2.15.0，新增 7 项测试以及原有频域损失、残差缩放、SK 融合、SSIM/旧检查点检查共 11 项通过。原测试中依赖真实 Mamba scan/CUDA 的 4 项未执行。命令为：

```bash
python -m pytest -q tests/test_innovate_1.py tests/test_dehazemamba.py \
  -k 'InnovationTests or frequency_complex or residual_scale or sk_weights or ssim_and_legacy'
```

Python 语法、启动脚本 Shell 语法和 Git 补丁格式检查通过。测试使用临时依赖环境，未修改项目依赖清单；CUDA 支持仍沿用原仓库安装要求。
