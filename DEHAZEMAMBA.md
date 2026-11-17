# DehazeMamba reproduction

This is a **paper-guided implementation**, not the authors' released model or a
verified numerical reproduction. The source is Zhao et al., *DehazeMamba:
SAR-Guided Optical Remote Sensing Image Dehazing With Adaptive State Space
Model*, IEEE JSTARS 19 (2026), DOI 10.1109/JSTARS.2025.3634360. See
[the implementation audit](REPRODUCTION_AUDIT.md) for corrections, assumptions,
and remaining experimental gaps.

## Implemented behavior

- Fig. 2: two dual-branch downsampling layers, two fusion rounds at the same
  bottleneck resolution, then two upsampling layers with SK skip fusion.
- HPDM: modality-specific states are decoded using the shared difference matrix
  `abs(C_rgb - C_sar)`, followed by output difference, LN, joint SiLU gate, and
  complementary optical/SAR weights (Eqs. 1–7 and Fig. 2b).
- VSS: 2D depthwise convolution, full-image selective scans in four directions,
  normalization, SiLU gate, and output projection (Fig. 2c).
- DMBlock: learnable scales on the residual paths (Fig. 3).
- PFM: coarse fusion and complementary weighting, then concatenation of all
  three features through VSS and convolution (Algorithm 2).
- T/B/L depth presets `[2,2,2,1,1]`, `[4,4,4,2,2]`, `[8,8,8,4,4]`.
- AdamW, learning rate `2e-4`, cosine decay to `1e-6`, batch size 6, 150 epochs.
- Spatial L1 plus `0.1 * mean(abs(fft2(pred) - fft2(target)))` (Eqs. 8–9).
- Per-image RGB PSNR, valid-window Gaussian SSIM and LPIPS, plus dataset means.

## Reproduction choices

The paper does not supply channel widths, complete VSS/MLP/SK internals,
SSM initialization, FFT normalization, AdamW weight decay or LPIPS backbone.
Defaults remain base width 24, state size 8; widths at the three resolutions
are 24/48/96. VSS expansion is 1 and local MLP expansion is 2. Directional SSMs
have separate parameters and their aligned outputs are averaged. A/D and delta
initialization follow the bundled Mamba-1 implementation. We use an unnormalized
forward FFT, mean reduction, AdamW default weight decay, and LPIPS AlexNet v0.1.
SSIM averages RGB channels with valid 11x11 windows and sigma 1.5. Predictions
are clamped to [0,1] before metrics; PNG rounding is used only for saved images.
These metric conventions are explicit choices, not confirmed author settings.

The five depth counts are allocated to encoder stages N1/N2, each fusion-round
branch at the bottleneck N3, and coarse/fine decoder stages N4/N5. Fig. 2 shows
the two fusion rounds but does not label this depth allocation, so it remains
an assumption. The resulting default T model has **2,535,637 parameters**, versus
**6.65M** in paper Table II. Exact topology/width/state settings must be obtained
from the authors to resolve that gap; changing widths just to match a total
would not establish correctness.

CSS2D follows the per-position input features in Eqs. (2)–(7). Algorithm 1's
line 16 also writes a next-input assignment from the previous output, which is
ambiguous alongside those equations and normal selective-scan operation. No
additional autoregressive input-feedback loop is assumed here.

## Environment and commands

Run from this checkout's root:

```bash
conda env create -f environment.yml
conda activate mamba
python tools/build_selective_scan.py
python train_dehaze.py --data-root /path/to/MRSHaze --model-size T
python evaluate_dehaze.py --data-root /path/to/MRSHaze \
  --checkpoint outputs/dehazemamba-v2/last.pt
python -m unittest discover -s tests -p test_dehazemamba.py -v
```

For an existing environment, install the added metric dependency with
`python -m pip install lpips==0.1.4`. LPIPS may download pretrained backbone
weights on first use. `--no-lpips` is available for quick evaluation checks;
that mode does not report the full metric set from the paper.

The dataset layout is `MRSHaze/{train,test}/{GT,hazy,SAR}/*.png`, matched by
filename. The released processed PNGs are consumed directly; do not apply BM3D
or synthetic haze generation to them again. The local split contains 6000
training and 2000 test pairs; sampled modalities are RGB/RGB/L at 512x512.

Checkpoints include the architecture version, model settings, optimizer,
learning-rate schedule and AMP scaler. The corrected model uses
`outputs/dehazemamba-v2` by default. Old checkpoints/results are preserved but
cannot be loaded into this architecture; **retraining is required**. Existing
v1 metrics and corrected metrics also use different loss/evaluation conventions.
`--max-steps` is a debugging option and does not produce a fully trained model.

On the local host, CUDA 12.0 nvcc differs from the PyTorch CUDA 11.8 build.
The existing in-place extension was reused for verification. On a new host,
prefer a matching toolkit. The helper's `--allow-major-mismatch` explicitly
bypasses PyTorch's build guard if required by the local environment.
