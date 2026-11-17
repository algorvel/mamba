"""Spatial and frequency-domain losses from the DehazeMamba paper."""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def dehaze_loss(prediction: Tensor, target: Tensor, frequency_weight: float = 0.1):
    spatial = F.l1_loss(prediction, target)
    # Eq. (9): complex L1 over the full spectrum. Averaging separate real/imag
    # components or an unweighted half-spectrum changes this objective.
    pred_fft = torch.fft.fft2(prediction.float())
    target_fft = torch.fft.fft2(target.float())
    frequency = (pred_fft - target_fft).abs().mean()
    total = spatial + frequency_weight * frequency
    return total, spatial.detach(), frequency.detach()


class InnovationLoss(torch.nn.Module):
    """P11–P16: L1 + RGB MSE + two-level, scale-normalized Haar SSIM."""
    def __init__(self, mse_weight: float = 1.0, wavelet_weight: float = 0.1):
        super().__init__()
        import math
        if any(not math.isfinite(w) or w < 0 for w in (mse_weight, wavelet_weight)):
            raise ValueError("Loss weights must be finite and non-negative")
        self.mse_weight, self.wavelet_weight = mse_weight, wavelet_weight
        coords = torch.arange(11, dtype=torch.float32) - 5
        gaussian = torch.exp(-coords.square() / (2 * 1.5**2))
        gaussian /= gaussian.sum()
        self.register_buffer("window", (gaussian[:, None] * gaussian[None, :])[None, None])

    def structural_error(self, x: Tensor, y: Tensor) -> Tensor:
        window = self.window.expand(x.shape[1], 1, 11, 11)
        def average(z):
            return F.conv2d(z, window, groups=x.shape[1])
        mx, my = average(x), average(y)
        vx = (average(x.square()) - mx.square()).clamp_min(0)
        vy = (average(y.square()) - my.square()).clamp_min(0)
        covariance = average(x * y) - mx * my
        score = ((2 * mx * my + 0.01**2) * (2 * covariance + 0.03**2)) / (
            (mx.square() + my.square() + 0.01**2) * (vx + vy + 0.03**2))
        return 1 - score.mean()

    def forward(self, prediction: Tensor, target: Tensor):
        from .wavelet import haar_dwt
        if prediction.ndim != 4 or prediction.shape != target.shape or prediction.shape[1] != 3:
            raise ValueError("Innovation loss requires equally shaped RGB BCHW tensors")
        height, width = prediction.shape[-2:]
        if min(height, width) < 44:
            raise ValueError("Two-level valid-window W-SSIM requires images at least 44x44")
        # Explicitly disable AMP: casting alone does not protect SSIM convolutions.
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            x, y = prediction.float(), target.float()
            l1, mse = F.l1_loss(x, y), F.mse_loss(x, y)
            structural = self.structural_error(x, y)
            pad = (0, (-width) % 4, 0, (-height) % 4)
            x, y = F.pad(x, pad, mode="replicate"), F.pad(y, pad, mode="replicate")
            r = 0.4
            for level in (1, 2):
                x, *xh = haar_dwt(x)
                y, *yh = haar_dwt(y)
                scale = 2**level
                h, w = height // scale, width // scale
                # Exclude coefficients whose support touches synthetic padding.
                for a, b, weight in zip(xh, yh, (r*(1-r), r*(1-r), (1-r)**2)):
                    structural = structural + r**(2*(level-1)) * weight * self.structural_error(
                        a[..., :h, :w] / scale + 0.5, b[..., :h, :w] / scale + 0.5)
            structural = (structural + r**4 * self.structural_error(
                x[..., :h, :w] / scale, y[..., :h, :w] / scale)) / 2
            total = l1 + self.mse_weight * mse + self.wavelet_weight * structural
        return total, {"l1": l1.detach(), "mse": mse.detach(), "w_ssim": structural.detach()}
