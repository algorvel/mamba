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
