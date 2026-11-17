"""Fixed orthonormal Haar transform and innovate_1 output refinement (P3–P9)."""
import torch
from torch import Tensor, nn
import torch.nn.functional as F


def haar_dwt(x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if x.ndim != 4 or x.shape[-2] % 2 or x.shape[-1] % 2:
        raise ValueError("Haar DWT requires BCHW with even spatial dimensions")
    a, b = x[..., 0::2, 0::2], x[..., 0::2, 1::2]
    c, d = x[..., 1::2, 0::2], x[..., 1::2, 1::2]
    return (a+b+c+d)/2, (a+b-c-d)/2, (a-b+c-d)/2, (a-b-c+d)/2


def haar_idwt(ll: Tensor, lh: Tensor, hl: Tensor, hh: Tensor) -> Tensor:
    if ll.ndim != 4 or any(x.shape != ll.shape for x in (lh, hl, hh)):
        raise ValueError("Haar IDWT requires four equally shaped BCHW subbands")
    pixels = torch.stack(((ll+lh+hl+hh)/2, (ll+lh-hl-hh)/2,
                          (ll-lh+hl-hh)/2, (ll-lh-hl+hh)/2), dim=2)
    return F.pixel_shuffle(pixels.flatten(1, 2), 2)


class WaveletRefinement(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        c = channels
        self.low = nn.Sequential(nn.Conv2d(c, c, 1), nn.SiLU(),
                                 nn.Conv2d(c, c, 7, padding=3, groups=c), nn.SiLU(),
                                 nn.Conv2d(c, c, 1))
        self.global_context = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c, 1))
        self.local_context = nn.Conv2d(c, c, 3, padding=1, groups=c)
        self.gate = nn.Conv2d(c, 3*c, 1)
        self.high = nn.Sequential(nn.Conv2d(3*c, 3*c, 3, padding=1, groups=3*c),
                                  nn.SiLU(), nn.Conv2d(3*c, 3*c, 1))
        self.joint = nn.Sequential(nn.Conv2d(c, c, 3, padding=1, groups=c),
                                   nn.SiLU(), nn.Conv2d(c, c, 1))
        for layer in (self.low[-1], self.high[-1], self.joint[-1], self.gate):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: Tensor) -> Tensor:
        low, *bands = haar_dwt(x)
        high = torch.cat(bands, dim=1)
        low = low + self.low(low)
        gate = torch.sigmoid(self.gate(F.silu(self.local_context(low) + self.global_context(low))))
        high = high + gate * self.high(high)
        restored = haar_idwt(low, *high.chunk(3, dim=1))
        return restored + self.joint(restored)
