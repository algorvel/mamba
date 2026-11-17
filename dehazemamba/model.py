"""DehazeMamba: Fig. 2/3 and Eqs. (1)-(7) of Zhao et al., JSTARS 2026.

Widths, SSM state size, expansion and stage-depth allocation are exposed
reproduction choices; the paper does not provide an executable architecture.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref

ARCHITECTURE_VERSION = 2
MODEL_DEPTHS = {"T": (2, 2, 2, 1, 1), "B": (4, 4, 4, 2, 2), "L": (8, 8, 8, 4, 4)}


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


def scan_sequences(x: Tensor) -> tuple[Tensor, ...]:
    """Four full-image traversals, each of length H*W (not separate rows)."""
    row = x.flatten(2)
    col = x.transpose(2, 3).flatten(2)
    return row, col, row.flip(-1), col.flip(-1)


def merge_sequences(ys: Sequence[Tensor], height: int, width: int) -> Tensor:
    row = ys[0] + ys[2].flip(-1)
    col = ys[1] + ys[3].flip(-1)
    col = col.reshape(*col.shape[:2], width, height).transpose(2, 3)
    return (row.reshape(*row.shape[:2], height, width) + col) / 4.0


class ScanParameters(nn.Module):
    """Input-dependent delta/B/C and stable learned A/D, as in Mamba S6.

    A/D parameterization and delta initialization follow the bundled Mamba-1;
    the article's algorithm abbreviates parameter generation as Linear(x).
    """
    def __init__(self, channels: int, state_dim: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.rank = math.ceil(channels / 16)
        self.x_proj = nn.Conv1d(channels, self.rank + 2 * state_dim, 1, bias=False)
        self.dt_proj = nn.Conv1d(self.rank, channels, 1)
        nn.init.uniform_(self.dt_proj.weight, -self.rank ** -0.5, self.rank ** -0.5)
        dt = torch.exp(torch.rand(channels) * math.log(100) + math.log(0.001))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(torch.arange(1, state_dim + 1).float().log().repeat(channels, 1))
        self.D = nn.Parameter(torch.ones(channels))

    def project(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        dt, b, c = torch.split(self.x_proj(x), (self.rank, self.state_dim, self.state_dim), dim=1)
        # Bias is applied in float32 inside the scan, before softplus.
        dt = F.conv1d(dt, self.dt_proj.weight)
        return dt, b.contiguous(), c.contiguous()

    def decode(self, x: Tensor, dt: Tensor, b: Tensor, c: Tensor) -> Tensor:
        # CPU reference is intended for small correctness checks only.
        scan = selective_scan_fn if x.is_cuda else selective_scan_ref
        return scan(
            x, dt, -self.A_log.float().exp(), b, c, self.D.float(),
            delta_bias=self.dt_proj.bias.float(), delta_softplus=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        dt, b, c = self.project(x)
        return self.decode(x, dt, b, c)


def recompute(module, *inputs: Tensor) -> Tensor:
    if torch.is_grad_enabled() and any(x.requires_grad for x in inputs):
        return checkpoint(module, *inputs, use_reentrant=False)
    return module(*inputs)


class SS2D(nn.Module):
    def __init__(self, channels: int, state_dim: int = 8) -> None:
        super().__init__()
        self.scans = nn.ModuleList(ScanParameters(channels, state_dim) for _ in range(4))

    def forward(self, x: Tensor) -> Tensor:
        ys = [recompute(scan, seq) for scan, seq in zip(self.scans, scan_sequences(x))]
        return merge_sequences(ys, *x.shape[-2:])


class CSS2D(nn.Module):
    """Cross-modal decoding with C_diff=abs(C_rgb-C_sar), Eqs. (5)-(7)."""
    def __init__(self, channels: int, state_dim: int = 8) -> None:
        super().__init__()
        self.rgb_scans = nn.ModuleList(ScanParameters(channels, state_dim) for _ in range(4))
        self.sar_scans = nn.ModuleList(ScanParameters(channels, state_dim) for _ in range(4))

    @staticmethod
    def difference(rgb_scan, sar_scan, rgb: Tensor, sar: Tensor) -> Tensor:
        dt_rgb, b_rgb, c_rgb = rgb_scan.project(rgb)
        dt_sar, b_sar, c_sar = sar_scan.project(sar)
        c_difference = (c_rgb - c_sar).abs()
        y_rgb = rgb_scan.decode(rgb, dt_rgb, b_rgb, c_difference)
        y_sar = sar_scan.decode(sar, dt_sar, b_sar, c_difference)
        return (y_rgb - y_sar).abs()

    def forward(self, rgb: Tensor, sar: Tensor) -> Tensor:
        ys = []
        for rgb_scan, sar_scan, r, s in zip(
            self.rgb_scans, self.sar_scans, scan_sequences(rgb), scan_sequences(sar)
        ):
            # Bind modules now: checkpoint recomputation must use the same direction.
            def run(a, b, rs=rgb_scan, ss=sar_scan):
                return self.difference(rs, ss, a, b)
            ys.append(recompute(run, r, s))
        return merge_sequences(ys, *rgb.shape[-2:])


class VSS2D(nn.Module):
    """Fig. 2(c): Linear/DConv/SiLU/SS2D/LN, SiLU gate, Linear."""
    def __init__(self, channels: int, state_dim: int = 8, expand: int = 1) -> None:
        super().__init__()
        inner = channels * expand
        self.in_proj = nn.Conv2d(channels, 2 * inner, 1)
        self.dconv = nn.Conv2d(inner, inner, 3, padding=1, groups=inner)
        self.scan = SS2D(inner, state_dim)
        self.norm = LayerNorm2d(inner)
        self.out_proj = nn.Conv2d(inner, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        feature, gate = self.in_proj(x).chunk(2, dim=1)
        feature = self.norm(self.scan(F.silu(self.dconv(feature))))
        return self.out_proj(feature * F.silu(gate))


class DMBlock(nn.Module):
    """Fig. 3 scales the skip paths, rather than the VSS/MLP outputs."""
    def __init__(self, channels: int, state_dim: int = 8, expand: int = 1) -> None:
        super().__init__()
        self.norm1 = LayerNorm2d(channels)
        self.vss = VSS2D(channels, state_dim, expand)
        self.scale1 = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.norm2 = LayerNorm2d(channels)
        hidden = channels * 2
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden), nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
        )
        self.scale2 = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        x = self.scale1 * x + self.vss(self.norm1(x))
        return self.scale2 * x + self.mlp(self.norm2(x))


class HPDM(nn.Module):
    def __init__(self, channels: int, state_dim: int = 8) -> None:
        super().__init__()
        def extract():
            return nn.Sequential(nn.Conv2d(channels, channels, 1),
                                 nn.Conv2d(channels, channels, 3, padding=1, groups=channels), nn.SiLU())
        self.extract_rgb = extract()
        self.extract_sar = extract()
        self.css = CSS2D(channels, state_dim)
        self.norm = LayerNorm2d(channels)
        self.gate_rgb = nn.Conv2d(channels, channels, 1)
        self.gate_sar = nn.Conv2d(channels, channels, 1)
        self.weight = nn.Conv2d(channels, channels, 1)

    def forward(self, rgb: Tensor, sar: Tensor) -> tuple[Tensor, Tensor]:
        difference = self.norm(self.css(self.extract_rgb(rgb), self.extract_sar(sar)))
        gate = F.silu(self.gate_rgb(rgb) + self.gate_sar(sar))
        weight = torch.sigmoid(self.weight(gate * difference))
        return (1 - weight) * rgb, weight * sar


class PFM(nn.Module):
    def __init__(self, channels: int, state_dim: int = 8) -> None:
        super().__init__()
        self.coarse = nn.Conv2d(channels * 2, channels, 3, padding=1)
        self.vss = VSS2D(channels * 3, state_dim)
        self.output = nn.Conv2d(channels * 3, channels, 3, padding=1)

    def forward(self, rgb: Tensor, sar: Tensor) -> Tensor:
        coarse = self.coarse(torch.cat((rgb, sar), dim=1))
        weight = torch.sigmoid(coarse)
        refined = torch.cat((weight * rgb, (1 - weight) * sar, coarse), dim=1)
        return self.output(self.vss(refined))


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.op = nn.Sequential(nn.PixelUnshuffle(2), nn.Conv2d(in_channels * 4, out_channels, 1))

    def forward(self, x: Tensor) -> Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.op = nn.Sequential(nn.Conv2d(in_channels, out_channels * 4, 1), nn.PixelShuffle(2))

    def forward(self, x: Tensor) -> Tensor:
        return self.op(x)


class SKFusion(nn.Module):
    """Selective-kernel branch attention: pooled joint context, channel weights."""
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(channels // 8, 4)
        self.weights = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1),
                                     nn.ReLU(), nn.Conv2d(hidden, channels * 2, 1))

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        b, c = x.shape[:2]
        weights = self.weights(x + skip).reshape(b, 2, c, 1, 1).softmax(dim=1)
        return weights[:, 0] * x + weights[:, 1] * skip


class DehazeMamba(nn.Module):
    """Two D-layers, two same-resolution fusion rounds, two U-layers (Fig. 2).

    N1/N2: encoder; N3: each bottleneck fusion round; N4/N5: decoder
    coarse-to-fine. This allocation is an explicit assumption (see audit).
    """
    def __init__(self, base_channels: int = 24, state_dim: int = 8,
                 depths: Sequence[int] = MODEL_DEPTHS["T"],
                 architecture_version: int = ARCHITECTURE_VERSION) -> None:
        super().__init__()
        if architecture_version != ARCHITECTURE_VERSION:
            raise ValueError("Incompatible architecture; corrected model requires retraining")
        if len(depths) != 5 or any(int(d) != d or d < 1 for d in depths):
            raise ValueError("depths must contain five positive integer stage counts")
        if base_channels < 1 or state_dim < 1:
            raise ValueError("base_channels and state_dim must be positive")
        self.base_channels, self.state_dim = base_channels, state_dim
        self.depths = tuple(int(d) for d in depths)
        self.widths = (base_channels, base_channels * 2, base_channels * 4)
        def blocks(c, n):
            return nn.Sequential(*(DMBlock(c, state_dim) for _ in range(n)))
        self.rgb_stem = nn.Conv2d(3, self.widths[0], 3, padding=1)
        self.sar_stem = nn.Conv2d(1, self.widths[0], 3, padding=1)
        self.rgb_stages = nn.ModuleList(blocks(self.widths[i], self.depths[i]) for i in range(2))
        self.sar_stages = nn.ModuleList(blocks(self.widths[i], self.depths[i]) for i in range(2))
        self.rgb_down = nn.ModuleList(Downsample(self.widths[i], self.widths[i+1]) for i in range(2))
        self.sar_down = nn.ModuleList(Downsample(self.widths[i], self.widths[i+1]) for i in range(2))
        c = self.widths[-1]
        self.fusion_rgb = nn.ModuleList(blocks(c, self.depths[2]) for _ in range(2))
        self.fusion_sar = nn.ModuleList(blocks(c, self.depths[2]) for _ in range(2))
        self.hpdm = nn.ModuleList(HPDM(c, state_dim) for _ in range(2))
        self.pfm = nn.ModuleList(PFM(c, state_dim) for _ in range(2))
        self.upsample = nn.ModuleList(Upsample(self.widths[i+1], self.widths[i]) for i in range(2))
        self.skip_fusion = nn.ModuleList(SKFusion(self.widths[i]) for i in range(2))
        self.decoder = nn.ModuleList(blocks(self.widths[i], self.depths[4-i]) for i in range(2))
        self.head = nn.Conv2d(self.widths[0], 3, 3, padding=1)

    def forward(self, hazy_rgb: Tensor, sar: Tensor) -> Tensor:
        if hazy_rgb.ndim != 4 or hazy_rgb.shape[1] != 3:
            raise ValueError(f"Expected RGB input [B,3,H,W], got {tuple(hazy_rgb.shape)}")
        if sar.ndim != 4 or sar.shape[1] != 1:
            raise ValueError(f"Expected SAR input [B,1,H,W], got {tuple(sar.shape)}")
        if hazy_rgb.shape[0] != sar.shape[0] or hazy_rgb.shape[-2:] != sar.shape[-2:]:
            raise ValueError("RGB and SAR inputs must have matching batch and spatial sizes")
        height, width = hazy_rgb.shape[-2:]
        pad = (0, (-width) % 4, 0, (-height) % 4)
        rgb = self.rgb_stem(F.pad(hazy_rgb, pad, mode="replicate"))
        sar = self.sar_stem(F.pad(sar, pad, mode="replicate"))
        skips = []
        for i in range(2):
            rgb, sar = self.rgb_stages[i](rgb), self.sar_stages[i](sar)
            skips.append(rgb)
            rgb, sar = self.rgb_down[i](rgb), self.sar_down[i](sar)
        for i in range(2):
            rgb, sar = self.fusion_rgb[i](rgb), self.fusion_sar[i](sar)
            r, s = self.hpdm[i](rgb, sar)
            rgb = rgb + self.pfm[i](r, s)
        for i in (1, 0):
            rgb = self.decoder[i](self.skip_fusion[i](self.upsample[i](rgb), skips[i]))
        return self.head(rgb)[..., :height, :width]


def require_current_checkpoint(payload: dict) -> None:
    if payload.get("model_config", {}).get("architecture_version") != ARCHITECTURE_VERSION:
        raise ValueError("Checkpoint predates the corrected DehazeMamba architecture/loss. "
                         "Retrain into a new output directory; legacy weights are incompatible.")
