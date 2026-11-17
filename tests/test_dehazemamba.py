"""Regression checks against paper equations and diagrams (no dataset needed)."""
import unittest
from unittest.mock import patch

import torch
from torch import nn

from dehazemamba.losses import dehaze_loss
from dehazemamba.model import (
    CSS2D, DMBlock, DehazeMamba, MODEL_DEPTHS, ScanParameters, SKFusion,
    merge_sequences, require_current_checkpoint, scan_sequences,
)
from evaluate_dehaze import ssim


def recurrence(module, x, dt, b, c):
    """Independent implementation of Eqs. (2)-(6)."""
    delta = torch.nn.functional.softplus(dt.float() + module.dt_proj.bias.float()[None, :, None])
    a = -module.A_log.float().exp()
    state = torch.zeros(x.shape[0], x.shape[1], module.state_dim, device=x.device)
    outputs = []
    for t in range(x.shape[-1]):
        state = torch.exp(delta[:, :, t, None] * a) * state + (
            delta[:, :, t, None] * b[:, None, :, t] * x[:, :, t, None]
        )
        outputs.append((state * c[:, None, :, t]).sum(-1) + module.D * x[:, :, t])
    return torch.stack(outputs, -1)


class PaperEquations(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)

    def test_full_image_scan_and_inverse(self):
        x = torch.arange(30.).reshape(1, 2, 3, 5)
        seqs = scan_sequences(x)
        self.assertTrue(all(s.shape == (1, 2, 15) for s in seqs))
        torch.testing.assert_close(merge_sequences(seqs, 3, 5), x)
        # A scan must carry context across a row boundary.
        scan = ScanParameters(2, 2)
        x = torch.ones(1, 2, 15, requires_grad=True)
        scan(x)[..., -1].sum().backward()
        self.assertGreater(x.grad[..., 0].abs().sum().item(), 0)

    def test_cross_modal_decoding_equations_and_gradients(self):
        rscan, sscan = ScanParameters(3, 2), ScanParameters(3, 2)
        rgb = torch.randn(2, 3, 5, requires_grad=True)
        sar = torch.randn(2, 3, 5, requires_grad=True)
        dr, br, cr = rscan.project(rgb)
        ds, bs, cs = sscan.project(sar)
        cd = (cr - cs).abs()
        expected = (recurrence(rscan, rgb, dr, br, cd) - recurrence(sscan, sar, ds, bs, cd)).abs()
        actual = CSS2D.difference(rscan, sscan, rgb, sar)
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        self.assertGreater(rscan.x_proj.weight.grad.abs().sum().item(), 0)
        self.assertGreater(sscan.x_proj.weight.grad.abs().sum().item(), 0)
        # Equal C projections cancel state decoding even when hidden states differ.
        rscan.x_proj.weight.data.zero_()
        sscan.x_proj.weight.data.zero_()
        torch.testing.assert_close(CSS2D.difference(rscan, sscan, rgb, sar), (rgb - sar).abs())

    def test_residual_scale_is_on_skip_path(self):
        block = DMBlock(2, 2)
        block.norm1 = block.norm2 = nn.Identity()
        block.vss = block.mlp = nn.Identity()
        block.scale1.data.fill_(2)
        block.scale2.data.fill_(3)
        x = torch.randn(1, 2, 2, 2)
        torch.testing.assert_close(block(x), 12 * x)

    def test_frequency_complex_full_spectrum(self):
        x = torch.zeros(1, 3, 5, 7, requires_grad=True)
        target = torch.zeros_like(x)
        target[:, :, 1, 2] = 1
        total, spatial, frequency = dehaze_loss(x, target)
        # A unit impulse has unit Fourier magnitude at every frequency.
        torch.testing.assert_close(frequency, torch.tensor(1.))
        torch.testing.assert_close(spatial, torch.tensor(1 / 35))
        torch.testing.assert_close(total, torch.tensor(1 / 35 + 0.1))
        total.backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_sk_weights_use_global_context(self):
        fusion = SKFusion(4)
        x, skip = torch.randn(2, 4, 5, 7), torch.randn(2, 4, 5, 7)
        weights = fusion.weights(x + skip).reshape(2, 2, 4, 1, 1).softmax(1)
        torch.testing.assert_close(fusion(x, skip), weights[:, 0] * x + weights[:, 1] * skip)

    def test_five_stage_topology_and_odd_image_backward(self):
        model = DehazeMamba(2, 2, (1, 1, 1, 1, 1))
        rgb, sar = torch.randn(1, 3, 5, 7), torch.randn(1, 1, 5, 7)
        fusion_sizes = []
        hooks = [h.register_forward_pre_hook(lambda m, inputs: fusion_sizes.append(inputs[0].shape[-2:]))
                 for h in model.hpdm]
        output = model(rgb, sar)
        self.assertEqual(output.shape, rgb.shape)
        self.assertEqual(fusion_sizes, [(2, 2), (2, 2)])
        self.assertEqual(len(model.rgb_down), 2)
        output.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        for h in hooks:
            h.remove()
        # Fig. 2 has a final Conv, without an image-level residual bypass.
        nn.init.zeros_(model.head.weight)
        nn.init.zeros_(model.head.bias)
        torch.testing.assert_close(model(rgb, sar), torch.zeros_like(rgb))
        for size, depths in MODEL_DEPTHS.items():
            scaled = DehazeMamba(2, 2, depths)
            self.assertEqual(len(scaled.decoder[1]), depths[3], size)
            self.assertEqual(len(scaled.decoder[0]), depths[4], size)

    def test_ssim_and_legacy_checkpoint(self):
        x = torch.rand(1, 3, 16, 19)
        self.assertAlmostEqual(ssim(x, x), 1.)
        # An 11x11 input has exactly one valid window (no artificial padding).
        with patch('evaluate_dehaze.F.conv2d', wraps=torch.nn.functional.conv2d) as conv:
            ssim(x[..., :11, :11], x[..., :11, :11])
            self.assertTrue(all('padding' not in call.kwargs for call in conv.call_args_list))
        with self.assertRaisesRegex(ValueError, 'Retrain'):
            require_current_checkpoint({'model_config': {'base_channels': 24}})

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_scan_matches_equations(self):
        r, s = ScanParameters(3, 2).cuda(), ScanParameters(3, 2).cuda()
        rgb = torch.randn(1, 3, 17, device='cuda', requires_grad=True)
        sar = torch.randn_like(rgb, requires_grad=True)
        dr, br, cr = r.project(rgb)
        ds, bs, cs = s.project(sar)
        cd = (cr - cs).abs()
        expected = (recurrence(r, rgb, dr, br, cd) - recurrence(s, sar, ds, bs, cd)).abs()
        actual = CSS2D.difference(r, s, rgb, sar)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        actual.sum().backward()
        self.assertTrue(torch.isfinite(rgb.grad).all() and torch.isfinite(sar.grad).all())


if __name__ == '__main__':
    unittest.main()
