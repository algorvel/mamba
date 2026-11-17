"""CPU checks for innovate_1 math, gradients, and model/checkpoint wiring."""
import unittest
from unittest.mock import patch

import torch

from dehazemamba.wavelet import haar_dwt, haar_idwt, WaveletRefinement
from dehazemamba.losses import InnovationLoss
from dehazemamba.model import DehazeMamba, ScanParameters


class InnovationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_haar_basis_energy_and_gradients(self):
        impulse = torch.tensor([[[[1., 0.], [0., 0.]]]])
        self.assertTrue(all(torch.equal(b, torch.full_like(b, .5)) for b in haar_dwt(impulse)))
        x = torch.randn(2, 3, 8, 10, dtype=torch.double, requires_grad=True)
        bands = haar_dwt(x)
        torch.testing.assert_close(haar_idwt(*bands), x)
        torch.testing.assert_close(sum(b.square().sum() for b in bands), x.square().sum())
        self.assertTrue(torch.autograd.gradcheck(lambda z: haar_idwt(*haar_dwt(z)), (x,)))
        with self.assertRaises(ValueError):
            haar_dwt(x[..., :7, :])

    def test_identity_and_learnable_gate(self):
        module = WaveletRefinement(3)
        x = torch.randn(2, 3, 8, 10, requires_grad=True)
        torch.testing.assert_close(module(x), x)
        module(x).square().mean().backward()
        self.assertGreater(module.high[-1].weight.grad.abs().sum().item(), 0)
        # Gate gradients activate once the high-frequency residual leaves zero.
        torch.nn.init.normal_(module.high[-1].weight, std=.1)
        module.zero_grad()
        module(x).square().mean().backward()
        self.assertGreater(module.gate.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters()))

    def test_constant_loss_matches_closed_form(self):
        criterion = InnovationLoss()
        x = torch.full((1, 3, 45, 47), .2, requires_grad=True)
        y = torch.full_like(x, .3)
        total, parts = criterion(x, y)
        raw_error = 1 - (2*.2*.3+.01**2)/(.2**2+.3**2+.01**2)
        expected_ws = .5 * (1 + .4**4) * raw_error
        self.assertAlmostEqual(parts['w_ssim'].item(), expected_ws, places=4)
        self.assertAlmostEqual(total.item(), .1 + .01 + .1*expected_ws, places=4)
        total.backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_identity_out_of_range_and_amp(self):
        criterion = InnovationLoss()
        y = torch.rand(1, 3, 45, 47)
        total, parts = criterion(y, y)
        self.assertAlmostEqual(total.item(), 0., places=6)
        x = (torch.randn_like(y) * 2).requires_grad_()
        with torch.autocast('cpu', dtype=torch.bfloat16):
            total, parts = criterion(x, y)
        self.assertEqual(total.dtype, torch.float32)
        total.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(x.grad[x.detach() > 1].abs().sum().item(), 0)
        with self.assertRaises(ValueError):
            criterion(y[..., :43, :], y[..., :43, :])
        with self.assertRaises(ValueError):
            InnovationLoss(float('nan'))

    def test_padded_subband_support_is_excluded(self):
        criterion = InnovationLoss()
        target = torch.rand(1, 3, 45, 47)
        prediction = target.clone()
        prediction[..., -1, :] += 1
        _, parts = criterion(prediction, target)
        # The final odd row affects original-image SSIM but no complete Haar block.
        expected = criterion.structural_error(prediction, target) / 2
        torch.testing.assert_close(parts['w_ssim'], expected)

    def test_training_cli_selects_innovation(self):
        from train_dehaze import parse_args
        with patch('sys.argv', ['train', '--experiment', 'innovate-1', '--mse-weight', '0.5']):
            args = parse_args()
        self.assertEqual(args.experiment, 'innovate-1')
        self.assertEqual(args.mse_weight, .5)
        self.assertEqual(args.wavelet_weight, .1)

    def test_model_wiring_and_checkpoint_roundtrip(self):
        cfg = dict(base_channels=2, state_dim=2, depths=[1]*5, architecture_version=2)
        baseline = DehazeMamba(**cfg)
        cfg['wavelet_refinement'] = True
        model = DehazeMamba(**cfg)
        state = {k: v for k, v in model.state_dict().items() if k.startswith('refinement.')}
        state.update(baseline.state_dict())
        model.load_state_dict(state, strict=True)
        restored = DehazeMamba(**cfg)
        restored.load_state_dict(model.state_dict(), strict=True)
        rgb, sar = torch.randn(1, 3, 5, 7), torch.randn(1, 1, 5, 7)
        # Isolate model wiring from CUDA-only Mamba dependencies on CPU hosts.
        with patch.object(ScanParameters, 'decode', lambda self, x, dt, b, c: x):
            output = model(rgb, sar)
            self.assertEqual(output.shape, rgb.shape)
            torch.testing.assert_close(output, baseline(rgb, sar))
            torch.testing.assert_close(output, restored(rgb, sar))
            output.square().mean().backward()
            self.assertGreater(model.refinement.high[-1].weight.grad.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
