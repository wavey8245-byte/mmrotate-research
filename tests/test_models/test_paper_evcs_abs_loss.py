"""Analytic correction, boundedness, gradients and real FCOS integration."""
import logging
import math
import unittest

import mmcv
import torch
from mmrotate.models.builder import build_head
from mmrotate.models.losses.paper_evcs_abs_loss import (
    EVCSAbsCorrectionLoss, evcs_abs_values)
from mmrotate.models.losses.paper_evcs_loss import PaperKFIoUDecodedLoss


class TestEVCSAbsCorrection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        from mmcv.utils import get_logger
        get_logger('mmcv', log_level=logging.ERROR)

    def test_length_error_penalizes_both_directions(self):
        t = torch.tensor([[0., 0., 10., 6., 0.]], dtype=torch.float64)
        for offset in (-1., 1., 10., 1000.):
            p = t.clone()
            p[:, 2:4] += offset
            p.requires_grad_()
            term, _, _ = evcs_abs_values(p, t)
            self.assertAlmostEqual(float(term), 2 - 2 * math.exp(-abs(offset)))
            term.sum().backward()
            # Gradient descent reduces overlarge sides and increases undersized sides.
            self.assertGreaterEqual(float(p.grad[0, 2]) * offset, 0.)
        self.assertAlmostEqual(float(evcs_abs_values(t, t)[0]), 0.)

    def test_combination_and_gradcheck(self):
        p = torch.tensor([[.3, .2, 8., 5., .23]], dtype=torch.float64, requires_grad=True)
        t = torch.tensor([[0., 0., 9., 6., -.17]], dtype=torch.float64)
        expected = PaperKFIoUDecodedLoss()(p, t) + .006 * evcs_abs_values(p, t)[0].sum()
        torch.testing.assert_close(EVCSAbsCorrectionLoss()(p, t), expected)
        self.assertTrue(torch.autograd.gradcheck(lambda x: EVCSAbsCorrectionLoss()(x, t), (p,)))

    def test_bounded_values_and_equivalent_geometry(self):
        torch.manual_seed(42)
        p, t = torch.randn(300, 5, dtype=torch.float64), torch.randn(300, 5, dtype=torch.float64)
        p[:, 2:4] = p[:, 2:4].exp() * 10
        t[:, 2:4] = t[:, 2:4].exp() * 10
        value, _, _ = evcs_abs_values(p, t)
        self.assertTrue((value >= -1e-12).all() and (value <= 6 + 1e-12).all())
        q = p.clone()
        q[:, 2], q[:, 3] = p[:, 3], p[:, 2]
        q[:, 4] += math.pi / 2
        torch.testing.assert_close(evcs_abs_values(p, t)[0], evcs_abs_values(q, t)[0], atol=1e-11, rtol=1e-11)

    def test_reduction_and_collapsed_boxes(self):
        t = torch.tensor([[0., 0., 9., 6., .2], [0., 0., 8., 5., -.3]], dtype=torch.float64)
        p = t.clone()
        p[0, 2:4] = 0
        p.requires_grad_()
        w = torch.tensor([1., .25], dtype=torch.float64)
        loss = EVCSAbsCorrectionLoss()
        raw = loss(p, t, reduction_override='none')
        weighted = loss(p, t, weight=w, avg_factor=w.sum())
        torch.testing.assert_close(weighted, (raw * w).sum() / w.sum())
        weighted.backward()
        self.assertTrue(torch.isfinite(p.grad).all())
        self.assertEqual(float(loss(p[:0], t[:0])), 0.)
        self.assertEqual(float(loss(p, t, weight=w * 0, avg_factor=1e-6)), 0.)

    def test_actual_fcos_cpu_and_cuda(self):
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            torch.manual_seed(42)
            cfg = mmcv.Config.fromfile('configs/dior_r/server_trainval/paper_evcs_abs_correction.py')
            head = build_head(cfg.model.bbox_head).to(device)
            head.init_weights()
            features = [torch.randn(1, 256, k, k, device=device) for k in (16, 8, 4, 2, 1)]
            boxes = [torch.tensor([[64., 64., 48., 24., .2]], device=device)]
            labels = [torch.tensor([1], device=device)]
            meta = [dict(img_shape=(128, 128, 3), pad_shape=(128, 128, 3), scale_factor=1.)]
            losses = head.loss(*head(features), boxes, labels, meta)
            total = sum(v for k, v in losses.items() if 'loss' in k)
            total.backward()
            self.assertTrue(torch.isfinite(total))
            for param in head.parameters():
                if param.grad is not None:
                    self.assertTrue(torch.isfinite(param.grad).all())
            self.assertGreater(float(head.conv_angle.weight.grad.abs().sum()), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
