"""Independent analytic, numerical-gradient and real FCOS tests for EVCS port."""
import math
import unittest

import mmcv
import torch
from mmrotate.models.builder import build_head
from mmrotate.models.losses.kf_iou_loss import kfiou_loss
from mmrotate.models.losses.paper_evcs_loss import (
    PaperEVCSPublishedLoss, PaperKFIoUDecodedLoss)


class TestPublishedEVCS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_axis_aligned_analytic_value_and_gradient(self):
        target = torch.tensor([[0., 0., 10., 6., 0.]], dtype=torch.float64)
        offset = torch.tensor(.3, dtype=torch.float64, requires_grad=True)
        pred = target + torch.stack([offset * 0, offset * 0, offset,
                                     offset, offset * 0])[None]
        objective = PaperEVCSPublishedLoss()
        value = objective(pred, target)
        base = kfiou_loss(pred, target, pred_decode=pred,
                          targets_decode=target, fun='ln', reduction='none').sum()
        expected = base + .006 * (2 - 2 * offset.exp())
        torch.testing.assert_close(value, expected, atol=1e-12, rtol=1e-12)
        actual_gradient, = torch.autograd.grad(value, offset, retain_graph=True)
        expected_gradient, = torch.autograd.grad(expected, offset)
        torch.testing.assert_close(actual_gradient, expected_gradient)

    def test_signed_exponent_retained_and_overflow_rejected(self):
        target = torch.tensor([[0., 0., 10., 6., 0.]], dtype=torch.float64)
        pred = target.clone()
        pred[:, 2:4] += 10
        self.assertLess(float(PaperEVCSPublishedLoss()(pred, target)), -200)
        pred[:, 2:4] += 1000
        with self.assertRaises(FloatingPointError):
            PaperEVCSPublishedLoss()(pred, target)

    def test_gradcheck_away_from_vertex_switch(self):
        pred = torch.tensor([[.3, .2, 8., 5., .23]], dtype=torch.float64,
                            requires_grad=True)
        target = torch.tensor([[0., 0., 9., 6., -.17]], dtype=torch.float64)
        self.assertTrue(torch.autograd.gradcheck(
            lambda p: PaperEVCSPublishedLoss()(p, target), (pred,)))

    def test_centerness_empty_collapsed_and_zero_weight(self):
        target = torch.tensor([[0., 0., 10., 6., .2],
                               [0., 0., 9., 5., -.3]], dtype=torch.float64)
        pred = target.clone()
        pred[0, 2:4] = 0
        pred[1, 2:4] += .1
        weights = torch.tensor([1., .25], dtype=torch.float64)
        for loss in (PaperKFIoUDecodedLoss(), PaperEVCSPublishedLoss()):
            p = pred.clone().requires_grad_()
            raw = loss(p, target, reduction_override='none')
            reduced = loss(p, target, weight=weights, avg_factor=weights.sum())
            torch.testing.assert_close(reduced, (raw * weights).sum() / weights.sum())
            reduced.backward()
            self.assertTrue(torch.isfinite(p.grad).all())
            empty = p[:0]
            self.assertEqual(float(loss(empty, target[:0])), 0.)
            self.assertEqual(float(loss(p, target, weight=weights * 0, avg_factor=1e-6)), 0.)

    def test_actual_fcos_head_cpu_and_cuda(self):
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            for filename in ('paper_kfiou_decoded.py', 'paper_evcs_published.py'):
                with self.subTest(device=device, config=filename):
                    torch.manual_seed(42)
                    cfg = mmcv.Config.fromfile('configs/dior_r/server_trainval/' + filename)
                    head = build_head(cfg.model.bbox_head).to(device)
                    head.init_weights()
                    features = [torch.randn(1, 256, size, size, device=device)
                                for size in (16, 8, 4, 2, 1)]
                    boxes = [torch.tensor([[64., 64., 48., 24., .2]], device=device)]
                    labels = [torch.tensor([1], device=device)]
                    meta = [dict(img_shape=(128, 128, 3), pad_shape=(128, 128, 3),
                                 scale_factor=1.)]
                    losses = head.loss(*head(features), boxes, labels, meta)
                    value = sum(v for k, v in losses.items() if 'loss' in k)
                    self.assertTrue(torch.isfinite(value))
                    value.backward()
                    for param in head.parameters():
                        if param.grad is not None:
                            self.assertTrue(torch.isfinite(param.grad).all())
                    self.assertGreater(float(head.conv_angle.weight.grad.abs().sum()), 0)
                    empty = head.loss(*head(features), [boxes[0][:0]], [labels[0][:0]], meta)
                    self.assertEqual(float(empty['loss_bbox']), 0.)
                    del head, features, losses, value


if __name__ == '__main__':
    unittest.main(verbosity=2)
