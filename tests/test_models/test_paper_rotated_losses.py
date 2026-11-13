"""Independent matrix references, derivative checks and real FCOS wiring."""
import importlib
import logging
import math
import unittest

import mmcv
import torch

from mmrotate.models.builder import build_head
from mmrotate.models.losses.paper_rotated_losses import (
    PaperBDLoss, PaperDCAGPLLoss, bd_values, dcagpl_values, evcs_literal)


def covariance(box, agbb=False):
    w, h, r = box[2], box[3], box[4]
    if agbb:
        first, second = h * (1 + (4 * r).cos() / 5), w * (1 - (4 * r).cos() / 5)
        r = 4 * r
    else:
        first, second = w, h
    rot = torch.stack([r.cos(), -r.sin(), r.sin(), r.cos()]).reshape(2, 2)
    return rot @ torch.diag(torch.stack([first, second]).square() / 4) @ rot.T


def reference_bd(p, t, agbb=False):
    cp, ct = covariance(p, agbb), covariance(t, agbb)
    avg = (cp + ct) / 2
    d = p[:2] - t[:2]
    db = 3 / 8 * (d @ torch.linalg.solve(avg, d))
    db += .5 * torch.log(torch.det(avg) / torch.sqrt(torch.det(cp) * torch.det(ct)))
    return 1 - 1 / (1 + db.sqrt())


def reference_dcagpl(p, t):
    cp, ct = covariance(p), covariance(t)
    product = cp - cp @ torch.linalg.solve(cp + ct, cp)
    area = lambda c: 4 * torch.linalg.det(c).sqrt()
    k = area(product) / (area(cp) + area(ct) - area(product))
    d = p[:2] - t[:2]
    return torch.log1p(d @ torch.linalg.solve(ct, d)) - torch.log(k + 1e-6)


class TestPaperLosses(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        from mmcv.utils import get_logger
        get_logger('mmcv', log_level=logging.ERROR)

    def boxes(self):
        return (torch.tensor([[1.3, -.7, 8., 5., .37]], dtype=torch.float64),
                torch.tensor([[.2, .4, 9., 6., -.19]], dtype=torch.float64))

    def test_values_and_gradients_match_independent_matrix_references(self):
        for kind in ('bd', 'agbb', 'dcagpl'):
            with self.subTest(kind=kind):
                p, t = self.boxes()
                if kind == 'agbb':
                    p[:, 2:4], t[:, 2:4] = torch.tensor([9., 8.8]), torch.tensor([9., 9.])
                p.requires_grad_()
                if kind == 'dcagpl':
                    value = sum(dcagpl_values(p, t))
                    expected = reference_dcagpl(p[0], t[0])[None]
                else:
                    value = bd_values(p, t, agbb=kind == 'agbb')
                    expected = reference_bd(p[0], t[0], kind == 'agbb')[None]
                torch.testing.assert_close(value, expected, atol=1e-10, rtol=1e-10)
                actual_grad, = torch.autograd.grad(value.sum(), p, retain_graph=True)
                ref_grad, = torch.autograd.grad(expected.sum(), p)
                torch.testing.assert_close(actual_grad, ref_grad, atol=1e-9, rtol=1e-9)

    def test_autograd_finite_difference(self):
        p, t = self.boxes()
        p.requires_grad_()
        for loss in (PaperBDLoss(), PaperBDLoss(agbb=True, square_ratio_threshold=2),
                     PaperDCAGPLLoss()):
            self.assertTrue(torch.autograd.gradcheck(lambda x: loss(x, t), (p,)))

    def test_identity_and_zero_subgradient(self):
        _, t = self.boxes()
        for agbb in (False, True):
            p = t.clone().requires_grad_()
            loss = PaperBDLoss(agbb=agbb, square_ratio_threshold=2)(p, t)
            self.assertLess(abs(float(loss)), 1e-12)
            loss.backward()
            self.assertTrue(torch.isfinite(p.grad).all())
        center, shape = dcagpl_values(t, t)
        self.assertAlmostEqual(float(center), 0.)
        self.assertAlmostEqual(float(shape), -math.log(1 / 3 + 1e-6))

    def test_translation_scale_and_pi_equivalences(self):
        p, t = self.boxes()
        for loss in (PaperBDLoss(), PaperBDLoss(agbb=True, square_ratio_threshold=2),
                     PaperDCAGPLLoss()):
            base = loss(p, t)
            pp, tt = p.clone(), t.clone()
            pp[:, :2] += 123
            tt[:, :2] += 123
            torch.testing.assert_close(loss(pp, tt), base)
            pp[:, :4] *= 8
            tt[:, :4] *= 8
            torch.testing.assert_close(loss(pp, tt), base)
            pp[:, 4] += math.pi
            torch.testing.assert_close(loss(pp, tt), base)

    def test_agbb_exposes_square_angle_and_gt_switch(self):
        t = torch.tensor([[0., 0., 10., 10., 0.]], dtype=torch.float64)
        p = t.clone()
        p[:, 4] = math.pi / 4
        self.assertLess(float(PaperBDLoss()(p, t)), 1e-10)
        self.assertGreater(float(PaperBDLoss(agbb=True)(p, t)), .01)
        # Non-square GT must leave BOTH representations standard.
        p, t = self.boxes()
        torch.testing.assert_close(PaperBDLoss()(p, t), PaperBDLoss(agbb=True)(p, t))

    def test_centerness_empty_and_zero_weight(self):
        p, t = self.boxes()
        p, t = p.repeat(2, 1), t.repeat(2, 1)
        p[1, 0] += 2
        w = torch.tensor([1., .25], dtype=torch.float64)
        for loss in (PaperBDLoss(), PaperDCAGPLLoss()):
            raw = loss(p, t, reduction_override='none')
            torch.testing.assert_close(loss(p, t, weight=w, avg_factor=w.sum()),
                                       (raw * w).sum() / w.sum())
            empty = p[:0].requires_grad_()
            zero = loss(empty, t[:0])
            zero.backward()
            self.assertEqual(float(zero), 0)
            pp = p.clone().requires_grad_()
            zero = loss(pp, t, weight=w * 0, avg_factor=1e-6)
            zero.backward()
            self.assertEqual(float(zero), 0)
            self.assertTrue(torch.isfinite(pp.grad).all())

    def test_thin_and_collapsed_prediction_gradients(self):
        for sides in ((0., 0.), (0., 2.), (1e-8, 1000.), (1000., 1e-8)):
            for loss in (PaperBDLoss(), PaperBDLoss(agbb=True), PaperDCAGPLLoss()):
                with self.subTest(sides=sides, loss=type(loss).__name__):
                    p = torch.tensor([[.4, .8, *sides, .3]], requires_grad=True)
                    t = torch.tensor([[0., 0., 10., 9.8, -.2]])
                    value = loss(p, t)
                    value.backward()
                    self.assertTrue(torch.isfinite(value))
                    self.assertTrue(torch.isfinite(p.grad).all())

    def test_evcs_printed_formula_is_unbounded(self):
        t = torch.tensor([[0., 0., 10., 10., 0.]])
        p = t.clone()
        p[:, 2:4] += 1
        self.assertAlmostEqual(float(evcs_literal(p, t)), 2 - 2 * math.e)
        p[:, 2:4] += 100
        self.assertLess(float(evcs_literal(p, t)), -1e40)

    def test_configs_only_change_loss_and_output(self):
        root = 'configs/dior_r/server_trainval/'
        base = mmcv.Config.fromfile(root + 'riou.py')
        for filename in ('paper_bd.py', 'paper_bd_agbb.py', 'paper_dcagpl.py'):
            cfg = mmcv.Config.fromfile(root + filename)
            importlib.import_module(cfg.custom_imports.imports[0])
            for key in ('data', 'optimizer', 'optimizer_config', 'lr_config',
                        'runner', 'evaluation', 'seed', 'load_from', 'resume_from'):
                self.assertEqual(cfg[key], base[key], (filename, key))
            cfg.model.bbox_head.loss_bbox = base.model.bbox_head.loss_bbox
            self.assertEqual(cfg.model, base.model)

    def test_real_fcos_head_forward_backward_cpu_and_cuda(self):
        devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
        for device in devices:
            for name in ('paper_bd.py', 'paper_bd_agbb.py', 'paper_dcagpl.py'):
                with self.subTest(device=device, config=name):
                    torch.manual_seed(42)
                    cfg = mmcv.Config.fromfile('configs/dior_r/server_trainval/' + name)
                    head = build_head(cfg.model.bbox_head).to(device)
                    head.init_weights()
                    features = [torch.randn(1, 256, k, k, device=device)
                                for k in (16, 8, 4, 2, 1)]
                    outputs = head(features)
                    boxes = [torch.tensor([[64., 64., 48., 24., .2]], device=device)]
                    labels = [torch.tensor([1], device=device)]
                    meta = [dict(img_shape=(128, 128, 3), pad_shape=(128, 128, 3),
                                 scale_factor=1.)]
                    losses = head.loss(*outputs, boxes, labels, meta)
                    total = sum(v for k, v in losses.items() if 'loss' in k)
                    self.assertTrue(torch.isfinite(total))
                    total.backward()
                    self.assertGreater(float(head.conv_angle.weight.grad.abs().sum()), 0)
                    for param in head.parameters():
                        if param.grad is not None:
                            self.assertTrue(torch.isfinite(param.grad).all())
                    outputs = head(features)
                    empty = head.loss(*outputs, [boxes[0][:0]], [labels[0][:0]], meta)
                    self.assertEqual(float(empty['loss_bbox']), 0.)
                    del head, outputs, losses, total


if __name__ == '__main__':
    unittest.main(verbosity=2)
