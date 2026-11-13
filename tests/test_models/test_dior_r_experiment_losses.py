"""Numerical, gradient and FCOS-integration checks for the two new experiments.

Run without pytest: python -m unittest discover -s tests/test_models
                       -p test_dior_r_experiment_losses.py -v
"""
import math
import unittest
from unittest.mock import patch

import mmcv
import torch
from mmcv.ops import box_iou_rotated

from mmrotate.models.builder import build_head
from mmrotate.models.losses import (EncodedRotatedSmoothL1Loss, RotatedIoULoss,
                                    RotatedSmoothL1Loss, RotatedWFIoULoss)
from mmrotate.models.losses.rotated_wf_iou_loss import (
    rotated_wf_geometry, wf_weighting_factor)


class TestDiorExperimentLosses(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_horizontal_paper_formula(self):
        # Independent closed-form rectangles cover high/mid/low/no overlap.
        shifts = torch.tensor([0., 1., 4., 8., 9.9, 20.], dtype=torch.float64)
        pred = torch.zeros((6, 5), dtype=torch.float64)
        pred[:, 0] = shifts
        pred[:, 2:4] = torch.tensor([10., 2.])
        target = pred.clone()
        target[:, 0] = 0
        iou, ratio = rotated_wf_geometry(pred, target)
        overlap = (10 - shifts).clamp(min=0)
        expected_iou = overlap / (20 - overlap)
        expected_ratio = torch.where(
            overlap > 0, (overlap.square() + 4) / ((10 + shifts).square() + 4),
            1e-14 / ((10 + shifts).square() + 4))
        torch.testing.assert_close(iou, expected_iou, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(ratio, expected_ratio, atol=1e-10, rtol=1e-10)
        expected = []
        for value, geometry in zip(expected_iou.tolist(), expected_ratio.tolist()):
            factor = geometry if value > .7 else (
                min(1 / geometry, 6) if value < .3 else 1)
            expected.append(max(1 - factor * value, 0))
        actual = RotatedWFIoULoss(reduction='none')(pred, target)
        torch.testing.assert_close(actual, torch.tensor(expected, dtype=actual.dtype))

    def test_thresholds_and_cap(self):
        iou = torch.tensor([.2, .3, .5, .7, .8])
        ratio = torch.full_like(iou, .1)
        torch.testing.assert_close(
            wf_weighting_factor(iou, ratio),
            torch.tensor([6., 1., 1., 1., .1]))

    def test_rotated_iou_against_compiled_reference(self):
        generator = torch.Generator().manual_seed(42)
        target = torch.rand((256, 5), generator=generator)
        target[:, :2] *= 800
        target[:, 2:4] = target[:, 2:4] * 80 + 2
        target[:, 4] = (target[:, 4] - .5) * math.pi
        pred = target.clone()
        pred[:, :2] += torch.randn((256, 2), generator=generator) * 15
        pred[:, 2:4] *= torch.rand((256, 2), generator=generator) + .5
        pred[:, 4] += torch.randn((256,), generator=generator) * .4
        actual, ratio = rotated_wf_geometry(pred, target)
        reference = box_iou_rotated(pred, target, aligned=True).reshape(-1)
        torch.testing.assert_close(actual, reference, atol=1e-4, rtol=1e-4)
        self.assertTrue(((ratio >= 0) & (ratio <= 1)).all())

    def test_identity_and_equivalent_box_representation(self):
        target = torch.tensor([[100., 200., 10., 4., .4],
                               [400., 300., 40., 30., -1.2]])
        loss = RotatedWFIoULoss(reduction='none')
        torch.testing.assert_close(loss(target, target), torch.zeros(2), atol=1e-6, rtol=0)
        pred = target.clone()
        pred[:, 2], pred[:, 3] = target[:, 3], target[:, 2]
        pred[:, 4] += math.pi / 2
        torch.testing.assert_close(loss(pred, target), torch.zeros(2), atol=1e-6, rtol=0)

    def test_joint_rotation_and_translation(self):
        pred = torch.tensor([[2., 1., 10., 4., .2]], dtype=torch.float64)
        target = torch.tensor([[0., 0., 10., 4., -.1]], dtype=torch.float64)
        expected = RotatedWFIoULoss()(pred, target)
        angle = .9
        matrix = pred.new_tensor([[math.cos(angle), -math.sin(angle)],
                                  [math.sin(angle), math.cos(angle)]])
        rotated = []
        for box in (pred, target):
            copy = box.clone()
            copy[:, :2] = box[:, :2] @ matrix.T + box.new_tensor([500., 600.])
            copy[:, 4] += angle
            rotated.append(copy)
        torch.testing.assert_close(RotatedWFIoULoss()(*rotated), expected)

    def test_wf_backward_weights_empty_and_nonoverlap(self):
        target = torch.tensor([[0., 0., 10., 4., 0.]]).repeat(4, 1)
        pred = target.clone()
        pred[:, 0] = torch.tensor([1., 4., 8., 20.])
        pred[:, 4] = .1
        pred.requires_grad_()
        loss = RotatedWFIoULoss()
        raw = loss(pred, target, reduction_override='none')
        weights = torch.tensor([1., .5, .25, 0.])
        actual = loss(pred, target, weight=weights, avg_factor=weights.sum())
        torch.testing.assert_close(actual, (raw * weights).sum() / weights.sum())
        torch.testing.assert_close(
            loss(pred, target, weight=weights[:, None].expand(-1, 5),
                 avg_factor=weights.sum()), actual)
        actual.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertGreater(pred.grad[:3].abs().sum().item(), 0)
        self.assertEqual(raw[-1].item(), 1)
        for reduction in ('none', 'mean', 'sum'):
            zero = loss(pred, target, weight=torch.zeros(4), reduction_override=reduction)
            self.assertEqual(zero.sum().item(), 0)
            empty = pred[:0]
            self.assertEqual(loss(empty, target[:0], reduction_override=reduction).sum().item(), 0)

    def test_wf_gradient_against_finite_difference(self):
        pred = torch.tensor([[.4, .8, 9., 5., .13]], dtype=torch.float64,
                            requires_grad=True)
        target = torch.tensor([[0., 0., 10., 6., -.2]], dtype=torch.float64)
        self.assertTrue(torch.autograd.gradcheck(
            lambda boxes: RotatedWFIoULoss()(boxes, target), (pred,),
            eps=1e-5, atol=1e-4, rtol=1e-3))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is unavailable')
    def test_wf_cuda_matches_cpu_and_backward(self):
        pred = torch.tensor([[1., .3, 10., 4., .2],
                             [4., .1, 10., 4., -.1],
                             [8., .2, 10., 4., .1],
                             [20., 0., 10., 4., 0.]])
        target = torch.tensor([[0., 0., 10., 4., 0.]]).repeat(4, 1)
        expected = RotatedWFIoULoss(reduction='none')(pred, target)
        gpu_pred = pred.cuda().requires_grad_()
        actual = RotatedWFIoULoss(reduction='none')(gpu_pred, target.cuda())
        torch.testing.assert_close(actual.cpu(), expected, atol=1e-6, rtol=1e-5)
        actual.sum().backward()
        self.assertTrue(torch.isfinite(gpu_pred.grad).all())
        linear = RotatedIoULoss(mode='linear', reduction='none')(
            gpu_pred.detach(), target.cuda())
        reference = 1 - box_iou_rotated(pred, target, aligned=True).reshape(-1)
        torch.testing.assert_close(linear.cpu(), reference, atol=2e-6, rtol=1e-5)

    def test_optional_linear_control_config(self):
        base = mmcv.Config.fromfile(
            'configs/dior_r/rotated_fcos_r50_fpn_1x_dior_r_le90_bs2.py')
        cfg = mmcv.Config.fromfile(
            'configs/dior_r/rotated_fcos_r50_fpn_1x_dior_r_le90_riou_linear_bs2.py')
        self.assertEqual(cfg.data, base.data)
        self.assertEqual(cfg.optimizer, base.optimizer)
        head = build_head(cfg.model.bbox_head)
        self.assertEqual(head.loss_bbox.mode, 'linear')

    def test_smooth_l1_centerness_broadcast_and_backward(self):
        target = torch.zeros((2, 5))
        pred = torch.tensor([[.5, 2., 0., 0., .1],
                             [1., 3., 0., 0., -.1]], requires_grad=True)
        weights = torch.tensor([1., .25])
        residual = pred.detach().abs()
        terms = torch.where(residual < 1, residual.square() / 2, residual - .5)
        expected = (terms * weights[:, None]).sum() / weights.sum()
        actual = RotatedSmoothL1Loss()(pred, target, weights, avg_factor=weights.sum())
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_encoded_smooth_l1_canonical_angle_and_weights(self):
        target = torch.tensor([[1., 2., 3., 4., 1.5],
                               [1., 2., 3., 4., 0.]])
        pred = target.clone()
        pred[0, 4] = -1.5
        pred[1, 0] += 2
        pred.requires_grad_()
        weights = torch.tensor([1., .25])
        expected = (2.5 + .25 * 1.5) / 1.25
        loss = EncodedRotatedSmoothL1Loss()
        actual = loss(pred, target, weight=weights, avg_factor=weights.sum())
        self.assertAlmostEqual(float(actual), expected, places=6)
        actual.backward()
        # Distances use the GT axes, so the angle must regress the same canonical
        # representation. Wrapping only this residual would learn wrong axes.
        self.assertLess(float(pred.grad[0, 4]), 0)

    def test_encoded_head_bypasses_decoder(self):
        cfg = mmcv.Config.fromfile(
            'configs/dior_r/rotated_fcos_r50_fpn_1x_dior_r_le90_smooth_l1_bs2.py')
        head = build_head(cfg.model.bbox_head)
        shapes = [(8, 8), (4, 4), (2, 2), (1, 1), (1, 1)]
        cls = [torch.zeros(1, 20, h, w, requires_grad=True) for h, w in shapes]
        boxes = [torch.ones(1, 4, h, w, requires_grad=True) for h, w in shapes]
        angles = [torch.zeros(1, 1, h, w, requires_grad=True) for h, w in shapes]
        centers = [torch.zeros(1, 1, h, w, requires_grad=True) for h, w in shapes]
        with patch.object(head.bbox_coder, 'decode', side_effect=AssertionError(
                'encoded training must not use rotated decoding')):
            losses = head.loss(cls, boxes, angles, centers,
                [torch.tensor([[32., 32., 32., 16., .1]])],
                [torch.tensor([0])], [dict(img_shape=(64, 64, 3))])
        sum(losses.values()).backward()
        self.assertGreater(sum(x.grad.abs().sum().item() for x in boxes), 0)
        self.assertGreater(sum(x.grad.abs().sum().item() for x in angles), 0)

    def test_encoded_head_requires_normalized_distances(self):
        cfg = mmcv.Config.fromfile(
            'configs/dior_r/rotated_fcos_r50_fpn_1x_dior_r_le90_smooth_l1_bs2.py')
        cfg.model.bbox_head.norm_on_bbox = False
        with self.assertRaisesRegex(ValueError, 'norm_on_bbox=True'):
            build_head(cfg.model.bbox_head)

    def test_encoded_negative_distances_can_recover(self):
        cfg = mmcv.Config.fromfile(
            'configs/dior_r/rotated_fcos_r50_fpn_1x_dior_r_le90_smooth_l1_bs2.py')
        head = build_head(cfg.model.bbox_head)
        head.init_weights()
        with torch.no_grad():
            head.conv_reg.weight.zero_()
            head.conv_reg.bias.fill_(-1)
        features = [torch.randn(1, 256, h, w) for h, w in
                    [(8, 8), (4, 4), (2, 2), (1, 1), (1, 1)]]
        head.train()
        outputs = head(features)
        self.assertTrue(all((b < 0).all() for b in outputs[1]))
        losses = head.loss(*outputs,
            [torch.tensor([[32., 32., 32., 16., .1]])],
            [torch.tensor([0])], [dict(img_shape=(64, 64, 3))])
        losses['loss_bbox'].backward()
        self.assertTrue((head.conv_reg.bias.grad < 0).all())
        head.eval()
        with torch.no_grad():
            self.assertTrue(all((b == 0).all() for b in head(features)[1]))

    def test_iou_and_wf_keep_original_training_clamp(self):
        for filename in ('rotated_fcos_r50_fpn_1x_dior_r_le90_bs2.py',
                         'rotated_fcos_r50_fpn_1x_dior_r_le90_wf_bs2.py'):
            with self.subTest(config=filename):
                cfg = mmcv.Config.fromfile('configs/dior_r/' + filename)
                head = build_head(cfg.model.bbox_head)
                with torch.no_grad():
                    head.conv_reg.weight.zero_()
                    head.conv_reg.bias.fill_(-1)
                head.train()
                outputs = head([torch.randn(1, 256, h, w) for h, w in
                                [(8, 8), (4, 4), (2, 2), (1, 1), (1, 1)]])
                self.assertTrue(all((b == 0).all() for b in outputs[1]))

    def test_encoded_angle_pi_alone_is_not_the_same_box(self):
        cfg = mmcv.Config.fromfile(
            'configs/dior_r/rotated_fcos_r50_fpn_1x_dior_r_le90_smooth_l1_bs2.py')
        head = build_head(cfg.model.bbox_head)
        point = torch.tensor([[32., 32.]])
        target = torch.tensor([[1., 2., 3., 4., .4]])
        pred = target.clone()
        pred[:, 4] += math.pi
        self.assertFalse(torch.allclose(head.bbox_coder.decode(point, pred)[:, :2],
                                       head.bbox_coder.decode(point, target)[:, :2]))
        self.assertGreater(float(head.loss_bbox(pred, target)), 0)
        self.assertEqual(float(head.loss_bbox(target, target)), 0)

    def test_full_800_fcos_loss_and_matched_configs(self):
        baseline = mmcv.Config.fromfile(
            'configs/dior_r/rotated_fcos_r50_fpn_1x_dior_r_le90_bs2.py')
        for suffix in ('smooth_l1', 'wf'):
            with self.subTest(loss=suffix):
                cfg = mmcv.Config.fromfile(
                    'configs/dior_r/rotated_fcos_r50_fpn_1x_dior_r_le90_'
                    f'{suffix}_bs2.py')
                self.assertEqual(cfg.data, baseline.data)
                self.assertEqual(cfg.optimizer, baseline.optimizer)
                self.assertEqual(cfg.runner, baseline.runner)
                self.assertEqual(cfg.lr_config, baseline.lr_config)
                self.assertEqual(cfg.evaluation, baseline.evaluation)
                self.assertEqual(cfg.seed, baseline.seed)
                head = build_head(cfg.model.bbox_head)
                shapes = [(100, 100), (50, 50), (25, 25), (13, 13), (7, 7)]
                cls = [torch.zeros(2, 20, h, w, requires_grad=True) for h, w in shapes]
                boxes = [torch.full((2, 4, h, w), 20., requires_grad=True) for h, w in shapes]
                angles = [torch.full((2, 1, h, w), .1, requires_grad=True) for h, w in shapes]
                centers = [torch.zeros(2, 1, h, w, requires_grad=True) for h, w in shapes]
                gt = [torch.tensor([[400., 400., 64., 32., .1]]) for _ in range(2)]
                labels = [torch.tensor([0]) for _ in range(2)]
                metas = [dict(img_shape=(800, 800, 3)) for _ in range(2)]
                losses = head.loss(cls, boxes, angles, centers, gt, labels, metas)
                self.assertTrue(all(torch.isfinite(x) for x in losses.values()))
                sum(losses.values()).backward()
                self.assertTrue(all(torch.isfinite(x.grad).all() for x in boxes))
                self.assertGreater(sum(x.grad.abs().sum().item() for x in boxes), 0)


if __name__ == '__main__':
    unittest.main()
