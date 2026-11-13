"""Numerical and FCOS integration checks for the RiO regression experiment."""
import math
import unittest
from unittest.mock import patch

import mmcv
import torch

from mmrotate.models.builder import build_head
from mmrotate.models.losses.gaussian_dist_loss import GDLoss
from mmrotate.models.losses.rotated_rio_loss import (
    RotatedRiOLoss, periodic_l1, stable_box_kld)


class TestRotatedRiOLoss(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_kld_matches_existing_stable_loss(self):
        generator = torch.Generator().manual_seed(42)
        pred = torch.rand(40, 5, generator=generator, dtype=torch.float64)
        target = torch.rand(40, 5, generator=generator, dtype=torch.float64)
        for box in (pred, target):
            box[:, :2] *= 10
            box[:, 2:4] = box[:, 2:4] * 10 + 1
            box[:, 4] = box[:, 4] * math.pi - math.pi / 2
        expected = GDLoss(loss_type='kld', fun='log1p', tau=1,
                          sqrt=False, reduction='none')(pred, target)
        torch.testing.assert_close(stable_box_kld(pred, target), expected,
                                   atol=1e-10, rtol=1e-10)

    def test_identity_and_pi_period(self):
        target = torch.tensor([[10., 20., 12., 3., .4]], dtype=torch.float64)
        scale = torch.full((1, 4), .01, dtype=torch.float64)
        loss = RotatedRiOLoss()
        self.assertAlmostEqual(float(loss(target, target, spatial_scale=scale)), 0)
        pred = target.clone()
        pred[:, 4] += math.pi
        self.assertAlmostEqual(float(loss(pred, target, spatial_scale=scale)), 0)

    def test_kld_width_height_equivalence(self):
        target = torch.tensor([[10., 20., 12., 3., .4]], dtype=torch.float64)
        pred = target.clone()
        pred[:, 2:4] = target[:, [3, 2]]
        pred[:, 4] += math.pi / 2
        self.assertAlmostEqual(float(stable_box_kld(pred, target)), 0)
        # Full parameter loss intentionally uses the GT canonical width/height:
        # only KLD has the width/height-swap equivalence.

    def test_periodic_seam_value_and_gradient(self):
        pred = torch.tensor([math.radians(89.)], dtype=torch.float64,
                            requires_grad=True)
        target = torch.tensor([math.radians(-89.)], dtype=torch.float64)
        loss = periodic_l1(pred, target)
        self.assertAlmostEqual(float(loss), math.radians(2))
        loss.backward()
        self.assertEqual(float(pred.grad), -1.)

    def test_kld_degenerate_boxes_have_finite_gradients(self):
        for sides in ((0., 0.), (0., 2.), (1e-8, 2.), (2., 0.)):
            with self.subTest(sides=sides):
                pred = torch.tensor([[0., 0., *sides, .3]], requires_grad=True)
                target = torch.tensor([[.2, .1, 2., 1., .2]])
                value = stable_box_kld(pred, target).sum()
                value.backward()
                self.assertTrue(torch.isfinite(value))
                self.assertTrue(torch.isfinite(pred.grad).all())

    def test_combined_loss_gradcheck(self):
        pred = torch.tensor([[.4, .8, 9., 5., .13]], dtype=torch.float64,
                            requires_grad=True)
        target = torch.tensor([[0., 0., 10., 6., -.2]], dtype=torch.float64)
        scale = torch.full((1, 4), .01, dtype=torch.float64)
        self.assertTrue(torch.autograd.gradcheck(
            lambda p: RotatedRiOLoss()(p, target, spatial_scale=scale), (pred,)))

    def test_centerness_weighting_components_and_empty(self):
        target = torch.tensor([[0., 0., 10., 4., .1]]).repeat(2, 1)
        pred = target + torch.tensor([[1., 0., 1., 0., .2],
                                     [0., 2., 0., 1., -.1]])
        scale = torch.full((2, 4), .01)
        weight = torch.tensor([1., .25])
        loss = RotatedRiOLoss()
        raw = loss(pred, target, spatial_scale=scale, reduction_override='none')
        parts = loss(pred, target, spatial_scale=scale, weight=weight,
                     avg_factor=weight.sum(), return_components=True)
        expected = (raw * weight).sum() / weight.sum()
        torch.testing.assert_close(sum(parts.values()), expected)
        self.assertTrue(all('loss' not in name for name in parts))
        self.assertEqual(float(loss(pred, target, spatial_scale=scale,
                                    weight=weight * 0, avg_factor=1e-6)), 0)
        empty = pred[:0].requires_grad_()
        zero = loss(empty, target[:0], spatial_scale=scale[:0])
        self.assertEqual(float(zero), 0)
        zero.backward()

    def test_spatial_scale_consistency_across_fpn_levels(self):
        target = torch.tensor([[100., 100., 40., 20., .2]], dtype=torch.float64)
        pred = target + torch.tensor([[8., 4., 16., 8., .1]], dtype=torch.float64)
        values = []
        for stride in (8, 16, 32):
            p, t = pred.clone(), target.clone()
            p[:, :4] /= stride
            t[:, :4] /= stride
            scale = torch.full((1, 4), stride / 800, dtype=torch.float64)
            values.append(RotatedRiOLoss()(p, t, spatial_scale=scale))
        torch.testing.assert_close(values[0], values[1])
        torch.testing.assert_close(values[0], values[2])

    def test_config_only_changes_loss_and_output(self):
        baseline = mmcv.Config.fromfile('configs/dior_r/server_trainval/riou.py')
        cfg = mmcv.Config.fromfile('configs/dior_r/server_trainval/rio.py')
        for key in ('data', 'optimizer', 'optimizer_config', 'runner',
                    'lr_config', 'evaluation', 'seed', 'load_from', 'resume_from'):
            self.assertEqual(cfg[key], baseline[key], key)
        cfg.model.bbox_head.loss_bbox = baseline.model.bbox_head.loss_bbox
        self.assertEqual(cfg.model, baseline.model)

    def test_head_normalizes_by_each_image_and_fpn_stride(self):
        cfg = mmcv.Config.fromfile('configs/dior_r/server_trainval/rio.py')
        head = build_head(cfg.model.bbox_head)
        shapes = [(8, 8), (4, 4), (2, 2), (1, 1), (1, 1)]
        cls = [torch.zeros(2, 20, h, w, requires_grad=True) for h, w in shapes]
        boxes = [torch.full((2, 4, h, w), 2., requires_grad=True) for h, w in shapes]
        angles = [torch.full((2, 1, h, w), .1, requires_grad=True) for h, w in shapes]
        centers = [torch.zeros(2, 1, h, w, requires_grad=True) for h, w in shapes]
        original = head.loss_bbox.forward
        observed = []

        def capture(*args, **kwargs):
            observed.append(kwargs['spatial_scale'])
            return original(*args, **kwargs)

        with patch.object(head.loss_bbox, 'forward', side_effect=capture):
            losses = head.loss(cls, boxes, angles, centers,
                [torch.tensor([[32., 32., 32., 16., .2]]) for _ in range(2)],
                [torch.tensor([0]) for _ in range(2)],
                [dict(img_shape=(64, 64, 3)), dict(img_shape=(64, 128, 3))])
        ratios = observed[0][:, 0] / observed[0][:, 1]
        self.assertTrue((ratios == 1).any())
        self.assertTrue((ratios == .5).any())
        torch.testing.assert_close(losses['loss_bbox'], sum(
            losses[k] for k in ('bbox_kld', 'bbox_l1', 'bbox_sp_l1')))
        sum(value for name, value in losses.items() if 'loss' in name).backward()
        self.assertTrue(all(torch.isfinite(x.grad).all() for x in boxes + angles))

    def test_bounded_head_matches_controlled_recovery_parameterization(self):
        cfg = mmcv.Config.fromfile('configs/dior_r/server_trainval/rio_bounded.py')
        bounded = build_head(cfg.model.bbox_head)
        raw_config = cfg.model.bbox_head.copy()
        raw_config.bounded_angle = False
        raw = build_head(raw_config)
        raw.load_state_dict(bounded.state_dict())
        features = torch.randn(2, 256, 8, 8)
        for training in (True, False):
            bounded.train(training)
            raw.train(training)
            expected = raw.forward_single(features, raw.scales[0], 8)
            actual = bounded.forward_single(features, bounded.scales[0], 8)
            torch.testing.assert_close(actual[2], math.pi * (expected[2].sigmoid() - .5))
            for index in (0, 1, 3):
                torch.testing.assert_close(actual[index], expected[index])

    def test_bounded_angle_range_and_logit_gradient(self):
        cfg = mmcv.Config.fromfile('configs/dior_r/server_trainval/rio_bounded.py')
        head = build_head(cfg.model.bbox_head)
        logits = torch.tensor([-100., -1., 0., 1., 100.]).reshape(1, 1, 1, 5)
        logits.requires_grad_()
        with patch.object(head.conv_angle, 'forward', return_value=logits):
            angles = head.forward_single(torch.randn(1, 256, 1, 5), head.scales[0], 8)[2]
        self.assertTrue((angles.abs() <= math.pi / 2).all())
        angles.sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertAlmostEqual(float(logits.grad.flatten()[2]), math.pi / 4, places=6)
        self.assertFalse(hasattr(head, 'scale_angle'))

    def test_bounded_config_preserves_loss_and_training_protocol(self):
        old = mmcv.Config.fromfile('configs/dior_r/server_trainval/rio.py')
        new = mmcv.Config.fromfile('configs/dior_r/server_trainval/rio_bounded.py')
        self.assertFalse(new.model.bbox_head.scale_angle)
        self.assertTrue(new.model.bbox_head.bounded_angle)
        for key in ('data', 'optimizer', 'optimizer_config', 'runner',
                    'lr_config', 'evaluation', 'seed', 'load_from', 'resume_from'):
            self.assertEqual(old[key], new[key], key)
        self.assertEqual(old.model.bbox_head.loss_bbox, new.model.bbox_head.loss_bbox)
        new.model.bbox_head.pop('bounded_angle')
        new.model.bbox_head.scale_angle = old.model.bbox_head.scale_angle
        self.assertEqual(old.model, new.model)

    def test_bounded_angle_rejects_learnable_scale(self):
        cfg = mmcv.Config.fromfile('configs/dior_r/server_trainval/rio_bounded.py')
        cfg.model.bbox_head.scale_angle = True
        with self.assertRaisesRegex(ValueError, 'scale_angle=False'):
            build_head(cfg.model.bbox_head)

    def test_unit_period_angle_matches_normalized_coordinates(self):
        pred = torch.tensor([[0., 0., 10., 4., math.radians(89.)]], requires_grad=True)
        target = torch.tensor([[0., 0., 10., 4., math.radians(-89.)]])
        loss = RotatedRiOLoss(kld_weight=0, spatial_weight=0, angle_weight=5,
                             angle_normalizer=math.pi)
        value = loss(pred, target, spatial_scale=torch.full((1, 4), .01))
        self.assertAlmostEqual(float(value), 5 * 2 / 180, places=6)
        value.backward()
        self.assertAlmostEqual(float(pred.grad[0, 4]), -5 / math.pi, places=6)

    def test_covariance_regularizer_matches_matrix_reference(self):
        generator = torch.Generator().manual_seed(71)
        pred = torch.rand(20, 5, dtype=torch.float64, generator=generator)
        target = torch.rand(20, 5, dtype=torch.float64, generator=generator)
        pred[:, 2:4] = pred[:, 2:4] * .2 + .001
        target[:, 2:4] = target[:, 2:4] * .2 + .001
        pred.requires_grad_()

        def covariance(box):
            c, s = box[:, 4].cos(), box[:, 4].sin()
            rotation = torch.stack((c, -s, s, c), -1).reshape(-1, 2, 2)
            diagonal = torch.diag_embed((box[:, 2:4].clamp_min(1e-8) / 2).square())
            return rotation @ diagonal @ rotation.transpose(-1, -2) + 1e-8 * torch.eye(
                2, dtype=box.dtype)

        p_cov, t_cov = covariance(pred), covariance(target)
        inverse = torch.linalg.inv(p_cov)
        delta = (target[:, :2] - pred[:, :2]).unsqueeze(-1)
        divergence = .5 * ((inverse @ t_cov).diagonal(dim1=-2, dim2=-1).sum(-1)
            + (delta.transpose(-1, -2) @ inverse @ delta).flatten()
            + torch.logdet(p_cov) - torch.logdet(t_cov) - 2)
        transformed = torch.log1p(divergence.clamp_min(0))
        expected = transformed / (1 + transformed)
        actual = stable_box_kld(pred, target, min_size=1e-8, covariance_epsilon=1e-8)
        torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-9)
        g_actual = torch.autograd.grad(actual.sum(), pred, retain_graph=True)[0]
        g_expected = torch.autograd.grad(expected.sum(), pred)[0]
        torch.testing.assert_close(g_actual, g_expected, atol=1e-7, rtol=1e-7)

    def test_normalized_kld_regularization_is_fpn_consistent(self):
        pred = torch.tensor([[8., 12., .01, 10., .3]], dtype=torch.float64)
        target = torch.tensor([[10., 10., 20., 12., .2]], dtype=torch.float64)
        values = []
        for stride in (8, 16, 32):
            p, t = pred.clone(), target.clone()
            p[:, :4] /= stride
            t[:, :4] /= stride
            scale = torch.full((1, 4), stride / 800, dtype=torch.float64)
            values.append(RotatedRiOLoss(min_size=1e-8, kld_normalized=True,
                covariance_epsilon=1e-8, angle_normalizer=math.pi)(p, t, spatial_scale=scale))
        torch.testing.assert_close(values[0], values[1])
        torch.testing.assert_close(values[0], values[2])

    def test_normalized_loss_has_finite_gradients_for_degenerate_boxes(self):
        pred = torch.tensor([[0., 0., 0., 2., .3]], requires_grad=True)
        target = torch.tensor([[.2, .1, 2., 1., .2]])
        loss = RotatedRiOLoss(min_size=1e-8, kld_normalized=True,
            covariance_epsilon=1e-8, angle_normalizer=math.pi)(
                pred, target, spatial_scale=torch.full((1, 4), .01))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_normalized_config_keeps_original_head_and_protocol(self):
        cfg = mmcv.Config.fromfile('configs/dior_r/server_trainval/rio_normalized.py')
        old = mmcv.Config.fromfile('configs/dior_r/server_trainval/rio.py')
        normalized = cfg.model.bbox_head.loss_bbox
        self.assertEqual(normalized.angle_normalizer, math.pi)
        self.assertEqual(normalized.covariance_epsilon, 1e-8)
        self.assertEqual(normalized.min_size, 1e-8)
        for name in ('kld_weight', 'spatial_weight', 'angle_weight'):
            self.assertEqual(normalized[name], 5.)
        cfg.model.bbox_head.loss_bbox = old.model.bbox_head.loss_bbox
        self.assertEqual(cfg.model, old.model)
        for key in ('data', 'optimizer', 'runner', 'lr_config', 'evaluation'):
            self.assertEqual(cfg[key], old[key], key)

    def test_adamw_candidate_has_a_matched_riou_control(self):
        candidate = mmcv.Config.fromfile('configs/dior_r/server_trainval/rio_adamw.py')
        control = mmcv.Config.fromfile('configs/dior_r/server_trainval/riou_adamw.py')
        normalized = mmcv.Config.fromfile('configs/dior_r/server_trainval/rio_normalized.py')
        self.assertEqual(candidate.model.bbox_head.loss_bbox,
                         normalized.model.bbox_head.loss_bbox)
        candidate.model.bbox_head.loss_bbox = control.model.bbox_head.loss_bbox
        for key in ('model', 'data', 'optimizer', 'optimizer_config', 'runner',
                    'lr_config', 'evaluation', 'load_from', 'resume_from', 'seed'):
            self.assertEqual(candidate[key], control[key], key)
        self.assertEqual(candidate.optimizer.type, 'AdamW')
        self.assertNotIn('momentum', candidate.optimizer)
        self.assertEqual(candidate.optimizer_config.grad_clip.max_norm, .1)


if __name__ == '__main__':
    unittest.main()
