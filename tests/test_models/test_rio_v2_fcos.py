"""Check source-formula agreement and FCOS training/inference integration."""
import ast
from pathlib import Path
import unittest
from unittest.mock import patch

import mmcv
import torch
import torch.nn.functional as F
from mmcv.utils import import_modules_from_strings

from mmrotate.models.builder import build_head
from mmrotate.models.losses.rio_v2_varifocal_loss import RiOV2VarifocalLoss


class TestRiOV2FCOS(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        cfg = mmcv.Config.fromfile('configs/dior_r/server_trainval/rio_v2_adamw.py')
        import_modules_from_strings(**cfg.custom_imports)
        self.cfg = cfg
        self.head = build_head(cfg.model.bbox_head)

    def inputs(self):
        shapes = [(8, 8), (4, 4), (2, 2), (1, 1), (1, 1)]
        return ([torch.randn(2, 20, h, w, requires_grad=True) for h, w in shapes],
                [torch.full((2, 4, h, w), 2., requires_grad=True) for h, w in shapes],
                [torch.full((2, 1, h, w), .1, requires_grad=True) for h, w in shapes],
                [torch.randn(2, 1, h, w, requires_grad=True) for h, w in shapes])

    def test_vfl_matches_pinned_repository_value_and_gradient(self):
        source = Path('work_dirs/dior_r/rio_v2_source_audit_20261023/repository/'
                      'engine/rtv4/rtv4_obb_criterion.py')
        if not source.is_file():
            self.skipTest('Pinned RiO v2 source snapshot is not present')
        tree = ast.parse(source.read_text())
        criterion = next(x for x in tree.body if isinstance(x, ast.ClassDef)
                         and x.name == 'RTv4OBBCriterion')
        methods = [x for x in criterion.body if isinstance(x, ast.FunctionDef)
                   and x.name in ('loss_labels_vfl', '_get_src_permutation_idx')]
        extracted = ast.Module(body=[ast.ClassDef(name='Reference', bases=[],
            keywords=[], body=methods, decorator_list=[])], type_ignores=[])
        namespace = dict(torch=torch, F=F)
        exec(compile(ast.fix_missing_locations(extracted), str(source), 'exec'), namespace)
        reference = namespace['Reference']()
        reference.num_classes, reference.alpha, reference.gamma = 3, .75, 2.
        pred = torch.tensor([[.2, -.3, 1.], [-1., 0., .4], [.8, -.5, .1]],
                            requires_grad=True)
        quality = torch.tensor([.3, .85, 0.], requires_grad=True)
        actual = RiOV2VarifocalLoss()(pred, torch.tensor([1, 0, 3]), quality, 2.)
        expected = reference.loss_labels_vfl(
            dict(pred_boxes=torch.zeros(1, 3, 5), pred_logits=pred[None]),
            [dict(labels=torch.tensor([1, 0]))],
            [(torch.tensor([0, 1]), torch.tensor([0, 1]))], 2.,
            values=quality[:2].detach())['loss_vfl']
        torch.testing.assert_close(actual, expected)
        ga = torch.autograd.grad(actual, (pred, quality), retain_graph=True,
                                 allow_unused=True)
        ge = torch.autograd.grad(expected, pred)[0]
        torch.testing.assert_close(ga[0], ge)
        self.assertIsNone(ga[1])

    def test_uniform_regression_and_detached_quality(self):
        inputs = self.inputs()
        metadata = [dict(img_shape=(64, 80, 3)), dict(img_shape=(80, 64, 3))]
        gt = [torch.tensor([[32., 32., 32., 16., .2]]) for _ in range(2)]
        original = self.head.loss_bbox.forward
        observed = []

        def capture(*args, **kwargs):
            observed.append((args, kwargs))
            return original(*args, **kwargs)

        with patch.object(self.head.loss_bbox, 'forward', side_effect=capture):
            losses = self.head.loss(*inputs, gt, [torch.tensor([0])] * 2, metadata)
        self.assertNotIn('loss_centerness', losses)
        args, kwargs = observed[0]
        self.assertNotIn('weight', kwargs)
        self.assertEqual(float(kwargs['avg_factor']), len(args[0]))
        self.assertTrue(0 <= float(losses['quality_mean']) <= 1)
        cls_to_bbox = torch.autograd.grad(losses['loss_cls'], inputs[1] + inputs[2],
                                         retain_graph=True, allow_unused=True)
        self.assertTrue(all(x is None for x in cls_to_bbox))
        (losses['loss_cls'] + losses['loss_bbox']).backward()
        self.assertTrue(all(x.grad is not None and torch.isfinite(x.grad).all()
                            for x in inputs[0] + inputs[1] + inputs[2]))
        self.assertTrue(all(x.grad is None for x in inputs[3]))
        self.assertGreater(sum(float(x.grad.abs().sum()) for x in inputs[2]), 0)

    def test_empty_gt_is_finite_and_backpropagates(self):
        inputs = self.inputs()
        losses = self.head.loss(*inputs, [torch.empty(0, 5)] * 2,
                                [torch.empty(0, dtype=torch.long)] * 2,
                                [dict(img_shape=(64, 64, 3))] * 2)
        self.assertEqual(float(losses['loss_bbox']), 0.)
        self.assertTrue(torch.isfinite(losses['loss_cls']))
        (losses['loss_cls'] + losses['loss_bbox']).backward()
        self.assertTrue(all(x.grad is not None for x in inputs[1] + inputs[2]))

    def test_inference_ranking_and_scores_ignore_centerness(self):
        scores = torch.full((20, 1, 2), -20.)
        scores[0, 0] = torch.logit(torch.tensor([.7, .8]))
        distances, angles = torch.full((4, 1, 2), 2.), torch.zeros(1, 1, 2)
        points = [torch.tensor([[4., 4.], [12., 4.]])]
        cfg = self.cfg.model.test_cfg.copy()
        cfg.nms_pre = 1
        results = []
        for center in (torch.tensor([[[100., -100.]]]),
                       torch.tensor([[[-100., 100.]]])):
            results.append(self.head._get_bboxes_single(
                [scores], [distances], [angles], [center], points,
                (64, 64, 3), [1., 1., 1., 1.], cfg))
        torch.testing.assert_close(results[0][0], results[1][0])
        self.assertEqual(results[0][0].shape[0], 1)
        self.assertAlmostEqual(float(results[0][0][0, 0]), 12.)
        self.assertAlmostEqual(float(results[0][0][0, -1]), .8, places=6)

    def test_original_protocol_and_predictions_are_preserved(self):
        old = mmcv.Config.fromfile('configs/dior_r/server_trainval/rio_adamw.py')
        for key in ('data', 'optimizer', 'optimizer_config', 'runner', 'lr_config',
                    'evaluation', 'seed', 'load_from', 'resume_from'):
            self.assertEqual(self.cfg[key], old[key], key)
        self.assertEqual(self.cfg.model.bbox_head.loss_bbox, old.model.bbox_head.loss_bbox)
        baseline = build_head(old.model.bbox_head)
        self.head.load_state_dict(baseline.state_dict())
        feature = torch.randn(1, 256, 8, 8)
        for training in (True, False):
            self.head.train(training)
            baseline.train(training)
            left = self.head.forward_single(feature, self.head.scales[0], self.head.strides[0])
            right = baseline.forward_single(feature, baseline.scales[0], baseline.strides[0])
            for x, y in zip(left, right):
                torch.testing.assert_close(x, y)


if __name__ == '__main__':
    unittest.main()
