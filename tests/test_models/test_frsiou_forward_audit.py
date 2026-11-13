"""Independent per-prediction formula checks, including production CUDA wiring."""
import copy
import math
import unittest

import mmcv
import torch
from mmrotate.models.builder import build_head, build_loss
from mmrotate.models.losses.fractional_semantic_iou_loss import (
    FCOSFrSIoUEngineeringLoss, FCOSFrSIoUForwardLoss, fractional_terms,
    gl_coefficients)
from mmrotate.models.losses.semantic_iou_loss import (
    SemanticIoULoss, geometric_iou, semantic_scores)
from mmrotate.models.dense_heads.fractional_rotated_fcos_head import (
    FractionalRotatedFCOSHead)


class TestFrSIoUForwardAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def terms(self, q, lag, scope='per_box'):
        return fractional_terms(
            q, q.new_zeros((len(q), 5)), torch.zeros(len(q), dtype=torch.long),
            q.new_tensor([1., 3.]), lag, q.new_zeros((1, 2, 5)),
            gl_coefficients(.5, 2), .3, modulation_scope=scope)

    def test_per_box_expansion_is_not_exp_of_current_group_mean(self):
        q = torch.tensor([.2, .8], dtype=torch.double, requires_grad=True)
        lag = torch.tensor([[.5, .4]], dtype=torch.double)
        actual, mean, _, derivative, modulation = self.terms(q, lag)
        # Written independently from the implementation, using Eq.(12).
        expected = q.new_tensor([
            math.exp(-.3 * (.2 - .5 * .5 - .125 * .4)) * .8,
            math.exp(-.3 * (.8 - .5 * .5 - .125 * .4)) * .2])
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(derivative, q.new_tensor([-.1, .5]))
        self.assertAlmostEqual(float(mean), .65)
        self.assertNotEqual(float(modulation[0]), float(modulation[1]))
        legacy = self.terms(q, lag, 'group')[0]
        self.assertGreater(float((actual - legacy).abs().max()), .05)
        # Zero-padded cold start still depends on each current prediction.
        cold = self.terms(q, lag * 0)[0]
        torch.testing.assert_close(cold, torch.exp(-.3 * q) * (1 - q))

    def test_current_gradients_include_weight_and_have_no_cross_box_terms(self):
        q = torch.tensor([.2, .8], dtype=torch.double, requires_grad=True)
        lag = torch.tensor([[.5, .4]], dtype=torch.double, requires_grad=True)
        values = self.terms(q, lag)[0]
        gradient, old_gradient = torch.autograd.grad(
            values.sum(), (q, lag), allow_unused=True)
        expected = q.new_tensor([
            -math.exp(.03) * (1 + .3 * .8),
            -math.exp(-.15) * (1 + .3 * .2)])
        torch.testing.assert_close(gradient, expected)
        self.assertIsNone(old_gradient)
        jacobian = torch.autograd.functional.jacobian(
            lambda x: self.terms(x, lag)[0], q)
        torch.testing.assert_close(jacobian, torch.diag(expected))
        # This is the production objective, not a frozen Taylor surrogate.
        self.assertTrue(torch.autograd.gradcheck(
            lambda x: self.terms(x, lag)[0], (q,)))

    def test_full_current_semantic_chain_rule(self):
        angle = torch.tensor(.2, dtype=torch.double, requires_grad=True)
        s, u = .25 + .1 * angle, .4 + .2 * angle
        q = s + (1 - s) * u
        derivative = q - .5 * .5 - .125 * .4
        loss = torch.exp(-.3 * derivative) * (1 - s) * (1 - u)
        actual, = torch.autograd.grad(loss, angle)
        dq = (1 - u.detach()) * .1 + (1 - s.detach()) * .2
        expected = -torch.exp(-.3 * derivative.detach()) * (
            1 + .3 * (1 - q.detach())) * dq
        torch.testing.assert_close(actual, expected)
        # Detaching semantics would omit a nonzero term: it is a different loss.
        geometric_only = -torch.exp(-.3 * derivative.detach()) * (
            1 + .3 * (1 - q.detach())) * (1 - s.detach()) * .2
        self.assertGreater(float((actual - geometric_only).abs()), .01)

    def test_new_history_signature_rejects_legacy_and_roundtrips(self):
        new = FCOSFrSIoUForwardLoss()
        new._pending = [(('a.jpg', 0, 0), .6, [0.] * 5)]
        new.commit_history()
        restored = FCOSFrSIoUForwardLoss()
        restored.import_history(new.export_history())
        self.assertEqual(new.export_history(), restored.export_history())
        old = FCOSFrSIoUEngineeringLoss(history_gradient='values')
        self.assertEqual(old.signature()['schema'], 1)
        self.assertNotIn('modulation_scope', old.signature())
        for receiver, state in ((new, old.export_history()),
                                (old, new.export_history())):
            with self.assertRaises(ValueError):
                receiver.import_history(state)
        for kwargs in (dict(history_gradient='box_tangent'),
                       dict(modulation_scope='group')):
            with self.assertRaises(ValueError):
                FCOSFrSIoUForwardLoss(**kwargs)

    @unittest.skipUnless(torch.cuda.is_available(), 'Native IoU requires CUDA')
    def test_registered_loss_matches_per_box_formula_and_box_feature_gradients(self):
        y, x = torch.meshgrid(torch.linspace(.1, 1., 64, device='cuda'),
                              torch.linspace(.1, 1., 64, device='cuda'), indexing='ij')
        feature = torch.stack([x, y, x*y, x*x+2*y*y])[None].requires_grad_()
        pred = torch.tensor([[30.13, 28.27, 9., 6., .23],
                             [12.19, 11.31, 7., 5., -.17]],
                            device='cuda', requires_grad=True)
        gt = torch.tensor([[32.19, 30.31, 10., 8., .11]], device='cuda')
        target = gt.expand(2, -1)
        idx = torch.zeros(2, dtype=torch.long, device='cuda')
        weights = pred.new_tensor([1., .4])
        kw = dict(feature_maps=[feature], image_indices=idx, level_indices=idx,
                  feature_strides=[1], weight=weights, avg_factor=weights.sum())
        loss = build_loss(dict(type='FCOSFrSIoUForwardLoss'))
        loss.history[('a.jpg', 0, 0)] = [(.5, (0.,)*5), (.4, (0.,)*5)]
        loss.set_batch_context([dict(ori_filename='a.jpg')], [gt])
        actual = loss(pred, target, **kw)
        u = geometric_iou(pred, target)
        s = semantic_scores(pred, target, [feature], idx, idx, [1])[0]
        q = s + (1 - s) * u
        expected_each = torch.exp(-.3 * (q - .5*.5 - .125*.4)) * (1-s) * (1-u)
        expected = (weights * expected_each).sum() / weights.sum()
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
        ga = torch.autograd.grad(actual, (pred, feature), retain_graph=True)
        ge = torch.autograd.grad(expected, (pred, feature))
        for a, e in zip(ga, ge):
            torch.testing.assert_close(a, e, atol=2e-6, rtol=2e-5)
            self.assertTrue(torch.isfinite(a).all())
        # Second prediction is disjoint: its semantic path still moves the box.
        self.assertEqual(float(u[1]), 0.)
        self.assertGreater(float(ga[0][1, :2].abs().sum()), 0.)
        self.assertGreater(float(ga[0][1, 4].abs()), 0.)
        self.assertEqual(loss.last_diagnostics['history_hit_groups'], 1)
        loss.commit_history()
        records = loss.history[('a.jpg', 0, 0)]
        self.assertAlmostEqual(records[0][0], float((weights*q).sum()/weights.sum()), places=6)
        self.assertEqual(records[0][1], (0.,)*5)
        self.assertEqual(records[1][0], .5)
        loss.clear_batch_context()
        # Lambda zero agrees in both value and gradient with the existing SIoU.
        zero = FCOSFrSIoUForwardLoss(modulation_lambda=0.)
        zero.set_batch_context([dict(ori_filename='a.jpg')], [gt])
        a = zero(pred, target, **kw)
        b = SemanticIoULoss()(pred, target, **kw)
        torch.testing.assert_close(a, b)
        for aa, bb in zip(torch.autograd.grad(a, (pred, feature)),
                          torch.autograd.grad(b, (pred, feature))):
            torch.testing.assert_close(aa, bb, atol=2e-6, rtol=2e-5)

    @unittest.skipUnless(torch.cuda.is_available(), 'FCOS head requires CUDA')
    def test_config_head_optimizer_history_restore_and_empty_batch(self):
        cfg = mmcv.Config.fromfile('configs/dior_r/server_trainval/paper_frsiou_forward.py')
        head = build_head(cfg.model.bbox_head).cuda()
        self.assertIsInstance(head, FractionalRotatedFCOSHead)
        self.assertIsInstance(head.loss_bbox, FCOSFrSIoUForwardLoss)
        self.assertIsNone(cfg.resume_from)
        self.assertIsNone(cfg.load_from)
        head.init_weights()
        optimizer = torch.optim.SGD(head.parameters(), lr=.001)
        gt = [torch.tensor([[64., 64., 48., 24., .2]], device='cuda')]
        labels = [torch.tensor([1], device='cuda')]
        meta = [dict(ori_filename='a.jpg', img_shape=(128,128,3),
                     pad_shape=(128,128,3), scale_factor=1.)]
        for step in range(3):
            features = [torch.randn(1,256,k,k,device='cuda',requires_grad=True)
                        for k in (16,8,4,2,1)]
            optimizer.zero_grad()
            losses = head.loss(*head(features), gt, labels, meta, semantic_features=features)
            total = sum(losses.values())
            total.backward()
            self.assertTrue(torch.isfinite(total))
            self.assertGreater(float(head.conv_angle.weight.grad.abs().sum()), 0.)
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in head.parameters()
                                if p.grad is not None))
            optimizer.step()
            if step:
                self.assertGreater(head.loss_bbox.last_diagnostics['history_hit_fraction'], 0.)
            self.assertEqual(head.loss_bbox.last_diagnostics['transported_history_gradient_l2'], 0.)
            head.loss_bbox.commit_history()
            self.assertIsNone(head.loss_bbox._context)
        saved = copy.deepcopy(head.loss_bbox.export_history())
        restored = build_head(cfg.model.bbox_head).cuda()
        restored.load_state_dict(head.state_dict())
        restored.loss_bbox.import_history(saved)
        with torch.no_grad():
            a = head.loss(*head(features), gt, labels, meta, semantic_features=features)
            b = restored.loss(*restored(features), gt, labels, meta, semantic_features=features)
        for name in a:
            torch.testing.assert_close(a[name], b[name])
        self.assertEqual(saved, head.loss_bbox.export_history())
        empty = head.loss(*head(features), [gt[0][:0]], [labels[0][:0]], meta,
                          semantic_features=features)
        self.assertEqual(float(empty['loss_bbox']), 0.)
        self.assertIsNone(head.loss_bbox._pending)


if __name__ == '__main__':
    unittest.main(verbosity=2)
