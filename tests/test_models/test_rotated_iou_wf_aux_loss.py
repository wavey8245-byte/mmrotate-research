"""Regression checks for the observed angle trap and the auxiliary variant."""
import math
import unittest

import torch

from mmrotate.models.losses import (RotatedIoULoss, RotatedIoUWithWFAuxLoss,
                                    RotatedWFIoULoss)


@unittest.skipUnless(torch.cuda.is_available(), 'Official differentiable IoU requires CUDA')
class TestRotatedIoUWithWFAuxLoss(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.target = torch.tensor([[0., 0., 10., 2., 0.]], device='cuda')

    def test_wrong_angle_keeps_loss_and_recovery_gradient(self):
        pred = self.target.repeat(5, 1)
        pred[:, 4] = torch.tensor([25., 30., 40., 44., 45.], device='cuda') * math.pi / 180
        pred.requires_grad_()
        target = self.target.repeat(5, 1)
        original = RotatedWFIoULoss(reduction='none')(pred, target)
        self.assertLess(float(original[3]), 1e-5)
        candidate = RotatedIoUWithWFAuxLoss(reduction='none')(pred, target)
        self.assertTrue((candidate > 0).all())
        self.assertTrue((candidate[1:] > candidate[:-1]).all())
        derivative = torch.autograd.grad(candidate.sum(), pred)[0][:, 4]
        # Positive angle derivative means gradient descent reduces this error.
        self.assertTrue((derivative > 0).all())

    def test_zero_auxiliary_is_exact_official_baseline(self):
        pred = self.target.clone()
        pred[:, 0] += 1
        pred[:, 4] += .2
        pred.requires_grad_()
        base = RotatedIoULoss()(pred, self.target)
        candidate = RotatedIoUWithWFAuxLoss(wf_weight=0)(pred, self.target)
        torch.testing.assert_close(candidate, base, atol=0, rtol=0)
        expected = torch.autograd.grad(base, pred, retain_graph=True)[0]
        actual = torch.autograd.grad(candidate, pred)[0]
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_centerness_reduction_and_identity(self):
        pred = self.target.repeat(2, 1)
        target = self.target.repeat(2, 1)
        pred[:, 4] += .2
        weights = torch.tensor([1., .25], device='cuda')
        loss = RotatedIoUWithWFAuxLoss()
        raw = loss(pred, target, reduction_override='none')
        actual = loss(pred, target, weight=weights, avg_factor=weights.sum())
        torch.testing.assert_close(actual, (raw * weights).sum()/weights.sum())
        self.assertLess(float(loss(target, target)), 1e-5)


if __name__ == '__main__':
    unittest.main()
