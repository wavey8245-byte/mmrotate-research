"""References independent of the implementation: printed expansion and integral."""
import math
import unittest

import mmcv
import numpy as np
import torch

from mmrotate.models.builder import build_head
from mmrotate.models.losses.paper_sgkld_loss import PaperSGKLDLoss, sgkld_divergence


def printed_reference(p, t):
    """Literal Eqs. (3), (11), (12a-e); retains signed mixed moments."""
    dx, dy = p[0] - t[0], p[1] - t[1]
    mx = t[4].cos() * dx + t[4].sin() * dy
    my = -t[4].sin() * dx + t[4].cos() * dy
    ux, uy = mx / t[2], my / t[3]
    wp, hp, wt, ht = p[2], p[3], t[2], t[3]
    c, s = (p[4] - t[4]).cos(), (p[4] - t[4]).sin()
    k = math.sqrt(2) * math.pi / math.gamma(.25)**2
    a = (.25 + 24*k*uy**2 + 16*uy**4) * ht**4
    b = (12*k*ux*uy + 16*ux*uy**3) * wt*ht**3
    cc = (k*k + 4*k*(uy**2 + ux**2) + 16*ux**2*uy**2) * wt**2*ht**2
    dd = (12*k*ux*uy + 16*ux**3*uy) * wt**3*ht
    e = (.25 + 24*k*ux**2 + 16*ux**4) * wt**4
    return ((wp/wt).log() + (hp/ht).log() - .5
            + a*(s**4/wp**4+c**4/hp**4)
            - 4*b*(c*s**3/wp**4-c**3*s/hp**4)
            + 6*cc*(1/wp**4+1/hp**4)*c**2*s**2
            - 4*dd*(c**3*s/wp**4-c*s**3/hp**4)
            + e*(c**4/wp**4+s**4/hp**4))


def integral_reference(p, t):
    """Deterministic quadrature of Eqs. (9),(10), no expanded moments."""
    nodes, weights = np.polynomial.legendre.leggauss(160)
    nodes, weights = nodes*3.5, weights*3.5
    x, y = np.meshgrid(nodes, nodes, indexing='ij')
    density = (2/math.gamma(.25))**2 * np.exp(-x**4-y**4)
    mass = density * weights[:, None] * weights[None, :]
    dx, dy = p[0]-t[0], p[1]-t[1]
    mx = np.cos(t[4])*dx + np.sin(t[4])*dy
    my = -np.sin(t[4])*dx + np.cos(t[4])*dy
    x, y = t[2]/2*x-mx, t[3]/2*y-my
    c, s = np.cos(p[4]-t[4]), np.sin(p[4]-t[4])
    z = ((c*x-s*y)/(p[2]/2))**4 + ((s*x+c*y)/(p[3]/2))**4
    return np.log(p[2]*p[3]/(t[2]*t[3]))-.5+(mass*z).sum()


class TestSGKLD(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def pair(self):
        return (torch.tensor([[1.3, -.7, 8., 5., .37]], dtype=torch.double),
                torch.tensor([[.2, .4, 9., 6., -.19]], dtype=torch.double))

    def test_printed_values_gradients_and_integral(self):
        rng = np.random.RandomState(42)
        for _ in range(20):
            p, t = self.pair()
            p[:, :2] = torch.from_numpy(rng.uniform(-2, 2, (1, 2)))
            p[:, 4] = float(rng.uniform(-1.5, 1.5))
            p.requires_grad_()
            actual = sgkld_divergence(p, t).sum()
            expected = printed_reference(p[0], t[0])
            torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
            g, = torch.autograd.grad(actual, p, retain_graph=True)
            ref, = torch.autograd.grad(expected, p)
            torch.testing.assert_close(g, ref, rtol=1e-10, atol=1e-10)
            self.assertAlmostEqual(float(actual), integral_reference(
                p.detach().numpy()[0], t.numpy()[0]), places=8)

    def test_gradcheck_and_equation17(self):
        p, t = self.pair()
        p.requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(lambda x: PaperSGKLDLoss()(x, t), (p,)))
        d = printed_reference(p[0], t[0])
        torch.testing.assert_close(PaperSGKLDLoss()(p, t), 1-1/(1+torch.log1p(d)))
        self.assertNotAlmostEqual(float(sgkld_divergence(p, t)),
                                  float(sgkld_divergence(t, p)))

    def test_identity_square_angle_and_representation(self):
        t = torch.tensor([[0., 0., 8., 8., 0.]], dtype=torch.double)
        p = t.clone().requires_grad_()
        value = PaperSGKLDLoss()(p, t)
        self.assertLess(abs(float(value)), 1e-12)
        value.backward()
        self.assertTrue(torch.isfinite(p.grad).all())
        p = t.clone(); p[:, 4] = .2; p.requires_grad_()
        value = PaperSGKLDLoss()(p, t)
        self.assertGreater(float(value), 0)
        value.backward()
        self.assertGreater(float(p.grad[0, 4]), 0)
        # At 45 degrees the square has a stationary angle (paper Sec. IV-B).
        p = t.clone(); p[:, 4] = math.pi/4; p.requires_grad_()
        PaperSGKLDLoss()(p, t).backward()
        self.assertLess(abs(float(p.grad[0, 4])), 1e-12)
        p, t = self.pair()
        expected = PaperSGKLDLoss()(p, t)
        pp = p.clone(); pp[:, 2:4] = pp[:, [3, 2]]; pp[:, 4] += math.pi/2
        torch.testing.assert_close(PaperSGKLDLoss()(pp, t), expected)

    def test_common_translation_rotation_scale(self):
        p, t = self.pair(); expected = PaperSGKLDLoss()(p, t)
        for scale in (1e-3, 1., 1e3):
            pp, tt = p.clone(), t.clone()
            for box in (pp, tt):
                box[:, :4] *= scale
                x, y = box[:, 0].clone(), box[:, 1].clone()
                box[:, 0] = math.cos(.7)*x-math.sin(.7)*y+100
                box[:, 1] = math.sin(.7)*x+math.cos(.7)*y-90
                box[:, 4] += .7
            torch.testing.assert_close(PaperSGKLDLoss()(pp, tt), expected,
                                       rtol=1e-9, atol=1e-9)

    def test_weighting_empty_zero_size_and_ignored(self):
        p, t = self.pair(); p = p.repeat(2, 1); t = t.repeat(2, 1)
        weight = torch.tensor([.3, .7], dtype=torch.double)
        f = PaperSGKLDLoss()
        vals = f(p, t, reduction_override='none')
        torch.testing.assert_close(f(p, t, weight, avg_factor=1), (vals*weight).sum())
        self.assertEqual(float(f(p[:0], t[:0])), 0)
        p[1] = float('nan')
        self.assertTrue(torch.isfinite(f(p, t, torch.tensor([1., 0.]))))
        p, t = self.pair(); p[:, 2] = 0; p.requires_grad_()
        f(p, t).backward()
        self.assertTrue(torch.isfinite(p.grad).all())

    def test_real_fcos_cpu_cuda_and_config(self):
        cfg = mmcv.Config.fromfile('configs/dior_r/server_trainval/paper_sgkld.py')
        base = mmcv.Config.fromfile('configs/dior_r/server_trainval/riou.py')
        for key in ('data', 'optimizer', 'lr_config', 'runner', 'evaluation'):
            self.assertEqual(cfg[key], base[key])
        for device in (['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']):
            head = build_head(cfg.model.bbox_head).to(device); head.init_weights()
            features = [torch.randn(1, 256, k, k, device=device) for k in (16, 8, 4, 2, 1)]
            gt = [torch.tensor([[64., 64., 48., 24., .2]], device=device)]
            labels = [torch.tensor([1], device=device)]
            meta = [dict(img_shape=(128, 128, 3), pad_shape=(128, 128, 3), scale_factor=1.)]
            losses = head.loss(*head(features), gt, labels, meta)
            sum(v for k, v in losses.items() if 'loss' in k).backward()
            self.assertGreater(float(head.conv_angle.weight.grad.abs().sum()), 0)
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in head.parameters()
                                if p.grad is not None))
            empty = head.loss(*head(features), [gt[0][:0]], [labels[0][:0]], meta)
            self.assertEqual(float(empty['loss_bbox']), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
