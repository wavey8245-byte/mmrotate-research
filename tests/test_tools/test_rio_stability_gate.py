"""Checks for finite-but-failed models that the original preflight missed."""
import importlib.util
from pathlib import Path
import unittest
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('rio_preflight', ROOT / 'tools/check_dior_r_rio.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TestRiOStabilityGate(unittest.TestCase):
    def test_warmup_preserves_parameter_group_learning_rate_ratios(self):
        optimizer = SimpleNamespace(param_groups=[{'lr': 1e-4}, {'lr': 1e-5}])
        initial = [1e-4, 1e-5]
        schedule = dict(warmup='linear', warmup_iters=500, warmup_ratio=1 / 3)
        for step, factor in ((0, 1 / 3), (250, 2 / 3), (500, 1.)):
            MODULE.warmup_learning_rates(optimizer, initial, step, schedule)
            for group, rate in zip(optimizer.param_groups, initial):
                self.assertAlmostEqual(group['lr'], rate * factor)

    def report(self):
        return dict(steps=1000, last200_grad_median=20., last200_grad_p95=100.,
                    angle_p95=1.5, positive_wh_fraction=1., train_image_mAP=.1,
                    regression_image_response=.02)

    def test_normal_learning_passes(self):
        self.assertEqual(MODULE.stability_errors(self.report()), [])

    def test_historical_numeric_only_pass_is_rejected(self):
        historical = self.report()
        historical.update(steps=600, last200_grad_median=10000,
                          last200_grad_p95=12000, angle_p95=1746173.)
        self.assertTrue(MODULE.stability_errors(historical))

    def test_positive_finite_boxes_do_not_hide_collapse(self):
        for change in (dict(train_image_mAP=0.), dict(regression_image_response=0.),
                       dict(last200_grad_median=float('inf')),
                       dict(train_image_mAP=float('nan')),
                       dict(regression_image_response=float('nan')),
                       dict(positive_wh_fraction=0.)):
            with self.subTest(change=change):
                report = self.report()
                report.update(change)
                self.assertTrue(MODULE.stability_errors(report))


if __name__ == '__main__':
    unittest.main()
