"""Reject the finite-but-regressing localization seen in the failed transfers."""
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

from mmrotate.core.rio_training_health import (
    RiOTrainingHealthHook, learning_failure)


class FakeDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scalar = torch.nn.Parameter(torch.tensor(1.))
        self.bbox_head = torch.nn.Identity()

    def forward(self, **kwargs):
        self.bbox_head(([self.scalar], [self.scalar], [self.scalar], [self.scalar]))
        return [[[]]]


class FakeDataset:
    def __init__(self, score):
        self.score = score
        self.data_infos = list(range(8))

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, index):
        return {}

    def evaluate(self, results, **kwargs):
        return dict(mAP=self.score)


class TestRiOEpochHealth(unittest.TestCase):
    def test_policy_rejects_failed_epoch_and_large_later_regression(self):
        self.assertIsNotNone(learning_failure(.007916, 0))
        self.assertIsNone(learning_failure(.1, 0))
        self.assertIsNotNone(learning_failure(.1, .3))
        self.assertIsNotNone(learning_failure(float('nan'), 0))
        self.assertIsNone(learning_failure(.2, .3))

    def test_test_split_is_forbidden(self):
        with self.assertRaisesRegex(ValueError, 'test split'):
            RiOTrainingHealthHook(dict(ann_file='ImageSets/Main/test.txt'))

    def test_diagnostic_snapshot_uses_completed_iteration_without_optimizer(self):
        hook = RiOTrainingHealthHook(dict(ann_file='train.txt'),
                                     diagnostic_iterations=[500])
        runner = types.SimpleNamespace(model=FakeDetector(), iter=498, work_dir='/tmp')
        with patch('mmcv.runner.save_checkpoint') as save:
            hook.after_train_iter(runner)
            save.assert_not_called()
            runner.iter = 499
            hook.after_train_iter(runner)
        self.assertTrue(save.call_args[0][1].endswith('diagnostic_iter_500.pth'))
        self.assertEqual(save.call_args[1]['meta']['iter'], 500)
        self.assertTrue(save.call_args[1]['meta']['diagnostic_only'])
        self.assertNotIn('optimizer', save.call_args[1])

    def test_epoch_check_restores_mode_and_writes_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = FakeDetector().train()
            runner = types.SimpleNamespace(
                model=model, epoch=0, work_dir=temporary,
                logger=types.SimpleNamespace(info=lambda *args: None))
            hook = RiOTrainingHealthHook(dict(ann_file='train.txt'), images=4)
            with patch('mmrotate.datasets.build_dataset', return_value=FakeDataset(.007916)), \
                    patch('mmcv.parallel.collate', return_value={}), \
                    patch('mmcv.parallel.scatter', return_value=[{}]):
                with self.assertRaisesRegex(RuntimeError, 'learning safeguard'):
                    hook.after_train_epoch(runner)
            self.assertTrue(model.training)
            self.assertEqual(float(model.scalar), 1.)
            record = json.loads((Path(temporary) / 'training_health.json').read_text())[0]
            self.assertEqual(record['status'], 'failed')
            self.assertEqual(record['training_images'], 4)


if __name__ == '__main__':
    unittest.main()
