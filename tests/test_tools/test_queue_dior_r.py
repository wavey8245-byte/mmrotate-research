"""Training-completion guards for the detached experiment queue."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('queue_dior_r', ROOT / 'tools/queue_dior_r.py')
QUEUE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUEUE)


class TestQueueCompletion(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.experiment = dict(work_dir=self.temporary.name, max_epochs=12)
        (self.directory / 'epoch_12.pth').write_bytes(b'x' * 1024)

    def log(self, rows):
        (self.directory / 'training.log.json').write_text(''.join(
            json.dumps(dict(mode='val', epoch=epoch, mAP=score)) + '\n'
            for epoch, score in rows))

    def test_final_checkpoint_requires_final_validation(self):
        self.log([(8, .5)])
        with self.assertRaisesRegex(RuntimeError, '最后一轮验证'):
            QUEUE.completed_training(self.experiment)

    def test_all_zero_validation_is_reported_as_training_failure(self):
        self.log([(epoch, 0) for epoch in range(1, 13)])
        with self.assertRaisesRegex(RuntimeError, '所有轮次验证 mAP 均为 0'):
            QUEUE.completed_training(self.experiment)

    def test_validation_selects_earlier_best_checkpoint(self):
        (self.directory / 'best_mAP_epoch_8.pth').write_bytes(b'x' * 1024)
        self.log([(8, .55), (12, .50)])
        result = QUEUE.completed_training(self.experiment)
        self.assertEqual(result['best_epoch'], 8)
        self.assertEqual(result['val_mAP'], .55)


if __name__ == '__main__':
    unittest.main()
