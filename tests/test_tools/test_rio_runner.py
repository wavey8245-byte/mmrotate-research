"""Exercise detached-worker phase transitions without a GPU or real weights."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('rio_runner', ROOT / 'tools/run_dior_r_rio.py')
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class TestRiORunner(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / 'output'
        self.output.mkdir()
        (self.root / 'tools').mkdir()
        self.train = self.root / 'tools/train.py'
        self.train.write_text(
            "from pathlib import Path\n"
            "Path('output/epoch_12.pth').write_bytes(b'fake')\n")
        self.test = self.root / 'tools/test.py'
        self.test.write_text(
            "import json,sys\nfrom pathlib import Path\n"
            "assert Path(sys.argv[2]).name == 'epoch_12.pth'\n"
            "out=Path(sys.argv[sys.argv.index('--work-dir')+1])\n"
            "out.mkdir()\n"
            "(out/'eval_fake.json').write_text(json.dumps({'metric':{'mAP':0.5}}))\n")
        self.config = self.output / 'snapshot.py'
        self.config.write_text('seed=42\n')
        patches = patch.multiple(RUNNER, ROOT=self.root, DIRECTORY=self.output,
                                 FROZEN_CONFIG=self.config,
                                 SOURCES=('tools/train.py', 'tools/test.py'))
        patches.start()
        self.addCleanup(patches.stop)
        self.lock = (self.root / 'lock').open('a')
        self.addCleanup(self.lock.close)
        self.reset_state()

    def reset_state(self):
        RUNNER.write_json(self.output / 'run.json', dict(
            source_sha256=RUNNER.hashes(RUNNER.SOURCES),
            frozen_config_sha256=RUNNER.hashes([str(self.config)])[str(self.config)]))

    def state(self):
        return json.loads((self.output / 'run.json').read_text())

    def test_train_then_fixed_checkpoint_test_and_report(self):
        RUNNER.worker(self.lock.fileno())
        state = self.state()
        self.assertEqual(state['status'], 'completed')
        self.assertEqual(state['training_returncode'], 0)
        self.assertEqual(state['testing_returncode'], 0)
        self.assertEqual(state['result']['epoch'], 12)
        self.assertEqual(state['result']['test_mAP'], .5)
        self.assertTrue((self.output / 'results.md').is_file())

    def test_training_failure_prevents_test(self):
        self.train.write_text('raise SystemExit(7)\n')
        self.reset_state()
        with self.assertRaisesRegex(RuntimeError, 'training exited with code 7'):
            RUNNER.worker(self.lock.fileno())
        self.assertEqual(self.state()['status'], 'failed')
        self.assertNotIn('testing_pid', self.state())
        self.assertFalse((self.output / 'results.json').exists())

    def test_missing_checkpoint_prevents_test(self):
        self.train.write_text('pass\n')
        self.reset_state()
        with self.assertRaisesRegex(RuntimeError, 'without epoch_12'):
            RUNNER.worker(self.lock.fileno())
        self.assertEqual(self.state()['status'], 'failed')
        self.assertNotIn('testing_pid', self.state())

    def test_modified_source_prevents_training(self):
        self.train.write_text('pass\n')
        with self.assertRaisesRegex(RuntimeError, 'Source changed'):
            RUNNER.worker(self.lock.fileno())
        self.assertEqual(self.state()['status'], 'failed')
        self.assertNotIn('training_pid', self.state())

    def test_nonfinite_test_result_fails_report(self):
        self.test.write_text(self.test.read_text().replace("'mAP':0.5", "'mAP':float('nan')"))
        self.reset_state()
        with self.assertRaisesRegex(RuntimeError, 'Nonfinite final test'):
            RUNNER.worker(self.lock.fileno())
        self.assertEqual(self.state()['status'], 'failed')
        self.assertFalse((self.output / 'results.json').exists())

    def test_health_guard_rejects_persistent_explosion(self):
        path = self.output / '20261016_110000.log.json'
        path.write_text('\n'.join(json.dumps(dict(
            mode='train', grad_norm=value, loss=8.))
            for value in (100., 100000., 120000., 90000.)))
        with self.assertRaisesRegex(RuntimeError, 'Sustained gradient explosion'):
            RUNNER.check_training_health()

    def test_health_guard_allows_one_transient_spike_and_partial_log(self):
        path = self.output / '20261016_110000.log.json'
        path.write_text('\n'.join(json.dumps(dict(
            mode='train', grad_norm=value, loss=8.))
            for value in (100., 100000., 120., 90.)) + '\n{"mode":')
        RUNNER.check_training_health()

    def test_health_guard_rejects_nonfinite_losses(self):
        path = self.output / '20261016_110000.log.json'
        path.write_text(json.dumps(dict(mode='train', loss=float('nan'), grad_norm=1.)))
        with self.assertRaisesRegex(RuntimeError, 'Nonfinite training metric'):
            RUNNER.check_training_health()


if __name__ == '__main__':
    unittest.main()
