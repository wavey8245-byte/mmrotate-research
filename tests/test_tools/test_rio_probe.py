"""The normalized probe must remain one epoch, training-only, and fail closed."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('rio_probe', ROOT / 'tools/run_dior_r_rio_probe.py')
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


class TestRiOProbe(unittest.TestCase):
    def test_adamw_probe_uses_a_matching_optimizer_reference(self):
        with patch.multiple(PROBE,
                CONFIG='configs/dior_r/server_trainval/rio_adamw.py',
                REFERENCE_CONFIG='configs/dior_r/server_trainval/riou_adamw.py'):
            cfg = PROBE.configuration()
        self.assertEqual(cfg.optimizer.type, 'AdamW')
        self.assertEqual(cfg.optimizer_config.grad_clip.max_norm, .1)
        self.assertEqual(cfg.runner.max_epochs, 1)
        self.assertTrue(cfg.model.bbox_head.scale_angle)

    def test_configuration_is_one_epoch_and_training_only(self):
        cfg = PROBE.configuration()
        self.assertEqual(cfg.runner.max_epochs, 1)
        self.assertTrue(cfg.model.bbox_head.scale_angle)
        self.assertFalse(cfg.model.bbox_head.get('bounded_angle', False))
        hook = cfg.custom_hooks[0]
        self.assertEqual(hook['images'], 128)
        self.assertTrue(hook['dataset_cfg']['ann_file'].endswith('/train.txt'))
        self.assertEqual(hook['priority'], 'LOW')

    def run_worker(self, healthy):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / 'output'
            output.mkdir()
            (root / 'tools').mkdir()
            config = output / 'frozen.py'
            config.write_text('runner=dict(max_epochs=1)\n')
            source = root / 'tools/train.py'
            source.write_text(
                "import json,sys\nfrom pathlib import Path\n"
                "assert '--no-validate' in sys.argv\n"
                "Path('output/epoch_1.pth').write_bytes(b'checkpoint')\n"
                f"Path('output/training_health.json').write_text(json.dumps(["
                f"{{'status':{'passed' if healthy else 'failed'!r},"
                f"'training_image_mAP':{.1 if healthy else .0079}}}]))\n"
                f"raise SystemExit({0 if healthy else 1})\n")
            with patch.multiple(PROBE, ROOT=root, DIRECTORY=output, FROZEN=config,
                                SOURCES=('tools/train.py',)), \
                    patch.multiple(PROBE.HELPERS, ROOT=root, DIRECTORY=output):
                PROBE.HELPERS.write_json(output / 'run.json', dict(
                    source_sha256=PROBE.HELPERS.hashes(PROBE.SOURCES),
                    frozen_config_sha256=PROBE.HELPERS.hashes([str(config)])[str(config)],
                    automatic_continuation=False))
                with (root / 'lock').open('a') as lock:
                    if healthy:
                        PROBE.worker(lock.fileno())
                    else:
                        with self.assertRaisesRegex(RuntimeError, 'training exited'):
                            PROBE.worker(lock.fileno())
                state = json.loads((output / 'run.json').read_text())
                self.assertTrue((output / 'epoch_1.pth').is_file())
                self.assertFalse((output / 'test.log').exists())
                return state

    def test_success_does_not_test_or_continue(self):
        state = self.run_worker(True)
        self.assertEqual(state['status'], 'completed_one_epoch')
        self.assertFalse(state['automatic_continuation'])

    def test_failed_learning_keeps_checkpoint_and_failure_report(self):
        state = self.run_worker(False)
        self.assertEqual(state['status'], 'failed')
        self.assertEqual(state['training_health']['status'], 'failed')
        self.assertFalse(state['automatic_continuation'])


if __name__ == '__main__':
    unittest.main()
