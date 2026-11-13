"""A weight-check queue must retain gate failures and reject invalid comparisons."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('rio_weight_queue',
    ROOT / 'tools/run_dior_r_rio_component_checks.py')
QUEUE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUEUE)


class TestWeightCheckQueue(unittest.TestCase):
    def run_fake_check(self, *, first_cls=1., last_step=1000, gate_status='passed'):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / 'output'
            folder.mkdir()
            (root / 'tools').mkdir()
            reference = root / 'reference.json'
            reference.write_text(json.dumps(dict(history=[dict(loss_cls=1., bbox_kld=2.)])))
            dependency = root / 'control.json'
            dependency.write_text(json.dumps(dict(status='completed', metric=dict(mAP=.3))))
            (folder / 'run.json').write_text(json.dumps(dict(source_sha256={'x':'same'}, results={})))
            result = dict(status=gate_status, train_image_mAP=.005 if gate_status=='failed' else .1,
                history=[dict(step=1, loss_cls=first_cls, bbox_kld=2.), dict(step=last_step)])
            program = (
                'import json,sys\nfrom pathlib import Path\n'
                'assert sys.argv[sys.argv.index("--steps")+1]=="1000"\n'
                'assert sys.argv[sys.argv.index("--config")+1]=="candidate.py"\n'
                'out=Path(sys.argv[sys.argv.index("--report")+1])\n'
                f'out.write_text(json.dumps({result!r}))\n'
                f'raise SystemExit({1 if gate_status=="failed" else 0})\n')
            (root / 'tools/check_dior_r_rio.py').write_text(program)
            with patch.multiple(QUEUE, ROOT=root, FOLDER=folder, REFERENCE=reference,
                    DEPENDENCY=dependency, ARMS={'candidate':'candidate.py'},
                    EXPECTED_WEIGHTS={'candidate':(.5,5.)}), \
                    patch.object(QUEUE, 'validate', return_value={'x':'same'}):
                error = None
                try:
                    QUEUE.worker()
                except RuntimeError as exception:
                    error = str(exception)
            return json.loads((folder / 'run.json').read_text()), error

    def test_completed_learning_gate_failure_is_retained_without_relabelling(self):
        state, error = self.run_fake_check(gate_status='failed')
        self.assertIsNone(error)
        self.assertEqual(state['status'], 'completed')
        self.assertEqual(state['results']['candidate']['status'], 'failed')
        self.assertEqual(state['results']['candidate']['returncode'], 1)

    def test_changed_first_batch_invalidates_the_comparison(self):
        state, error = self.run_fake_check(first_cls=1.2)
        self.assertEqual(state['status'], 'failed')
        self.assertIn('unmatched first-batch loss_cls', error)
        self.assertNotIn('candidate', state['results'])

    def test_early_stop_is_not_reported_as_a_completed_1000_step_check(self):
        state, error = self.run_fake_check(last_step=200)
        self.assertEqual(state['status'], 'failed')
        self.assertIn('stopped before step 1000', error)
        self.assertNotIn('candidate', state['results'])


if __name__ == '__main__':
    unittest.main()
