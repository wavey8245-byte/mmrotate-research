"""Server queue orchestration checks using a fake trainer, without a GPU."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
FAKE_PYTHON = r'''#!/usr/bin/env python3
import json,os,sys
from pathlib import Path
args=sys.argv[1:]
with open(os.environ['TEST_CALL_LOG'],'a') as stream:
    stream.write(json.dumps(args)+'\n')
if args[0]=='tools/train.py':
    variant=Path(args[1]).stem
    if variant==os.environ.get('TEST_FAIL_VARIANT'):
        sys.exit(19)
    names={'riou':'fcos_riou_bs2_800',
           'smooth_l1':'fcos_smooth_l1_encoded_bs2_800',
           'wf':'fcos_wf_bs2_800'}
    folder=Path('work_dirs/dior_r_trainval')/names[variant]
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'epoch_12.pth').write_bytes(b'checkpoint')
elif args[0]=='tools/test.py':
    output=Path(args[args.index('--out')+1])
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_bytes(b'predictions')
'''


class TestTrainvalRunner(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        (self.directory / 'tools').mkdir()
        shutil.copy(ROOT / 'tools/run_dior_r_trainval.sh', self.directory / 'tools')
        binary = self.directory / 'bin'
        binary.mkdir()
        python = binary / 'python'
        python.write_text(FAKE_PYTHON)
        python.chmod(0o755)
        self.log = self.directory / 'calls.jsonl'
        self.env = dict(os.environ, PATH=str(binary) + ':' + os.environ['PATH'],
                        TEST_CALL_LOG=str(self.log))

    def run_queue(self, *args):
        return subprocess.run(['bash', 'tools/run_dior_r_trainval.sh', *args],
            cwd=self.directory, env=self.env, capture_output=True, text=True)

    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def test_serial_training_and_fixed_final_checkpoint(self):
        result = self.run_queue('all')
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        training = [Path(c[1]).stem for c in calls if c[0] == 'tools/train.py']
        testing = [Path(c[1]).stem for c in calls if c[0] == 'tools/test.py']
        self.assertEqual(training, ['riou', 'smooth_l1', 'wf'])
        self.assertEqual(testing, training)
        for i, call in enumerate(calls):
            if call[0] == 'tools/test.py':
                self.assertTrue(call[2].endswith('/epoch_12.pth'))
                self.assertEqual(calls[i - 1][0], 'tools/train.py')

    def test_failure_stops_remaining_gpu_jobs(self):
        self.env['TEST_FAIL_VARIANT'] = 'smooth_l1'
        result = self.run_queue('all')
        self.assertEqual(result.returncode, 19)
        gpu_calls = [c for c in self.calls() if c[0] in ('tools/train.py', 'tools/test.py')]
        self.assertEqual([(c[0], Path(c[1]).stem) for c in gpu_calls], [
            ('tools/train.py', 'riou'), ('tools/test.py', 'riou'),
            ('tools/train.py', 'smooth_l1')])

    def test_running_queue_lock_blocks_individual_training(self):
        directory = self.directory / 'work_dirs/dior_r_trainval'
        directory.mkdir(parents=True)
        with (directory / 'server_queue.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = self.run_queue('train', 'riou')
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.log.exists())

    def test_existing_weights_are_not_overwritten(self):
        directory = self.directory / 'work_dirs/dior_r_trainval/fcos_riou_bs2_800'
        directory.mkdir(parents=True)
        checkpoint = directory / 'epoch_1.pth'
        checkpoint.write_bytes(b'original')
        result = self.run_queue('train', 'riou')
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.log.exists())
        self.assertEqual(checkpoint.read_bytes(), b'original')


if __name__ == '__main__':
    unittest.main()
