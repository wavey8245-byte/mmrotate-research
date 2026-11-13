"""Evaluate a completed one-epoch probe with its frozen config and checkpoint."""
import argparse
from datetime import datetime
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('rio_evaluation_helpers', ROOT / 'tools/run_dior_r_rio.py')
HELPERS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPERS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    folder = (ROOT / args.work_dir).resolve()
    output = folder / 'test_epoch_1'
    checkpoint = folder / 'epoch_1.pth'
    frozen = folder / 'snapshot/experiment.py'
    lock_path = ROOT / 'work_dirs/dior_r_trainval/server_queue.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        training = json.loads((folder / 'run.json').read_text())
        if training['status'] != 'completed_one_epoch':
            raise RuntimeError('The probe must complete its training-image check first')
        if HELPERS.hashes(training['source_sha256']) != training['source_sha256']:
            raise RuntimeError('Probe sources changed after training')
        if HELPERS.hashes([str(frozen)])[str(frozen)] != training['frozen_config_sha256']:
            raise RuntimeError('Frozen training configuration changed')
        if output.exists() and any(output.iterdir()):
            raise RuntimeError(f'Existing test output will not be overwritten: {output}')
        sources = [str(frozen), str(checkpoint), 'tools/test.py',
                   'tools/evaluate_dior_r_probe.py',
                   'mmrotate/models/dense_heads/rotated_fcos_head.py',
                   'mmrotate/models/losses/rotated_rio_loss.py',
                   'mmrotate/models/losses/rotated_iou_loss.py', 'mmrotate/datasets/dior.py']
        before = HELPERS.hashes(sources)
        output.mkdir(parents=True, exist_ok=True)
        (output / 'evaluator.py').write_bytes(Path(__file__).read_bytes())
        state = dict(status='starting', created_at=datetime.now().astimezone().isoformat(),
                     checkpoint=str(checkpoint), checkpoint_epoch=1, config=str(frozen),
                     test_images=11738, iou_thr=.5, use_07_metric=True,
                     source_sha256=before,
                     scope='Full test split at epoch 1; not the fixed epoch-12 final result.')
        HELPERS.DIRECTORY = output
        HELPERS.write_json(output / 'run.json', state)
        try:
            HELPERS.run_child([sys.executable, str(ROOT / 'tools/test.py'), str(frozen),
                               str(checkpoint), '--work-dir', str(output),
                               '--out', str(output / 'predictions.pkl'), '--eval', 'mAP'],
                              output / 'test.log', state, 'testing', lock.fileno())
            evaluations = list(output.glob('eval_*.json'))
            if len(evaluations) != 1:
                raise RuntimeError('Expected exactly one full test evaluation')
            metric = json.loads(evaluations[0].read_text())['metric']
            score = float(metric['mAP'])
            if not math.isfinite(score):
                raise RuntimeError('Nonfinite test AP')
            if HELPERS.hashes(sources) != before:
                raise RuntimeError('Evaluation inputs changed while testing')
            state.update(status='completed', finished_at=datetime.now().astimezone().isoformat(),
                         metric=metric, mAP_percent=100 * score,
                         evaluation_source=str(evaluations[0]))
            HELPERS.write_json(output / 'result.json', state)
            HELPERS.write_json(output / 'run.json', state)
            print(json.dumps(dict(mAP_percent=100 * score, result=str(output / 'result.json'))), flush=True)
        except Exception as error:
            state.update(status='failed', error=str(error),
                         finished_at=datetime.now().astimezone().isoformat())
            HELPERS.write_json(output / 'run.json', state)
            raise


if __name__ == '__main__':
    main()
