"""Fresh, single-process DCAGPL twelve-epoch run, followed by full test."""
import argparse
import copy
from datetime import datetime
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'work_dirs/dior_r_trainval/fcos_paper_dcagpl_fresh12_20261025'
FROZEN = DIRECTORY / 'snapshot/experiment.py'
PREVIOUS = ROOT / 'work_dirs/dior_r_trainval/fcos_paper_dcagpl_probe1_20261020'
OLD_RESULT = ROOT / 'work_dirs/dior_r_trainval/fcos_paper_dcagpl_resume12_20261021/results.json'
BASELINE = ROOT / 'work_dirs/dior_r_trainval/fcos_riou_bs2_800/test_final/eval_20261006_092447.json'
spec = importlib.util.spec_from_file_location('dcagpl_fresh_probe', ROOT / 'tools/run_paper_dcagpl_probe.py')
PROBE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(PROBE)
H = PROBE.HELPERS
H.DIRECTORY = DIRECTORY


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def hashes(paths):
    return {str(path): digest(ROOT / path) for path in paths}


def configuration():
    import mmcv
    previous = json.loads((PREVIOUS / 'run.json').read_text())
    if previous['status'] != 'completed_one_epoch':
        raise RuntimeError('Original DCAGPL epoch is not verified')
    if hashes(previous['source_sha256']) != previous['source_sha256']:
        raise RuntimeError('Sources differ from the original DCAGPL experiment')
    old_config = PREVIOUS / 'snapshot/experiment.py'
    if digest(old_config) != previous['frozen_config_sha256']:
        raise RuntimeError('Historical frozen configuration changed')
    cfg = PROBE.configuration()
    old = mmcv.Config.fromfile(str(old_config))
    a, b = copy.deepcopy(cfg.to_dict()), copy.deepcopy(old.to_dict())
    a.pop('work_dir', None)
    b.pop('work_dir', None)
    if a != b:
        raise RuntimeError('Reconstructed configuration differs from historical epoch one')
    cfg.runner.max_epochs = 12
    cfg.work_dir = str(DIRECTORY)
    cfg.load_from = None
    cfg.resume_from = None
    cfg.auto_resume = False
    if cfg.seed != 42 or cfg.data.samples_per_gpu != 2 or cfg.data.workers_per_gpu != 2:
        raise RuntimeError('Unexpected seed/batch/loader configuration')
    split_root = ROOT / 'data/DIOR-R/ImageSets/Main'
    splits = {key: set((split_root / (key + '.txt')).read_text().split())
              for key in ('train', 'val', 'test')}
    if [len(splits[key]) for key in ('train', 'val', 'test')] != [5862, 5863, 11738]:
        raise RuntimeError('Unexpected DIOR split sizes')
    if splits['train'] & splits['val'] or (splits['train'] | splits['val']) & splits['test']:
        raise RuntimeError('Dataset splits overlap')
    sources = list(dict.fromkeys(list(previous['source_sha256']) + [
        'tools/test.py', 'tools/run_paper_dcagpl_fresh12_20261025.py']))
    proof = dict(original_sources_unchanged=True, matches_original_epoch1_config=True,
                 original_config=str(old_config), original_config_sha256=digest(old_config),
                 historical_result=str(OLD_RESULT), historical_test_mAP=json.loads(OLD_RESULT.read_text())['test_mAP'],
                 fresh_detector=True, initialization='ImageNet-pretrained backbone; fresh detector and optimizer',
                 load_from=None, resume_from=None, auto_resume=False, seed=42,
                 trainval_images=11725, test_images=11738, dataset_splits_disjoint=True,
                 batch_size=2, workers_per_gpu=2, max_epochs=12, planned_training_processes=1,
                 changed_config_fields=['runner.max_epochs: 1 -> 12', 'work_dir', 'auto_resume: False explicitly'],
                 final_test=True, intermediate_test=False)
    return cfg, sources, proof


def guard(state):
    if hashes(state['source_sha256']) != state['source_sha256']:
        raise RuntimeError('Experiment sources changed during execution')
    if digest(FROZEN) != state['frozen_config_sha256']:
        raise RuntimeError('Frozen configuration changed')
    if hashes(state['reference_sha256']) != state['reference_sha256']:
        raise RuntimeError('Historical comparison records changed')


def worker(lock_fd):
    os.fstat(lock_fd)
    state = json.loads((DIRECTORY / 'run.json').read_text())
    state.update(worker_pid=os.getpid(), started_at=H.now())
    try:
        guard(state)
        H.run_child([sys.executable, str(ROOT / 'tools/train.py'), str(FROZEN),
                     '--seed', '42', '--no-validate'],
                    DIRECTORY / 'train.log', state, 'training', lock_fd)
        final = DIRECTORY / 'epoch_12.pth'
        health = json.loads((DIRECTORY / 'training_health.json').read_text())
        if not final.is_file() or [item['epoch'] for item in health] != list(range(1, 13)):
            raise RuntimeError('Expected twelve consecutive epoch checks and epoch_12.pth')
        if any(item['status'] != 'passed' for item in health):
            raise RuntimeError('Training-image health check failed')
        import torch
        checkpoint = torch.load(str(final), map_location='cpu')
        if checkpoint['meta'].get('epoch') != 12 or checkpoint['meta'].get('iter') != 70356:
            raise RuntimeError('Unexpected final native epoch/iteration')
        if not checkpoint.get('optimizer', {}).get('state'):
            raise RuntimeError('Missing optimizer state in final checkpoint')
        if any(not torch.isfinite(t).all() for t in checkpoint['state_dict'].values()):
            raise RuntimeError('Nonfinite final model weights')
        del checkpoint
        guard(state)
        state.update(training_completed_at=H.now(), checkpoint=str(final),
                     checkpoint_sha256=digest(final), training_health=health)
        H.run_child([sys.executable, str(ROOT / 'tools/test.py'), str(FROZEN), str(final),
                     '--work-dir', str(DIRECTORY / 'test_final'),
                     '--out', str(DIRECTORY / 'test_final_predictions.pkl'),
                     '--eval', 'mAP', '--eval-options', 'iou_thr=0.5', 'use_07_metric=True'],
                    DIRECTORY / 'test.log', state, 'testing', lock_fd)
        files = list((DIRECTORY / 'test_final').glob('eval_*.json'))
        if len(files) != 1:
            raise RuntimeError('Expected exactly one final full-test evaluation')
        value = float(json.loads(files[0].read_text())['metric']['mAP'])
        if not math.isfinite(value):
            raise RuntimeError('Nonfinite final mAP')
        guard(state)
        if digest(final) != state['checkpoint_sha256']:
            raise RuntimeError('Final checkpoint changed during testing')
        old = float(json.loads(OLD_RESULT.read_text())['test_mAP'])
        baseline = float(json.loads(BASELINE.read_text())['metric']['mAP'])
        result = dict(status='completed', method='DCAGPL', epoch=12, seed=42,
                      fresh_initialization=True, resumed_from=None,
                      training_processes=1, uninterrupted_training_returncode=0,
                      loss=state['loss'], trainval_images=11725, test_images=11738,
                      test_mAP=value, mAP_percent=100 * value, iou_thr=.5, use_07_metric=True,
                      checkpoint=str(final), checkpoint_sha256=state['checkpoint_sha256'],
                      evaluation_source=str(files[0]), source_sha256=state['source_sha256'],
                      previous_resumed_test_mAP=old, difference_vs_resumed_percentage_points=100*(value-old),
                      riou_test_mAP=baseline, difference_vs_riou_percentage_points=100*(value-baseline),
                      finished_at=H.now(), interpretation='One seed only; differences do not establish stable method superiority.')
        H.write_json(DIRECTORY / 'results.json', result)
        lines = ['# DCAGPL fresh 12-epoch run', '',
                 'One training process from ImageNet backbone initialization; seed=42; no resume.',
                 'DIOR-R train+val=11725, full test=11738; fixed epoch 12; VOC07 mAP50.', '',
                 '| Run | Test mAP50 (%) |', '|---|---:|',
                 f'| DCAGPL fresh | {100*value:.6f} |',
                 f'| DCAGPL previous resumed | {100*old:.6f} |',
                 f'| Rotated IoU baseline | {100*baseline:.6f} |', '',
                 f'Difference versus previous resumed DCAGPL: {100*(value-old):+.6f} pp.',
                 f'Difference versus baseline: {100*(value-baseline):+.6f} pp.', '',
                 'Single-seed observations; the 128 training-image health checks are not test results.', '']
        (DIRECTORY / 'results.md').write_text('\n'.join(lines), encoding='utf-8')
        state.update(status='completed', finished_at=H.now(), result=result)
        H.write_json(DIRECTORY / 'run.json', state)
    except Exception as error:
        state.update(status='failed', finished_at=H.now(), error=str(error))
        H.write_json(DIRECTORY / 'run.json', state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'start', 'worker', 'status'))
    parser.add_argument('--lock-fd', type=int)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.action == 'worker':
        if args.lock_fd is None:
            parser.error('An inherited experiment lock is required')
        worker(args.lock_fd)
        return
    if args.action == 'status':
        print((DIRECTORY / 'run.json').read_text())
        return
    cfg, sources, proof = configuration()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    (torch.ones(1, device='cuda') + 1).cpu()
    if args.action == 'check':
        print(json.dumps(proof, indent=2))
        return
    with (ROOT / 'work_dirs/dior_r_trainval/server_queue.lock').open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('Another experiment holds the trainval queue lock')
        if DIRECTORY.exists() and any(DIRECTORY.iterdir()):
            parser.error('Refusing to overwrite an existing run')
        FROZEN.parent.mkdir(parents=True, exist_ok=True)
        cfg.dump(str(FROZEN))
        for name in sources:
            path = DIRECTORY / 'snapshot/source' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((ROOT / name).read_bytes())
        H.write_json(DIRECTORY / 'preflight.json', proof)
        H.write_json(DIRECTORY / 'run.json', dict(
            status='starting', created_at=H.now(), target_epoch=12, seed=42,
            config='configs/dior_r/server_trainval/paper_dcagpl.py',
            frozen_config=str(FROZEN), frozen_config_sha256=digest(FROZEN),
            source_sha256=hashes(sources), reference_sha256=hashes([str(OLD_RESULT), str(BASELINE)]),
            initialization=proof['initialization'], load_from=None, resume_from=None, auto_resume=False,
            loss=dict(cfg.model.bbox_head.loss_bbox), training_processes_planned=1,
            final_test=True, intermediate_test=False, checkpoint_selection='fixed epoch 12',
            allocator='max_split_size_mb:128',
            health_guard='Original finite-metric/gradient guard; same 128 training images each epoch.'))
        env = os.environ.copy()
        env.update(PYTHONUNBUFFERED='1', CUDA_VISIBLE_DEVICES='0',
                   PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:128')
        env['LD_LIBRARY_PATH'] = '/usr/lib/wsl/lib' + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
        with (DIRECTORY / 'worker.log').open('w') as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                                        'worker', '--lock-fd', str(lock.fileno())],
                                       cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, pass_fds=(lock.fileno(),))
        (DIRECTORY / 'worker.pid').write_text(str(process.pid) + '\n')
        print(json.dumps(dict(worker_pid=process.pid, work_dir=str(DIRECTORY),
                              fresh=True, target_epoch=12, final_test=True)))


if __name__ == '__main__':
    main()
