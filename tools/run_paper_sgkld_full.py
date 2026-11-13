"""Fresh SGKLD 12 epochs, fixed epoch-12 full test, then cached AP75.

Uses the existing trainval job lock. No test-based model selection/stopping.
"""
import argparse
import copy
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
CONFIG = 'configs/dior_r/server_trainval/paper_sgkld.py'
DIRECTORY = ROOT / 'work_dirs/dior_r_trainval/fcos_paper_sgkld_bs2_800_20261103'
FROZEN = DIRECTORY / 'snapshot/experiment.py'
PROBE = ROOT / 'work_dirs/dior_r_trainval/sgkld_probe_20261103/report.json'
BASELINE = ROOT / 'work_dirs/dior_r_trainval/fcos_riou_bs2_800/test_final/eval_20261006_092447.json'
SPEC = importlib.util.spec_from_file_location('sgkld_helpers', ROOT / 'tools/run_dior_r_rio.py')
H = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(H)
H.DIRECTORY = DIRECTORY


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(4*1024*1024), b''):
            sha.update(chunk)
    return sha.hexdigest()


def hashes(paths):
    return {str(p): digest(ROOT / p) for p in paths}


def configuration():
    import mmcv
    cfg = mmcv.Config.fromfile(str(ROOT / CONFIG))
    base = mmcv.Config.fromfile(str(ROOT / 'configs/dior_r/server_trainval/riou.py'))
    model = copy.deepcopy(cfg.model)
    model.bbox_head.loss_bbox = base.model.bbox_head.loss_bbox
    if model != base.model:
        raise RuntimeError('Model differs beyond bbox loss')
    for key in ('data', 'optimizer', 'optimizer_config', 'lr_config', 'runner',
                'checkpoint_config', 'seed', 'load_from', 'resume_from'):
        if cfg[key] != base[key]:
            raise RuntimeError('Training protocol differs: ' + key)
    expected = dict(type='PaperSGKLDLoss', min_size=1e-7, loss_weight=1.)
    if dict(cfg.model.bbox_head.loss_bbox) != expected:
        raise RuntimeError('Loss differs from verified SGKLD')
    proof = json.loads(PROBE.read_text())
    if (proof['status'] != 'completed_short_controls'
            or proof['arms']['sgkld']['steps_completed'] != 100
            or not proof['arms']['sgkld']['identical_augmented_batches']):
        raise RuntimeError('Verified 100-step matched controls required')
    if hashes(proof['source_sha256']) != proof['source_sha256']:
        raise RuntimeError('Source changed since matched controls')
    split_root = ROOT / 'data/DIOR-R/ImageSets/Main'
    splits = {s: set((split_root / (s+'.txt')).read_text().split())
              for s in ('train', 'val', 'test')}
    if [len(splits[s]) for s in splits] != [5862, 5863, 11738]:
        raise RuntimeError('Unexpected split sizes')
    if splits['train'] & splits['val'] or (splits['train'] | splits['val']) & splits['test']:
        raise RuntimeError('Dataset splits overlap')
    if cfg.runner.max_epochs != 12 or cfg.seed != 42:
        raise RuntimeError('Expected 12 epochs and seed 42')
    cfg.work_dir = str(DIRECTORY)
    cfg.load_from = cfg.resume_from = None
    cfg.auto_resume = False
    sources = list(dict.fromkeys(list(proof['source_sha256']) + [
        'tools/run_paper_sgkld_full.py', 'tools/run_dior_r_rio.py',
        'tools/train.py', 'tools/test.py',
        'mmrotate/models/losses/__init__.py',
        'configs/dior_r/rotated_fcos_r50_fpn_1x_dior_r_le90.py',
        'configs/dior_r/rotated_fcos_r50_fpn_1x_dior_r_le90_bs2.py',
        'configs/rotated_fcos/rotated_fcos_r50_fpn_1x_dota_le90.py',
        'configs/_base_/schedules/schedule_1x.py',
        'configs/_base_/default_runtime.py']))
    return cfg, sources


def guard(state):
    if hashes(state['source_sha256']) != state['source_sha256']:
        raise RuntimeError('Source changed during run')
    if digest(FROZEN) != state['frozen_config_sha256']:
        raise RuntimeError('Frozen config changed')


def evaluate_ap75():
    import mmcv
    from mmrotate.datasets import build_dataset
    cfg = mmcv.Config.fromfile(str(FROZEN))
    dataset = build_dataset(cfg.data.test)
    predictions = mmcv.load(str(DIRECTORY / 'test_final_predictions.pkl'))
    result = dataset.evaluate(predictions, iou_thr=.75, use_07_metric=True, nproc=2)
    H.write_json(DIRECTORY / 'ap75.json', dict(
        iou_thr=.75, use_07_metric=True, test_images=len(dataset),
        mAP=float(result['mAP']), source='same fixed-epoch-12 cached predictions'))


def worker(lock_fd):
    os.fstat(lock_fd)
    state = json.loads((DIRECTORY / 'run.json').read_text())
    state.update(worker_pid=os.getpid(), started_at=H.now())
    try:
        guard(state)
        H.run_child([sys.executable, str(ROOT / 'tools/train.py'), str(FROZEN),
                     '--seed', '42', '--no-validate'],
                    DIRECTORY / 'train.log', state, 'training', lock_fd)
        import torch
        final = DIRECTORY / 'epoch_12.pth'
        checkpoint = torch.load(str(final), map_location='cpu')
        if (checkpoint['meta'].get('epoch') != 12
                or checkpoint['meta'].get('iter') != 70356
                or not checkpoint.get('optimizer', {}).get('state')):
            raise RuntimeError('Final checkpoint epoch/iterations/optimizer mismatch')
        if any(not torch.isfinite(v).all() for v in checkpoint['state_dict'].values()):
            raise RuntimeError('Nonfinite final weights')
        del checkpoint
        guard(state)
        state.update(training_completed_at=H.now(), checkpoint=str(final),
                     checkpoint_sha256=digest(final))
        H.run_child([sys.executable, str(ROOT / 'tools/test.py'), str(FROZEN), str(final),
                     '--work-dir', str(DIRECTORY / 'test_final'),
                     '--out', str(DIRECTORY / 'test_final_predictions.pkl'),
                     '--eval', 'mAP', '--eval-options', 'iou_thr=0.5', 'use_07_metric=True'],
                    DIRECTORY / 'test.log', state, 'testing', lock_fd)
        evaluations = list((DIRECTORY / 'test_final').glob('eval_*.json'))
        if len(evaluations) != 1:
            raise RuntimeError('Expected one final AP50 evaluation')
        ap50 = float(json.loads(evaluations[0].read_text())['metric']['mAP'])
        baseline = float(json.loads(BASELINE.read_text())['metric']['mAP'])
        result = dict(status='completed_ap50', method='SGKLD', epoch=12, seed=42,
                      trainval_images=11725, test_images=11738, test_mAP=ap50,
                      mAP_percent=ap50*100, checkpoint=str(final),
                      checkpoint_sha256=state['checkpoint_sha256'],
                      baseline_test_mAP=baseline,
                      difference_percentage_points=100*(ap50-baseline),
                      evaluation_source=str(evaluations[0]), finished_at=H.now())
        if not math.isfinite(ap50):
            raise RuntimeError('Nonfinite AP50')
        H.write_json(DIRECTORY / 'results.json', result)
        H.run_child([sys.executable, str(Path(__file__).resolve()), 'ap75'],
                    DIRECTORY / 'ap75.log', state, 'evaluating_ap75', lock_fd)
        ap75 = float(json.loads((DIRECTORY / 'ap75.json').read_text())['mAP'])
        if not math.isfinite(ap75):
            raise RuntimeError('Nonfinite AP75')
        guard(state)
        if digest(final) != state['checkpoint_sha256']:
            raise RuntimeError('Checkpoint changed during evaluation')
        result.update(status='completed', test_AP75=ap75, finished_at=H.now())
        H.write_json(DIRECTORY / 'results.json', result)
        (DIRECTORY / 'results.md').write_text(
            '# SGKLD fixed epoch-12 full-test results\n\n'
            'Fresh ImageNet backbone initialization; DIOR-R trainval=11725, test=11738.\n'
            'R50-FPN FCOS, 800x800, batch=2, SGD lr=0.0025, seed=42.\n'
            'No intermediate test or test-based checkpoint selection.\n\n'
            f'Test AP50: {100*ap50:.4f}%; AP75: {100*ap75:.4f}%.\n'
            f'Historical Rotated IoU AP50: {100*baseline:.4f}%; '
            f'difference: {100*(ap50-baseline):+.4f} percentage points.\n\n'
            'Single-seed loss transfer experiment, not full original-paper reproduction.\n',
            encoding='utf-8')
        state.update(status='completed', finished_at=H.now(), result=result)
        H.write_json(DIRECTORY / 'run.json', state)
    except Exception as error:
        state.update(status='failed', finished_at=H.now(), error=str(error))
        H.write_json(DIRECTORY / 'run.json', state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'start', 'worker', 'status', 'ap75'))
    parser.add_argument('--lock-fd', type=int)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.action == 'status':
        print((DIRECTORY / 'run.json').read_text()); return
    if args.action == 'ap75':
        evaluate_ap75(); return
    if args.action == 'worker':
        if args.lock_fd is None:
            parser.error('Inherited experiment lock required')
        worker(args.lock_fd); return
    cfg, sources = configuration()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    if args.action == 'check':
        print(json.dumps(dict(config=CONFIG, loss=dict(cfg.model.bbox_head.loss_bbox),
                              epochs=12, fresh=True, batch=cfg.data.samples_per_gpu,
                              final_AP50_and_AP75=True))); return
    with (ROOT / 'work_dirs/dior_r_trainval/server_queue.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if DIRECTORY.exists() and any(DIRECTORY.iterdir()):
            parser.error('Refusing to overwrite existing run')
        FROZEN.parent.mkdir(parents=True, exist_ok=True)
        cfg.dump(str(FROZEN))
        for name in sources:
            dest = DIRECTORY / 'snapshot/source' / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((ROOT / name).read_bytes())
        H.write_json(DIRECTORY / 'run.json', dict(
            status='starting', created_at=H.now(), target_epoch=12, seed=42,
            config=CONFIG, frozen_config=str(FROZEN), frozen_config_sha256=digest(FROZEN),
            source_sha256=hashes(sources), loss=dict(cfg.model.bbox_head.loss_bbox),
            trainval_images=11725, test_images=11738, fresh_initialization=True,
            resume_from=None, load_from=None, auto_resume=False,
            intermediate_test=False, final_test=True, final_AP75=True,
            checkpoint_selection='fixed epoch 12', baseline_source=str(BASELINE)))
        env = os.environ.copy()
        env.update(PYTHONUNBUFFERED='1', CUDA_VISIBLE_DEVICES='0',
                   PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:128')
        env['LD_LIBRARY_PATH'] = '/usr/lib/wsl/lib' + (
            ':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
        with (DIRECTORY / 'worker.log').open('w') as log:
            process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), 'worker',
                 '--lock-fd', str(lock.fileno())], cwd=ROOT, env=env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, pass_fds=(lock.fileno(),))
        (DIRECTORY / 'worker.pid').write_text(str(process.pid)+'\n')
        print(json.dumps(dict(worker_pid=process.pid, work_dir=str(DIRECTORY),
                              epochs=12, final_test=True, AP75=True)))


if __name__ == '__main__':
    main()
