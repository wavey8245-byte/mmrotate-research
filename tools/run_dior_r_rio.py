"""Run one isolated 12-epoch RiO-loss experiment and fixed-checkpoint test."""
import argparse
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

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG = 'configs/dior_r/server_trainval/rio.py'
DIRECTORY = ROOT / 'work_dirs/dior_r_trainval/fcos_rio_bs2_800_20261015'
PREFLIGHT = ROOT / 'work_dirs/dior_r/rio_preflight_20261015.json'
FROZEN_CONFIG = DIRECTORY / 'snapshot/experiment.py'
SOURCES = (
    'mmrotate/models/losses/rotated_rio_loss.py',
    'mmrotate/models/losses/__init__.py',
    'mmrotate/models/dense_heads/rotated_fcos_head.py',
    'mmrotate/datasets/dior.py', 'tools/train.py', 'tools/test.py', CONFIG)


def now():
    return datetime.now().astimezone().isoformat()


def hashes(paths):
    return {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            for path in paths}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n',
                         encoding='utf-8')
    temporary.replace(path)


def validate():
    import mmcv
    cfg = mmcv.Config.fromfile(str(ROOT / CONFIG))
    baseline = mmcv.Config.fromfile(str(ROOT / 'configs/dior_r/server_trainval/riou.py'))
    left, right = cfg.to_dict(), baseline.to_dict()
    if cfg.model.bbox_head.get('bounded_angle', False):
        assert cfg.model.bbox_head.scale_angle is False
        assert cfg.model.bbox_head.bounded_angle is True
        left['model']['bbox_head'].pop('bounded_angle')
        left['model']['bbox_head']['scale_angle'] = right['model']['bbox_head']['scale_angle']
    for value in (left, right):
        value.pop('work_dir')
        value['model']['bbox_head'].pop('loss_bbox')
    if left != right:
        raise RuntimeError('Experiment differs from baseline beyond loss/work_dir')
    assert cfg.runner.max_epochs == 12 and cfg.seed == 42
    assert cfg.load_from is None and cfg.resume_from is None
    assert (ROOT / cfg.work_dir).resolve() == DIRECTORY.resolve()
    split_root = ROOT / 'data/DIOR-R/ImageSets/Main'
    split_ids = {name: set((split_root / f'{name}.txt').read_text().split())
                 for name in ('train', 'val', 'test')}
    assert len(split_ids['train']) == 5862 and len(split_ids['val']) == 5863
    assert len(split_ids['test']) == 11738
    assert not split_ids['train'] & split_ids['val']
    assert not (split_ids['train'] | split_ids['val']) & split_ids['test']
    preflight = json.loads(PREFLIGHT.read_text())
    if preflight['status'] != 'passed' or preflight.get('protocol_version', 0) < 2:
        raise RuntimeError('A successful version-2 stability check is required')
    specification = importlib.util.spec_from_file_location(
        'rio_preflight_checks', ROOT / 'tools/check_dior_r_rio.py')
    checks = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(checks)
    errors = checks.stability_errors(preflight)
    if errors:
        raise RuntimeError('; '.join(errors))
    if preflight['config'] != CONFIG:
        raise RuntimeError('Preflight was run with a different configuration')
    if preflight.get('resolved_config_sha256') != hashlib.sha256(cfg.pretty_text.encode()).hexdigest():
        raise RuntimeError('Resolved configuration changed since stability check')
    if hashes(preflight['source_sha256']) != preflight['source_sha256']:
        raise RuntimeError('Loss/head/config changed since stability check')
    return cfg, preflight


def check_training_health():
    """Stop sustained large gradients; do not use test mAP as a stopping rule."""
    records = []
    for path in DIRECTORY.glob('20*.log.json'):
        lines = path.read_text().splitlines()
        for index, line in enumerate(lines):
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                if index == len(lines) - 1:
                    continue  # Logger may currently be writing this line.
                raise RuntimeError(f'Corrupt training JSON log: {path}')
            if item.get('mode') != 'train':
                continue
            for name, value in item.items():
                if ('loss' in name or name == 'grad_norm') and not math.isfinite(float(value)):
                    raise RuntimeError(f'Nonfinite training metric {name}: {value}')
            records.append(item)
    if len(records) >= 4:
        recent = [item.get('grad_norm', 0) for item in records[-4:]]
        if float(np.median(recent)) > 3500:
            raise RuntimeError('Sustained gradient explosion: last four logged '
                               f'norms are {recent}; training stopped')


def run_child(command, logfile, state, phase, lock_fd):
    state.update(status=phase, phase_started_at=now())
    with logfile.open('w') as output:
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, pass_fds=(lock_fd,))
        state[phase + '_pid'] = process.pid
        write_json(DIRECTORY / 'run.json', state)
        try:
            while True:
                try:
                    code = process.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    if phase == 'training':
                        check_training_health()
        except Exception:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    state[phase + '_returncode'] = code
    if code:
        raise RuntimeError(f'{phase} exited with code {code}; see {logfile.name}')


def worker(lock_fd):
    os.fstat(lock_fd)
    state = json.loads((DIRECTORY / 'run.json').read_text())
    state.update(worker_pid=os.getpid(), started_at=now())
    try:
        if hashes(SOURCES) != state['source_sha256']:
            raise RuntimeError('Source changed after experiment launch')
        if hashes([str(FROZEN_CONFIG)])[str(FROZEN_CONFIG)] != state['frozen_config_sha256']:
            raise RuntimeError('Frozen configuration changed after experiment launch')
        run_child([sys.executable, str(ROOT / 'tools/train.py'),
                   str(FROZEN_CONFIG), '--seed', '42'],
                  DIRECTORY / 'train.log', state, 'training', lock_fd)
        checkpoint = DIRECTORY / 'epoch_12.pth'
        if not checkpoint.is_file():
            raise RuntimeError('Training ended without epoch_12.pth')
        if hashes(SOURCES) != state['source_sha256']:
            raise RuntimeError('Source changed during training; evaluation stopped')
        if hashes([str(FROZEN_CONFIG)])[str(FROZEN_CONFIG)] != state['frozen_config_sha256']:
            raise RuntimeError('Frozen configuration changed during training')
        run_child([sys.executable, str(ROOT / 'tools/test.py'),
                   str(FROZEN_CONFIG), str(checkpoint),
                   '--work-dir', str(DIRECTORY / 'test_final'),
                   '--out', str(DIRECTORY / 'test_final_predictions.pkl'),
                   '--eval', 'mAP'],
                  DIRECTORY / 'test.log', state, 'testing', lock_fd)
        evaluations = sorted((DIRECTORY / 'test_final').glob('eval_*.json'))
        if len(evaluations) != 1:
            raise RuntimeError('Expected exactly one final test evaluation')
        score = float(json.loads(evaluations[0].read_text())['metric']['mAP'])
        if not math.isfinite(score):
            raise RuntimeError('Nonfinite final test mAP')
        result = dict(config=CONFIG, checkpoint=str(checkpoint), epoch=12,
                      test_mAP=score, trainval_images=11725, test_images=11738,
                      seed=42, loss_type='RotatedRiOLoss',
                      angle_parameterization=state.get('angle_parameterization', 'unbounded'),
                      regression_weights=dict(kld=5, spatial_l1=5, sp_l1=5),
                      evaluation_source=str(evaluations[0]),
                      source_sha256=state['source_sha256'])
        reference = ROOT / ('work_dirs/dior_r_trainval/fcos_riou_bs2_800/'
                            'test_final/eval_20261006_092447.json')
        if reference.is_file():
            baseline = float(json.loads(reference.read_text())['metric']['mAP'])
            result.update(baseline_test_mAP=baseline,
                          difference_percentage_points=100 * (score - baseline),
                          baseline_source=str(reference))
        write_json(DIRECTORY / 'results.json', result)
        lines = ['# RiO 三项回归损失：固定第 12 轮结果', '',
                 'Rotated FCOS：5×KLD + 5×空间 L1 + 5×SP-L1。',
                 'trainval=11,725；test=11,738；800×800；batch=2；seed=42。',
                 '从 ImageNet 预训练主干开始，固定第 12 轮权重，不按 test 选轮次。', '',
                 f'最终 test mAP@0.5：**{score * 100:.2f}%**。']
        if 'baseline_test_mAP' in result:
            lines.append(f'原 RIoU 基线：{baseline * 100:.2f}%；差值：'
                         f'{100 * (score - baseline):+.2f} 个百分点。')
        lines.append('角度参数化：' + state.get('angle_parameterization', 'unbounded'))
        lines.extend(['', '这是 RiO-DETR 损失组合在 FCOS 上的适配实验，'
                      '并非完整 RiO-DETR 复现。', ''])
        (DIRECTORY / 'results.md').write_text('\n'.join(lines), encoding='utf-8')
        state.update(status='completed', finished_at=now(), result=result)
        write_json(DIRECTORY / 'run.json', state)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    except Exception as error:
        state.update(status='failed', finished_at=now(), error=str(error))
        write_json(DIRECTORY / 'run.json', state)
        raise


def main():
    global CONFIG, DIRECTORY, PREFLIGHT, FROZEN_CONFIG, SOURCES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('start', 'worker', 'status', 'check'))
    parser.add_argument('--lock-fd', type=int)
    parser.add_argument('--bounded', action='store_true',
                        help='Run the bounded-angle recovery experiment in a new directory.')
    args = parser.parse_args()
    if args.bounded:
        CONFIG = 'configs/dior_r/server_trainval/rio_bounded.py'
        DIRECTORY = ROOT / 'work_dirs/dior_r_trainval/fcos_rio_bounded_bs2_800_20261016'
        PREFLIGHT = ROOT / 'work_dirs/dior_r/rio_bounded_preflight_20261016.json'
        FROZEN_CONFIG = DIRECTORY / 'snapshot/experiment.py'
        SOURCES = (*SOURCES[:-1], 'configs/dior_r/server_trainval/rio.py',
                   'tools/check_dior_r_rio.py', 'tools/run_dior_r_rio.py', CONFIG)
    os.chdir(ROOT)
    if args.action == 'worker':
        if args.lock_fd is None:
            parser.error('worker requires an inherited lock descriptor')
        worker(args.lock_fd)
        return
    if args.action == 'status':
        print((DIRECTORY / 'run.json').read_text(encoding='utf-8'))
        return
    cfg, preflight = validate()
    if args.action == 'check':
        print('Configuration, disjoint dataset splits and preflight hashes passed.')
        return
    lock_path = ROOT / 'work_dirs/dior_r_trainval/server_queue.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('Another trainval queue is running')
        if DIRECTORY.exists() and any(DIRECTORY.iterdir()):
            parser.error(f'Existing experiment will not be overwritten: {DIRECTORY}')
        DIRECTORY.mkdir(parents=True, exist_ok=True)
        FROZEN_CONFIG.parent.mkdir(parents=True, exist_ok=True)
        cfg.dump(str(FROZEN_CONFIG))
        for name in SOURCES:
            snapshot = DIRECTORY / 'snapshot/source' / name
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes((ROOT / name).read_bytes())
        write_json(DIRECTORY / 'preflight.json', preflight)
        write_json(DIRECTORY / 'run.json', dict(
            status='starting', config=CONFIG, frozen_config=str(FROZEN_CONFIG),
            created_at=now(), target_epoch=12, seed=42,
            source_sha256=hashes(SOURCES),
            frozen_config_sha256=hashes([str(FROZEN_CONFIG)])[str(FROZEN_CONFIG)],
            angle_parameterization=('pi*(sigmoid(logit)-0.5), scale_angle=False'
                                    if args.bounded else 'unbounded, learnable angle scale'),
            health_guard='Stop nonfinite metrics or last-four logged gradient median > 3500; '
                         'no test mAP stopping rule.',
            adaptation='FCOS regression adaptation; KLD(log1p,tau=1,sqrt=False), '
                       'image-normalized spatial L1, pi-periodic angular L1; '
                       'weights 5/5/5.' + (' Bounded le90 angle, scale_angle=False.'
                                           if args.bounded else '')))
        env = os.environ.copy()
        env.update(PYTHONUNBUFFERED='1', CUDA_VISIBLE_DEVICES='0')
        env['LD_LIBRARY_PATH'] = '/usr/lib/wsl/lib' + (
            ':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
        with (DIRECTORY / 'worker.log').open('w') as logfile:
            command = [sys.executable, str(Path(__file__).resolve()),
                       'worker', '--lock-fd', str(lock.fileno())]
            if args.bounded:
                command.append('--bounded')
            process = subprocess.Popen(
                command,
                cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=logfile, stderr=subprocess.STDOUT,
                start_new_session=True, pass_fds=(lock.fileno(),))
        (DIRECTORY / 'worker.pid').write_text(str(process.pid) + '\n')
        print(json.dumps(dict(worker_pid=process.pid, work_dir=str(DIRECTORY))))


if __name__ == '__main__':
    main()
