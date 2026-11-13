"""Resume the completed ordinary-BD epoch to epoch 12, then test epoch 12."""
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
PREVIOUS = ROOT / 'work_dirs/dior_r_trainval/fcos_paper_bd_probe1_20261019'
CHECKPOINT = PREVIOUS / 'epoch_1.pth'
DIRECTORY = ROOT / 'work_dirs/dior_r_trainval/fcos_paper_bd_resume12_20261019'
FROZEN = DIRECTORY / 'snapshot/experiment.py'
REFERENCE = ROOT / 'work_dirs/dior_r_trainval/fcos_riou_bs2_800/test_final'
SPEC = importlib.util.spec_from_file_location('bd_full_run_helpers', ROOT / 'tools/run_dior_r_rio.py')
HELPERS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPERS)
HELPERS.DIRECTORY = DIRECTORY


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def hashes(paths):
    return {str(path): digest(ROOT / path) for path in paths}


def configuration():
    import mmcv
    import torch
    previous = json.loads((PREVIOUS / 'run.json').read_text())
    if previous['status'] != 'completed_one_epoch':
        raise RuntimeError('The first epoch must have completed successfully')
    if hashes(previous['source_sha256']) != previous['source_sha256']:
        raise RuntimeError('First-epoch sources changed; do not silently resume')
    original = PREVIOUS / 'snapshot/experiment.py'
    if digest(original) != previous['frozen_config_sha256']:
        raise RuntimeError('First-epoch configuration changed')
    cfg = mmcv.Config.fromfile(str(original))
    baseline = mmcv.Config.fromfile(str(ROOT / 'configs/dior_r/server_trainval/riou.py'))
    for key in ('data', 'optimizer', 'optimizer_config', 'lr_config', 'seed'):
        if cfg[key] != baseline[key]:
            raise RuntimeError('Unexpected baseline difference: ' + key)
    model, reference = copy.deepcopy(cfg.model), copy.deepcopy(baseline.model)
    loss = model.bbox_head.pop('loss_bbox')
    reference.bbox_head.pop('loss_bbox')
    if model != reference or loss.type != 'PaperBDLoss' or loss.alpha != 3. or loss.agbb:
        raise RuntimeError('Expected baseline FCOS with ordinary BD alpha=3')
    checkpoint = torch.load(str(CHECKPOINT), map_location='cpu')
    meta, optimizer = checkpoint['meta'], checkpoint.get('optimizer', {})
    if meta.get('epoch') != 1 or meta.get('iter') != 5863:
        raise RuntimeError('Expected standard epoch 1 / iteration 5863 checkpoint')
    if not optimizer.get('state') or not optimizer.get('param_groups'):
        raise RuntimeError('Missing SGD optimizer state')
    for group in optimizer['param_groups']:
        if not math.isclose(group['lr'], .0025) or not math.isclose(group['momentum'], .9):
            raise RuntimeError('Unexpected checkpoint optimizer settings')
    proof = dict(checkpoint_epoch=meta['epoch'], checkpoint_iter=meta['iter'],
                 optimizer_state_present=True, checkpoint_sha256=digest(CHECKPOINT),
                 optimizer_lrs=[group['lr'] for group in optimizer['param_groups']],
                 source_config_sha256=digest(original),
                 rng_state_note='Native checkpoint resumes model/optimizer/epoch/iteration; '
                                'it has no complete RNG/worker-state snapshot, so the next '
                                'augmented batches need not match uninterrupted training.')
    del checkpoint, optimizer
    cfg.runner.max_epochs = 12
    cfg.work_dir = str(DIRECTORY)
    cfg.load_from = None
    cfg.resume_from = str(CHECKPOINT)
    cfg.auto_resume = False
    # Retain the same training-only health check; global iteration exceeds 1000.
    cfg.custom_hooks[0]['diagnostic_iterations'] = []
    return cfg, previous, proof


def guard(state):
    if hashes(state['source_sha256']) != state['source_sha256']:
        raise RuntimeError('Sources changed during the full BD experiment')
    if digest(FROZEN) != state['frozen_config_sha256']:
        raise RuntimeError('Frozen full-run configuration changed')
    if digest(CHECKPOINT) != state['resume_checkpoint_sha256']:
        raise RuntimeError('The source checkpoint changed')


def worker(lock_fd):
    os.fstat(lock_fd)
    state = json.loads((DIRECTORY / 'run.json').read_text())
    state.update(worker_pid=os.getpid(), started_at=HELPERS.now())
    try:
        guard(state)
        HELPERS.run_child(
            [sys.executable, str(ROOT / 'tools/train.py'), str(FROZEN),
             '--resume-from', str(CHECKPOINT), '--seed', '42', '--no-validate'],
            DIRECTORY / 'train.log', state, 'training', lock_fd)
        final = DIRECTORY / 'epoch_12.pth'
        health = json.loads((DIRECTORY / 'training_health.json').read_text())
        if not final.is_file() or len(health) != 11:
            raise RuntimeError('Expected epoch 12 checkpoint and eleven resumed epoch checks')
        if [row['epoch'] for row in health] != list(range(2, 13)):
            raise RuntimeError('Unexpected resumed epoch sequence')
        if any(row['status'] != 'passed' for row in health):
            raise RuntimeError('Training-image health did not pass')
        state.update(training_completed_at=HELPERS.now(), checkpoint=str(final),
                     final_checkpoint_sha256=digest(final), training_health=health[-1])
        guard(state)
        HELPERS.run_child(
            [sys.executable, str(ROOT / 'tools/test.py'), str(FROZEN), str(final),
             '--work-dir', str(DIRECTORY / 'test_final'),
             '--out', str(DIRECTORY / 'test_final_predictions.pkl'),
             '--eval', 'mAP', '--eval-options', 'iou_thr=0.5', 'use_07_metric=True'],
            DIRECTORY / 'test.log', state, 'testing', lock_fd)
        evaluations = list((DIRECTORY / 'test_final').glob('eval_*.json'))
        if len(evaluations) != 1:
            raise RuntimeError('Expected one final test evaluation')
        metric = json.loads(evaluations[0].read_text())['metric']
        score = float(metric['mAP'])
        if not math.isfinite(score):
            raise RuntimeError('Nonfinite final test mAP')
        guard(state)
        if digest(final) != state['final_checkpoint_sha256']:
            raise RuntimeError('Final checkpoint changed during evaluation')
        result = dict(status='completed', epoch=12, loss='PaperBDLoss', alpha=3., agbb=False,
                      trainval_images=11725, test_images=11738, iou_thr=.5,
                      use_07_metric=True, test_mAP=score, mAP_percent=100 * score,
                      checkpoint=str(final), checkpoint_sha256=state['final_checkpoint_sha256'],
                      resumed_from=str(CHECKPOINT), evaluation_source=str(evaluations[0]),
                      finished_at=HELPERS.now(), source_sha256=state['source_sha256'])
        baseline_files = list(REFERENCE.glob('eval_*.json'))
        if len(baseline_files) == 1:
            baseline = float(json.loads(baseline_files[0].read_text())['metric']['mAP'])
            result.update(baseline_test_mAP=baseline,
                          difference_percentage_points=100 * (score - baseline),
                          baseline_source=str(baseline_files[0]))
        HELPERS.write_json(DIRECTORY / 'results.json', result)
        lines = ['# 普通 BD：固定第 12 轮测试结果', '',
                 'Rotated FCOS / ResNet-50 / FPN；alpha=3，AGBB 关闭。',
                 'DIOR-R train+val=11,725；test=11,738；800×800；batch=2；SGD。',
                 '从第 1 轮标准检查点恢复模型、优化器和计数，续训至第 12 轮。', '',
                 f'完整测试集 mAP50：**{100 * score:.2f}%**。']
        if 'baseline_test_mAP' in result:
            lines.append(f'原 SGD Rotated IoU 第 12 轮：{100 * baseline:.2f}%；'
                         f'差值：{100 * (score - baseline):+.2f} 个百分点。')
        lines.extend(['', '按固定第 12 轮报告，没有按测试集选择权重。'
                      '这是单次历史对照；续训未恢复完整随机数与数据工作进程状态。', ''])
        (DIRECTORY / 'results.md').write_text('\n'.join(lines), encoding='utf-8')
        state.update(status='completed', finished_at=HELPERS.now(), result=result)
        HELPERS.write_json(DIRECTORY / 'run.json', state)
    except Exception as error:
        state.update(status='failed', finished_at=HELPERS.now(), error=str(error))
        HELPERS.write_json(DIRECTORY / 'run.json', state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'start', 'worker', 'status'))
    parser.add_argument('--lock-fd', type=int)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.action == 'status':
        print((DIRECTORY / 'run.json').read_text())
        return
    if args.action == 'worker':
        if args.lock_fd is None:
            parser.error('worker requires an inherited queue lock')
        worker(args.lock_fd)
        return
    cfg, previous, proof = configuration()
    if args.action == 'check':
        print(json.dumps(dict(target_epoch=12, remaining_epochs=11,
                              final_test=True, resume=proof), indent=2))
        return
    lock_path = ROOT / 'work_dirs/dior_r_trainval/server_queue.lock'
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('Another trainval experiment is running')
        if DIRECTORY.exists() and any(DIRECTORY.iterdir()):
            parser.error(f'Existing output will not be overwritten: {DIRECTORY}')
        FROZEN.parent.mkdir(parents=True, exist_ok=True)
        cfg.dump(str(FROZEN))
        sources = list(dict.fromkeys(list(previous['source_sha256']) +
                                    ['tools/test.py', 'tools/run_paper_bd_full.py']))
        for name in sources:
            path = DIRECTORY / 'snapshot/source' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((ROOT / name).read_bytes())
        HELPERS.write_json(DIRECTORY / 'resume_check.json', proof)
        HELPERS.write_json(DIRECTORY / 'run.json', dict(
            status='starting', created_at=HELPERS.now(), target_epoch=12,
            completed_epochs_before_resume=1, remaining_epochs=11,
            resume_from=str(CHECKPOINT), resume_checkpoint_sha256=proof['checkpoint_sha256'],
            resume_epoch=1, resume_iteration=5863, optimizer_state_present=True,
            frozen_config=str(FROZEN), frozen_config_sha256=digest(FROZEN),
            source_sha256=hashes(sources), seed=42,
            loss=dict(type='PaperBDLoss', alpha=3., agbb=False, loss_weight=1.),
            intermediate_test=False, final_test=True, checkpoint_selection='fixed epoch 12',
            lr_schedule='Original SGD schedule; step at epochs 8/11, no warmup restart.',
            rng_state_note=proof['rng_state_note'],
            health_guard='Existing finite-metric/gradient guard plus 128 fixed training '
                         'images per epoch; no test-based stopping or checkpoint selection.'))
        env = os.environ.copy()
        env.update(PYTHONUNBUFFERED='1', CUDA_VISIBLE_DEVICES='0')
        env['LD_LIBRARY_PATH'] = '/usr/lib/wsl/lib' + (
            ':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
        with (DIRECTORY / 'worker.log').open('w') as logfile:
            process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), 'worker',
                 '--lock-fd', str(lock.fileno())], cwd=ROOT, env=env,
                stdin=subprocess.DEVNULL, stdout=logfile, stderr=subprocess.STDOUT,
                start_new_session=True, pass_fds=(lock.fileno(),))
        (DIRECTORY / 'worker.pid').write_text(str(process.pid) + '\n')
        print(json.dumps(dict(worker_pid=process.pid, work_dir=str(DIRECTORY),
                              target_epoch=12, final_test=True)))


if __name__ == '__main__':
    main()
