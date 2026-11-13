"""Detached original-WF retry, optionally resuming its first epoch to epoch 12."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
CONFIG = 'configs/dior_r/server_trainval/wf_retry_1epoch.py'
DIRECTORY = ROOT / 'work_dirs/dior_r_trainval/fcos_wf_retry_1epoch_20261007'
TARGET_EPOCH = 1
SOURCE_CHECKPOINT = None
LOW_MEM = False


def write_state(state):
    temporary = DIRECTORY / 'run.json.tmp'
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(DIRECTORY / 'run.json')


def validate():
    import mmcv
    reference = mmcv.Config.fromfile(str(ROOT / 'configs/dior_r/server_trainval/wf.py'))
    cfg = mmcv.Config.fromfile(str(ROOT / CONFIG))
    for key in ('model', 'optimizer', 'optimizer_config', 'lr_config',
                'evaluation', 'seed', 'load_from'):
        assert cfg[key] == reference[key], f'Unexpected difference: {key}'
    if LOW_MEM:
        expected_data = reference.data.copy()
        expected_data.workers_per_gpu = 0
        assert cfg.data == expected_data, 'Unexpected data loader change'
    else:
        assert cfg.data == reference.data, 'Unexpected data change'
    assert cfg.runner.max_epochs == TARGET_EPOCH
    assert (ROOT / cfg.work_dir).resolve() == DIRECTORY.resolve()
    if SOURCE_CHECKPOINT is None:
        assert cfg.resume_from is None
    else:
        import torch
        assert Path(cfg.resume_from).resolve() == SOURCE_CHECKPOINT.resolve()
        checkpoint = torch.load(SOURCE_CHECKPOINT, map_location='cpu')
        assert checkpoint['meta']['epoch'] == 1
        assert checkpoint['meta']['iter'] == 5863
        assert checkpoint.get('optimizer', {}).get('state'), 'Missing optimizer state'
        previous = json.loads((SOURCE_CHECKPOINT.parent / 'run.json').read_text())
        assert previous['status'] == 'completed'
        for path, digest in previous['source_sha256'].items():
            if path.startswith('mmrotate/') or path == 'tools/train.py':
                assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest
        print(json.dumps(dict(resume_epoch=1, resume_iteration=5863,
                              optimizer_state_present=True), ensure_ascii=False))
    return cfg


def worker(lock_fd):
    # Keep the inherited common queue lock until training and reporting finish.
    os.fstat(lock_fd)
    state = dict(status='training', worker_pid=os.getpid(), config=CONFIG,
                 target_epoch=TARGET_EPOCH,
                 started_at=datetime.now().astimezone().isoformat(),
                 source_sha256={path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
                    for path in ('mmrotate/models/losses/rotated_wf_iou_loss.py',
                                 'mmrotate/models/dense_heads/rotated_fcos_head.py',
                                 'mmrotate/datasets/dior.py', 'tools/train.py', CONFIG)})
    if LOW_MEM:
        state['loader_workers_per_gpu'] = 0
    write_state(state)
    if SOURCE_CHECKPOINT is not None:
        state['resume_from'] = str(SOURCE_CHECKPOINT)
        state['resume_checkpoint_sha256'] = hashlib.sha256(SOURCE_CHECKPOINT.read_bytes()).hexdigest()
        state['rng_state_note'] = ('Framework resumes model, optimizer, epoch and iteration; '
                                  'this checkpoint does not store RNG states.')
        write_state(state)
    print(f'Original WF: train to epoch {TARGET_EPOCH}, evaluation on test.txt', flush=True)
    with (DIRECTORY / 'train.log').open('w') as logfile:
        process = subprocess.Popen([sys.executable, str(ROOT / 'tools/train.py'),
                                    str(ROOT / CONFIG), '--seed', '42'],
                                   cwd=ROOT, stdout=logfile, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL)
        state['train_pid'] = process.pid
        write_state(state)
        code = process.wait()
    state['returncode'] = code
    state['finished_at'] = datetime.now().astimezone().isoformat()
    if code:
        state['status'] = 'failed'
        write_state(state)
        raise SystemExit(code)
    records = [json.loads(line) for path in DIRECTORY.glob('20*.log.json')
               for line in path.read_text().splitlines()]
    evaluations = [r for r in records if r.get('mode') == 'val' and r.get('epoch') == TARGET_EPOCH]
    final_checkpoint = DIRECTORY / f'epoch_{TARGET_EPOCH}.pth'
    if not evaluations or not final_checkpoint.is_file():
        state['status'] = 'failed'
        state['error'] = 'Missing final checkpoint or final evaluation'
        write_state(state)
        raise RuntimeError(state['error'])
    score = float(evaluations[-1]['mAP'])
    if not math.isfinite(score):
        raise RuntimeError('Nonfinite evaluation mAP')
    result = dict(config=CONFIG, checkpoint=str(final_checkpoint),
                  epoch=TARGET_EPOCH, trainval_images=11725, test_images=11738,
                  test_mAP=score, loss_type='RotatedWFIoULoss', seed=42,
                  evaluation_source='Epoch(val) uses ImageSets/Main/test.txt')
    (DIRECTORY / 'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    lines = [f'# 原 WF：训练到第 {TARGET_EPOCH} 轮', '',
             ('从首次重训的 epoch_1.pth 恢复模型、优化器、轮数和迭代次数。'
              if SOURCE_CHECKPOINT is not None else
              '从 ImageNet 预训练主干重新训练，未恢复旧检测器权重。'),
             'trainval=11,725；test=11,738；800×800、batch=2、SGD lr=0.0025、seed=42。',
             '原 RotatedWFIoULoss：tau=0.7、eps=1e-7、loss_weight=1.0。',
             '保持原 warmup=500 和 step=[8,11]；续训按已有迭代数执行，不重新 warmup。', '',
             f'完整 test mAP：**{score*100:.2f}%**，权重：epoch_{TARGET_EPOCH}.pth。', '',
             '该指标来自训练结束后的配置评估，日志阶段名仍为 Epoch(val)，实际数据为 test.txt。',
             ('报告预先固定的第 12 轮结果，不按 test 分数选择最佳轮次。'
              if TARGET_EPOCH == 12 else
              '这是一次 1 轮重现检查，不是新的 12 轮最终模型结果。'), '']
    (DIRECTORY / 'results.md').write_text('\n'.join(lines), encoding='utf-8')
    state.update(status='completed', result=result)
    write_state(state)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def main():
    global CONFIG, DIRECTORY, TARGET_EPOCH, SOURCE_CHECKPOINT, LOW_MEM
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('start', 'worker'))
    parser.add_argument('--lock-fd', type=int)
    parser.add_argument('--continue-to-12', action='store_true')
    parser.add_argument('--retry2', action='store_true',
                        help='Retry the 12-epoch continuation in a fresh work directory.')
    parser.add_argument('--low-memory-retry', action='store_true',
                        help='Resume epoch 1 with zero data-loader subprocesses.')
    parser.add_argument('--low-memory-retry2', action='store_true',
                        help='Retry the low-memory continuation in a fresh directory.')
    args = parser.parse_args()
    if args.continue_to_12 or args.retry2 or args.low_memory_retry or args.low_memory_retry2:
        SOURCE_CHECKPOINT = DIRECTORY / 'epoch_1.pth'
        if args.low_memory_retry2:
            CONFIG = 'configs/dior_r/server_trainval/wf_retry_resume12_lowmem2.py'
            DIRECTORY = ROOT / 'work_dirs/dior_r_trainval/fcos_wf_retry_resume12_lowmem2_20261007'
            LOW_MEM = True
        elif args.low_memory_retry:
            CONFIG = 'configs/dior_r/server_trainval/wf_retry_resume12_lowmem.py'
            DIRECTORY = ROOT / 'work_dirs/dior_r_trainval/fcos_wf_retry_resume12_lowmem_20261007'
            LOW_MEM = True
        elif args.retry2:
            CONFIG = 'configs/dior_r/server_trainval/wf_retry_resume12_retry2.py'
            DIRECTORY = ROOT / 'work_dirs/dior_r_trainval/fcos_wf_retry_resume12_retry2_20261007'
        else:
            CONFIG = 'configs/dior_r/server_trainval/wf_retry_resume12.py'
            DIRECTORY = ROOT / 'work_dirs/dior_r_trainval/fcos_wf_retry_resume12_20261007'
        TARGET_EPOCH = 12
    if args.action == 'worker':
        if args.lock_fd is None:
            parser.error('worker requires inherited lock-fd')
        try:
            worker(args.lock_fd)
        except Exception as error:
            state_file = DIRECTORY / 'run.json'
            state = json.loads(state_file.read_text()) if state_file.exists() else {}
            state.update(status='failed', error=str(error),
                         finished_at=datetime.now().astimezone().isoformat())
            write_state(state)
            raise
        return
    validate()
    common_lock = ROOT / 'work_dirs/dior_r_trainval/server_queue.lock'
    common_lock.parent.mkdir(parents=True, exist_ok=True)
    lock = common_lock.open('a')
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error('Another trainval queue is running')
    if DIRECTORY.exists() and any(DIRECTORY.iterdir()):
        parser.error(f'Existing run directory will not be overwritten: {DIRECTORY}')
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED='1', CUDA_VISIBLE_DEVICES='0')
    env['LD_LIBRARY_PATH'] = '/usr/lib/wsl/lib' + (
        ':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    with (DIRECTORY / 'worker.log').open('w') as logfile:
        command = [sys.executable, str(Path(__file__).resolve()),
                   'worker', '--lock-fd', str(lock.fileno())]
        if args.continue_to_12:
            command.append('--continue-to-12')
        if args.retry2:
            command.append('--retry2')
        if args.low_memory_retry:
            command.append('--low-memory-retry')
        if args.low_memory_retry2:
            command.append('--low-memory-retry2')
        process = subprocess.Popen(command,
                                   cwd=ROOT, env=env, stdout=logfile, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True,
                                   pass_fds=(lock.fileno(),))
    (DIRECTORY / 'worker.pid').write_text(str(process.pid)+'\n')
    print(json.dumps(dict(worker_pid=process.pid, work_dir=str(DIRECTORY)), ensure_ascii=False))


if __name__ == '__main__':
    main()
