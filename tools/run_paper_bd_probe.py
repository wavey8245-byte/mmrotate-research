"""One standard-runner epoch of ordinary BD on FCOS; training-only diagnostics."""
import argparse
import copy
from datetime import datetime
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CONFIG = 'configs/dior_r/server_trainval/paper_bd.py'
REFERENCE_CONFIG = 'configs/dior_r/server_trainval/riou.py'
DIRECTORY = ROOT / 'work_dirs/dior_r_trainval/fcos_paper_bd_probe1_20261019'
FROZEN = DIRECTORY / 'snapshot/experiment.py'
CORE_SOURCES = ('mmrotate/models/losses/paper_rotated_losses.py',
           'mmrotate/models/losses/rotated_iou_loss.py',
           'mmrotate/models/dense_heads/rotated_fcos_head.py',
           'mmrotate/core/rio_training_health.py', 'mmrotate/datasets/dior.py',
           'tools/train.py', 'tools/run_dior_r_rio.py',
           'tools/run_paper_bd_probe.py')
SOURCES = CORE_SOURCES + (CONFIG, REFERENCE_CONFIG)
SPEC = importlib.util.spec_from_file_location('rio_run_helpers', ROOT / 'tools/run_dior_r_rio.py')
HELPERS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPERS)
HELPERS.DIRECTORY = DIRECTORY


def configuration():
    import mmcv
    cfg = mmcv.Config.fromfile(str(ROOT / CONFIG))
    baseline = mmcv.Config.fromfile(str(ROOT / REFERENCE_CONFIG))
    for key in ('data', 'optimizer', 'optimizer_config', 'lr_config', 'seed',
                'load_from', 'resume_from'):
        if cfg[key] != baseline[key]:
            raise RuntimeError(f'Unexpected protocol difference: {key}')
    if cfg.load_from is not None or cfg.resume_from is not None:
        raise RuntimeError('The probe must start from ImageNet initialization')
    model, reference = copy.deepcopy(cfg.model), copy.deepcopy(baseline.model)
    model.bbox_head.pop('loss_bbox')
    reference.bbox_head.pop('loss_bbox')
    if model != reference:
        raise RuntimeError('Probe head must match the baseline, including raw angles')
    if cfg.model.bbox_head.loss_bbox.type != 'PaperBDLoss':
        raise RuntimeError('Expected ordinary PaperBDLoss')
    if cfg.model.bbox_head.loss_bbox.get('agbb', False):
        raise RuntimeError('This job selects ordinary BD, with AGBB disabled')
    proof_path = ROOT / 'work_dirs/dior_r_trainval/paper_losses_probe_1000_20261019/report.json'
    proof = json.loads(proof_path.read_text())
    if proof['status'] != 'completed_short_controls' or proof['arms']['bd']['steps_completed'] < 1000:
        raise RuntimeError('Completed 1000-step ordinary BD check required')
    if HELPERS.hashes(proof['source_sha256']) != proof['source_sha256']:
        raise RuntimeError('Short-check sources changed')
    if proof['arms']['bd']['loss'] != dict(cfg.model.bbox_head.loss_bbox):
        raise RuntimeError('BD specification changed since the short check')
    cfg.runner.max_epochs = 1
    cfg.work_dir = str(DIRECTORY)
    dataset_cfg = copy.deepcopy(cfg.data.test)
    dataset_cfg.ann_file = 'data/DIOR-R/ImageSets/Main/train.txt'
    dataset_cfg.img_prefix = 'data/DIOR-R/JPEGImages-trainval/'
    dataset_cfg.test_mode = True
    imports = list(cfg.custom_imports['imports'])
    imports.append('mmrotate.core.rio_training_health')
    cfg.custom_imports = dict(imports=list(dict.fromkeys(imports)),
                              allow_failed_imports=False)
    cfg.custom_hooks = [dict(type='RiOTrainingHealthHook', dataset_cfg=dataset_cfg,
                             images=128, seed=42, minimum_map=.05,
                             minimum_fraction=.5, diagnostic_iterations=[500, 1000],
                             priority='LOW')]
    return cfg


def worker(lock_fd):
    os.fstat(lock_fd)
    state = json.loads((DIRECTORY / 'run.json').read_text())
    state['worker_pid'] = os.getpid()
    try:
        if HELPERS.hashes(SOURCES) != state['source_sha256']:
            raise RuntimeError('Probe sources changed after launch')
        if HELPERS.hashes([str(FROZEN)])[str(FROZEN)] != state['frozen_config_sha256']:
            raise RuntimeError('Probe configuration changed after launch')
        # The epoch hook checks training images before any further stage.
        # --no-validate keeps the test split unused in this diagnostic probe.
        HELPERS.run_child([sys.executable, str(ROOT / 'tools/train.py'), str(FROZEN),
                           '--seed', '42', '--no-validate'],
                          DIRECTORY / 'train.log', state, 'training', lock_fd)
        if HELPERS.hashes(SOURCES) != state['source_sha256']:
            raise RuntimeError('Source changed during the epoch')
        if HELPERS.hashes([str(FROZEN)])[str(FROZEN)] != state['frozen_config_sha256']:
            raise RuntimeError('Frozen configuration changed during the epoch')
        checkpoint = DIRECTORY / 'epoch_1.pth'
        checks = json.loads((DIRECTORY / 'training_health.json').read_text())
        if not checkpoint.is_file() or checks[-1]['status'] != 'passed':
            raise RuntimeError('Missing checkpoint or failed training-image health check')
        state.update(status='completed_one_epoch',
                     finished_at=datetime.now().astimezone().isoformat(),
                     training_health=checks[-1], checkpoint=str(checkpoint),
                     note='One-epoch diagnostic only. No test AP and no automatic continuation.')
    except Exception as error:
        state.update(status='failed', error=str(error),
                     finished_at=datetime.now().astimezone().isoformat())
        health = DIRECTORY / 'training_health.json'
        if health.is_file():
            state['training_health'] = json.loads(health.read_text())[-1]
        HELPERS.write_json(DIRECTORY / 'run.json', state)
        raise
    HELPERS.write_json(DIRECTORY / 'run.json', state)
    summary = [
        '# 普通 BD：一轮训练验证结果', '',
        '模型：Rotated FCOS / R50-FPN；标准高斯 BD，alpha=3，AGBB 关闭。',
        '从 ImageNet 预训练主干开始；DIOR-R train+val；800×800；batch=2；原 SGD。',
        '只训练一轮，轮末诊断使用固定 128 张训练图片，未读取 test AP。', '',
        '状态：' + state['status'],
        '训练图片诊断 AP50：' + format(state['training_health']['training_image_mAP'] * 100, '.2f') + '%',
        '权重：epoch_1.pth', '',
        '该指标用于检查学习情况，不代表测试集精度；未自动延长为12轮。', '']
    (DIRECTORY / 'results.md').write_text('\n'.join(summary), encoding='utf-8')


def main():
    global CONFIG, REFERENCE_CONFIG, DIRECTORY, FROZEN, SOURCES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'start', 'worker', 'status'))
    parser.add_argument('--lock-fd', type=int)
    parser.add_argument('--config', default=CONFIG)
    parser.add_argument('--reference-config', default=REFERENCE_CONFIG)
    parser.add_argument('--work-dir', default=str(DIRECTORY))
    args = parser.parse_args()
    os.chdir(ROOT)
    CONFIG = args.config
    REFERENCE_CONFIG = args.reference_config
    DIRECTORY = (ROOT / args.work_dir).resolve()
    if not DIRECTORY.is_relative_to(ROOT / 'work_dirs'):
        parser.error('Output must stay inside repository work_dirs')
    FROZEN = DIRECTORY / 'snapshot/experiment.py'
    SOURCES = tuple(dict.fromkeys(CORE_SOURCES + (CONFIG, REFERENCE_CONFIG)))
    HELPERS.DIRECTORY = DIRECTORY
    if args.action == 'worker':
        if args.lock_fd is None:
            parser.error('worker requires an inherited lock descriptor')
        worker(args.lock_fd)
        return
    if args.action == 'status':
        print((DIRECTORY / 'run.json').read_text())
        return
    cfg = configuration()
    if args.action == 'check':
        print('One epoch; baseline head/optimizer/data; 128 training-image health; test unused.')
        return
    lock_path = ROOT / 'work_dirs/dior_r_trainval/server_queue.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('Another trainval experiment is running')
        if DIRECTORY.exists() and any(DIRECTORY.iterdir()):
            parser.error(f'Existing probe will not be overwritten: {DIRECTORY}')
        FROZEN.parent.mkdir(parents=True, exist_ok=True)
        cfg.dump(str(FROZEN))
        for name in SOURCES:
            relative = (ROOT / name).resolve().relative_to(ROOT)
            snapshot = DIRECTORY / 'snapshot/source' / relative
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes((ROOT / name).read_bytes())
        HELPERS.write_json(DIRECTORY / 'run.json', dict(
            status='starting', config=CONFIG, reference_config=REFERENCE_CONFIG,
            frozen_config=str(FROZEN),
            created_at=datetime.now().astimezone().isoformat(), target_epoch=1,
            seed=42, source_sha256=HELPERS.hashes(SOURCES),
            frozen_config_sha256=HELPERS.hashes([str(FROZEN)])[str(FROZEN)],
            test_evaluation=False, automatic_continuation=False,
            purpose='One standard-runner epoch of ordinary BD on FCOS, checked against '
                    'the original RIoU training protocol. No test-based selection.'))
        env = os.environ.copy()
        env.update(PYTHONUNBUFFERED='1', CUDA_VISIBLE_DEVICES='0')
        env['LD_LIBRARY_PATH'] = '/usr/lib/wsl/lib' + (
            ':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
        with (DIRECTORY / 'worker.log').open('w') as logfile:
            process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), 'worker',
                 '--config', CONFIG, '--reference-config', REFERENCE_CONFIG,
                 '--work-dir', str(DIRECTORY),
                 '--lock-fd', str(lock.fileno())],
                cwd=ROOT, env=env, stdout=logfile, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True,
                pass_fds=(lock.fileno(),))
        (DIRECTORY / 'worker.pid').write_text(str(process.pid) + '\n')
        print(json.dumps(dict(worker_pid=process.pid, work_dir=str(DIRECTORY))))


if __name__ == '__main__':
    main()
