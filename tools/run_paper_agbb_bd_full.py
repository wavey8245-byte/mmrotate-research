"""Train AGBB+BD FCOS for twelve epochs from ImageNet, then test epoch 12."""
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
CONFIG = 'configs/dior_r/server_trainval/paper_bd_agbb.py'
DIRECTORY = ROOT / 'work_dirs/dior_r_trainval/fcos_paper_agbb_bd_bs2_800_20261020'
FROZEN = DIRECTORY / 'snapshot/experiment.py'
PROOF = ROOT / 'work_dirs/dior_r_trainval/paper_losses_probe_20261018/report.json'
BD_REFERENCE = ROOT / 'work_dirs/dior_r_trainval/fcos_paper_bd_recovery12_20261019/results.json'
RIOU_REFERENCE = ROOT / 'work_dirs/dior_r_trainval/fcos_riou_bs2_800/test_final/eval_20261006_092447.json'
SPEC = importlib.util.spec_from_file_location('agbb_run_helpers', ROOT / 'tools/run_dior_r_rio.py')
HELPERS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPERS)
HELPERS.DIRECTORY = DIRECTORY


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def hashes(paths):
    return {str(path): digest(ROOT / path) for path in paths}


def configuration():
    import mmcv
    cfg = mmcv.Config.fromfile(str(ROOT / CONFIG))
    baseline = mmcv.Config.fromfile(str(ROOT / 'configs/dior_r/server_trainval/riou.py'))
    ordinary = mmcv.Config.fromfile(str(ROOT / 'configs/dior_r/server_trainval/paper_bd.py'))
    for key in ('data', 'optimizer', 'optimizer_config', 'lr_config', 'seed',
                'load_from', 'resume_from', 'runner', 'checkpoint_config'):
        if cfg[key] != baseline[key] or cfg[key] != ordinary[key]:
            raise RuntimeError('Unexpected protocol difference: ' + key)
    model, reference = copy.deepcopy(cfg.model), copy.deepcopy(baseline.model)
    loss = model.bbox_head.pop('loss_bbox')
    reference.bbox_head.pop('loss_bbox')
    if model != reference:
        raise RuntimeError('Model must match baseline except for regression loss')
    expected = dict(type='PaperBDLoss', alpha=3., agbb=True, min_size=1e-4,
                    loss_weight=1., square_ratio_threshold=1.1, delta=5.)
    if dict(loss) != expected:
        raise RuntimeError('Unexpected AGBB loss parameters')
    if cfg.load_from is not None or cfg.resume_from is not None or cfg.runner.max_epochs != 12:
        raise RuntimeError('AGBB must start fresh and train twelve epochs')
    proof = json.loads(PROOF.read_text())
    if proof['status'] != 'completed_short_controls' or proof['arms']['bd_agbb']['steps_completed'] != 500:
        raise RuntimeError('Expected completed AGBB 500-step controls')
    if hashes(proof['source_sha256']) != proof['source_sha256']:
        raise RuntimeError('Sources changed since AGBB controls')
    if proof['arms']['bd_agbb']['loss'] != expected:
        raise RuntimeError('Loss differs from verified AGBB controls')
    bd = json.loads(BD_REFERENCE.read_text())
    if bd['status'] != 'completed' or bd['epoch'] != 12 or bd['agbb']:
        raise RuntimeError('Expected completed ordinary-BD comparison')
    if hashes(bd['source_sha256']) != bd['source_sha256']:
        raise RuntimeError('Sources differ from ordinary-BD run')
    ids_root = ROOT / 'data/DIOR-R/ImageSets/Main'
    ids = {key: set((ids_root / (key + '.txt')).read_text().split())
           for key in ('train', 'val', 'test')}
    if [len(ids[key]) for key in ('train', 'val', 'test')] != [5862, 5863, 11738]:
        raise RuntimeError('Unexpected dataset split sizes')
    if ids['train'] & ids['val'] or (ids['train'] | ids['val']) & ids['test']:
        raise RuntimeError('Dataset splits overlap')
    health_data = copy.deepcopy(cfg.data.test)
    health_data.ann_file = 'data/DIOR-R/ImageSets/Main/train.txt'
    health_data.img_prefix = 'data/DIOR-R/JPEGImages-trainval/'
    health_data.test_mode = True
    imports = list(cfg.custom_imports['imports']) + ['mmrotate.core.rio_training_health']
    cfg.custom_imports = dict(imports=list(dict.fromkeys(imports)), allow_failed_imports=False)
    cfg.custom_hooks = [dict(type='RiOTrainingHealthHook', dataset_cfg=health_data,
                             images=128, seed=42, minimum_map=.05, minimum_fraction=.5,
                             diagnostic_iterations=[], priority='LOW')]
    cfg.work_dir = str(DIRECTORY)
    cfg.auto_resume = False
    sources = list(dict.fromkeys(list(proof['source_sha256']) +
                                ['mmrotate/core/rio_training_health.py', 'tools/train.py',
                                 'tools/test.py', 'tools/run_dior_r_rio.py',
                                 'tools/run_paper_agbb_bd_full.py']))
    summary = dict(protocol_matches=True, training_images=11725, test_images=11738,
                   dataset_splits_disjoint=True, epochs=12, fresh_initialization=True,
                   loss=expected, verified_agbb_steps=500,
                   ordinary_bd_mAP_percent=bd['mAP_percent'],
                   ordinary_bd_sources_unchanged=True,
                   threshold_note='1.1 is an explicit port setting; not specified numerically in the paper.')
    return cfg, sources, summary


def guard(state):
    if hashes(state['source_sha256']) != state['source_sha256']:
        raise RuntimeError('Sources changed during AGBB training/testing')
    if digest(FROZEN) != state['frozen_config_sha256']:
        raise RuntimeError('Frozen AGBB configuration changed')
    if hashes(state['reference_sha256']) != state['reference_sha256']:
        raise RuntimeError('Comparison result records changed')


def worker(lock_fd):
    os.fstat(lock_fd)
    state = json.loads((DIRECTORY / 'run.json').read_text())
    state.update(worker_pid=os.getpid(), started_at=HELPERS.now())
    try:
        guard(state)
        HELPERS.run_child([sys.executable, str(ROOT / 'tools/train.py'), str(FROZEN),
                           '--seed', '42', '--no-validate'],
                          DIRECTORY / 'train.log', state, 'training', lock_fd)
        final = DIRECTORY / 'epoch_12.pth'
        health = json.loads((DIRECTORY / 'training_health.json').read_text())
        if not final.is_file() or [row['epoch'] for row in health] != list(range(1, 13)):
            raise RuntimeError('Expected twelve epochs and the epoch-12 checkpoint')
        if any(row['status'] != 'passed' for row in health):
            raise RuntimeError('Training-image health did not pass')
        guard(state)
        state.update(training_completed_at=HELPERS.now(), checkpoint=str(final),
                     final_checkpoint_sha256=digest(final), training_health=health[-1])
        HELPERS.run_child([sys.executable, str(ROOT / 'tools/test.py'), str(FROZEN), str(final),
                           '--work-dir', str(DIRECTORY / 'test_final'),
                           '--out', str(DIRECTORY / 'test_final_predictions.pkl'),
                           '--eval', 'mAP', '--eval-options', 'iou_thr=0.5', 'use_07_metric=True'],
                          DIRECTORY / 'test.log', state, 'testing', lock_fd)
        files = list((DIRECTORY / 'test_final').glob('eval_*.json'))
        if len(files) != 1:
            raise RuntimeError('Expected one final test result')
        score = float(json.loads(files[0].read_text())['metric']['mAP'])
        if not math.isfinite(score):
            raise RuntimeError('Nonfinite test mAP')
        guard(state)
        if digest(final) != state['final_checkpoint_sha256']:
            raise RuntimeError('Final weights changed during testing')
        bd = float(json.loads(BD_REFERENCE.read_text())['test_mAP'])
        riou = float(json.loads(RIOU_REFERENCE.read_text())['metric']['mAP'])
        result = dict(status='completed', epoch=12, loss=state['loss'],
                      test_mAP=score, mAP_percent=100 * score,
                      trainval_images=11725, test_images=11738, iou_thr=.5,
                      use_07_metric=True, checkpoint=str(final),
                      checkpoint_sha256=state['final_checkpoint_sha256'],
                      evaluation_source=str(files[0]),
                      ordinary_bd_test_mAP=bd, difference_vs_bd_percentage_points=100 * (score - bd),
                      riou_test_mAP=riou, difference_vs_riou_percentage_points=100 * (score - riou),
                      finished_at=HELPERS.now(), source_sha256=state['source_sha256'])
        HELPERS.write_json(DIRECTORY / 'results.json', result)
        lines = ['# AGBB＋BD：完整 12 轮测试结果', '',
                 'Rotated FCOS / ResNet-50 / FPN；alpha=3、delta=5、近正方形阈值=1.1。',
                 '阈值 1.1 是适配设置，论文没有指定数值；实现是独立公式移植。',
                 'ImageNet 预训练主干开始；trainval=11,725；test=11,738；800×800；batch=2；SGD。', '',
                 '| 损失 | 固定第12轮 test mAP50 |', '| --- | ---: |',
                 f'| AGBB＋BD | {score * 100:.2f}% |',
                 f'| 普通 BD | {bd * 100:.2f}% |', f'| 原 Rotated IoU | {riou * 100:.2f}% |', '',
                 f'相对普通 BD：{100 * (score - bd):+.2f} 个百分点；'
                 f'相对原 Rotated IoU：{100 * (score - riou):+.2f} 个百分点。', '',
                 '单随机种子的历史对照，固定第12轮；不按 test 选权重。', '']
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
    cfg, sources, checks = configuration()
    if args.action == 'check':
        print(json.dumps(checks, indent=2))
        return
    with (ROOT / 'work_dirs/dior_r_trainval/server_queue.lock').open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('Another trainval experiment is running')
        if DIRECTORY.exists() and any(DIRECTORY.iterdir()):
            parser.error('Existing AGBB output will not be overwritten')
        FROZEN.parent.mkdir(parents=True, exist_ok=True)
        cfg.dump(str(FROZEN))
        for name in sources:
            target = DIRECTORY / 'snapshot/source' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / name).read_bytes())
        HELPERS.write_json(DIRECTORY / 'preflight.json', checks)
        HELPERS.write_json(DIRECTORY / 'run.json', dict(
            status='starting', created_at=HELPERS.now(), target_epoch=12,
            config=CONFIG, frozen_config=str(FROZEN), frozen_config_sha256=digest(FROZEN),
            source_sha256=hashes(sources),
            reference_sha256=hashes([str(BD_REFERENCE), str(RIOU_REFERENCE)]),
            seed=42, loss=checks['loss'], initialization='ImageNet backbone; fresh detector',
            load_from=None, resume_from=None, intermediate_test=False, final_test=True,
            checkpoint_selection='fixed epoch 12', threshold_note=checks['threshold_note'],
            health_guard='Finite metrics/gradient checks and 128 fixed training images per epoch.'))
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
