"""One standard EVCS-abs correction epoch on FCOS, then full-test evaluation."""
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
CONFIG = 'configs/dior_r/server_trainval/paper_evcs_abs_correction.py'
REFERENCE_CONFIG = 'configs/dior_r/server_trainval/riou.py'
DIRECTORY = ROOT / 'work_dirs/dior_r_trainval/fcos_evcs_abs_probe1_20261021'
FROZEN = DIRECTORY / 'snapshot/experiment.py'
CORE_SOURCES = ('mmrotate/models/losses/paper_rotated_losses.py',
           'mmrotate/models/losses/rotated_iou_loss.py',
           'mmrotate/models/dense_heads/rotated_fcos_head.py',
           'mmrotate/core/rio_training_health.py', 'mmrotate/datasets/dior.py',
           'tools/train.py', 'tools/run_dior_r_rio.py',
           'tools/run_paper_evcs_abs_epoch.py', 'tools/test.py',
           'mmrotate/models/losses/paper_evcs_loss.py',
           'mmrotate/models/losses/paper_evcs_abs_loss.py',
           'mmrotate/models/losses/kf_iou_loss.py')
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
    if cfg.model.bbox_head.loss_bbox.type != 'EVCSAbsCorrectionLoss':
        raise RuntimeError('Expected explicitly corrected EVCS loss')
    expected_loss = dict(type='EVCSAbsCorrectionLoss', fun='ln', beta=1./9.,
                         eps=1e-6, min_size=1e-7, loss_weight=1., evcs_weight=.006)
    if dict(cfg.model.bbox_head.loss_bbox) != expected_loss:
        raise RuntimeError('Unexpected EVCS-abs correction parameters')
    proof_path = ROOT / 'work_dirs/dior_r_trainval/paper_evcs_abs_short_500_20261021/report.json'
    proof = json.loads(proof_path.read_text())
    if proof['status'] != 'completed_short_controls' or proof['arms']['evcs_abs']['steps_completed'] != 500:
        raise RuntimeError('Completed 500-step EVCS-abs check required')
    if HELPERS.hashes(proof['source_sha256']) != proof['source_sha256']:
        raise RuntimeError('Short-check sources changed')
    if proof['arms']['evcs_abs']['loss'] != dict(cfg.model.bbox_head.loss_bbox):
        raise RuntimeError('EVCS-abs specification changed since the short check')
    if not proof.get('sources_unchanged') or not proof.get('initial_state_matches_control'):
        raise RuntimeError('Short-run provenance checks missing')
    if not proof['arms']['evcs_abs'].get('identical_augmented_batches'):
        raise RuntimeError('Short-run batches did not match the KFIoU control')
    split_root = ROOT / 'data/DIOR-R/ImageSets/Main'
    split_ids = {name: set((split_root / (name + '.txt')).read_text().split())
                 for name in ('train', 'val', 'test')}
    if [len(split_ids[name]) for name in ('train', 'val', 'test')] != [5862, 5863, 11738]:
        raise RuntimeError('Unexpected dataset split sizes')
    if split_ids['train'] & split_ids['val'] or (split_ids['train'] | split_ids['val']) & split_ids['test']:
        raise RuntimeError('Dataset splits overlap')
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


def guard(state):
    if HELPERS.hashes(SOURCES) != state['source_sha256']:
        raise RuntimeError('One-epoch experiment sources changed')
    if HELPERS.hashes([str(FROZEN)])[str(FROZEN)] != state['frozen_config_sha256']:
        raise RuntimeError('Frozen one-epoch configuration changed')


def worker(lock_fd):
    import math
    import torch
    os.fstat(lock_fd)
    state = json.loads((DIRECTORY / 'run.json').read_text())
    state.update(worker_pid=os.getpid(), started_at=HELPERS.now())
    try:
        guard(state)
        HELPERS.run_child([sys.executable, str(ROOT / 'tools/train.py'), str(FROZEN),
                           '--seed', '42', '--no-validate'],
                          DIRECTORY / 'train.log', state, 'training', lock_fd)
        guard(state)
        checkpoint = DIRECTORY / 'epoch_1.pth'
        health = json.loads((DIRECTORY / 'training_health.json').read_text())
        if not checkpoint.is_file() or len(health) != 1 or health[0]['epoch'] != 1 or health[0]['status'] != 'passed':
            raise RuntimeError('Missing standard epoch-1 checkpoint or failed training-image check')
        saved = torch.load(str(checkpoint), map_location='cpu')
        meta, optimizer = saved['meta'], saved.get('optimizer', {})
        if meta.get('epoch') != 1 or meta.get('iter') != 5863 or not optimizer.get('state'):
            raise RuntimeError('Expected epoch 1, iteration 5863 and complete SGD state')
        state.update(training_completed_at=HELPERS.now(), training_health=health[-1],
                     checkpoint=str(checkpoint), checkpoint_epoch=meta['epoch'],
                     checkpoint_iteration=meta['iter'], optimizer_state_present=True,
                     checkpoint_sha256=HELPERS.hashes([str(checkpoint)])[str(checkpoint)])
        del saved, optimizer
        test_dir = DIRECTORY / 'test_epoch_1'
        test_dir.mkdir(exist_ok=False)
        HELPERS.run_child([sys.executable, str(ROOT / 'tools/test.py'), str(FROZEN),
                           str(checkpoint), '--work-dir', str(test_dir),
                           '--out', str(test_dir / 'predictions.pkl'), '--eval', 'mAP',
                           '--eval-options', 'iou_thr=0.5', 'use_07_metric=True'],
                          test_dir / 'test.log', state, 'testing', lock_fd)
        evaluations = list(test_dir.glob('eval_*.json'))
        if len(evaluations) != 1:
            raise RuntimeError('Expected exactly one full-test evaluation')
        metric = json.loads(evaluations[0].read_text())['metric']
        score = float(metric['mAP'])
        if not math.isfinite(score):
            raise RuntimeError('Nonfinite full-test mAP')
        guard(state)
        if HELPERS.hashes([str(checkpoint)])[str(checkpoint)] != state['checkpoint_sha256']:
            raise RuntimeError('Epoch-1 checkpoint changed during testing')
        comparisons = []
        for label, relative in [
            ('ordinary BD', 'work_dirs/dior_r_trainval/fcos_paper_bd_probe1_20261019/test_epoch_1/result.json'),
            ('DCAGPL', 'work_dirs/dior_r_trainval/fcos_paper_dcagpl_probe1_20261020/test_epoch_1/result.json')]:
            record = json.loads((ROOT / relative).read_text())
            if record['status'] != 'completed' or record['checkpoint_epoch'] != 1 or record['test_images'] != 11738:
                raise RuntimeError('Unexpected historical epoch-1 reference: ' + label)
            ref_score = float(record['metric']['mAP'])
            comparisons.append(dict(method=label, epoch=1, test_mAP=ref_score,
                                    difference_percentage_points=100 * (score - ref_score),
                                    source=str(ROOT / relative)))
        riou_path = ROOT / 'work_dirs/dior_r_trainval/fcos_riou_bs2_800/20261006_000534.log.json'
        riou_rows = [json.loads(line) for line in riou_path.read_text().splitlines()]
        riou_rows = [row for row in riou_rows if row.get('mode') == 'val' and row.get('epoch') == 1]
        if len(riou_rows) != 1 or riou_rows[0]['iter'] != 11738:
            raise RuntimeError('Expected one full-test RIoU epoch-1 record')
        ref_score = float(riou_rows[0]['mAP'])
        comparisons.append(dict(method='Rotated IoU', epoch=1, test_mAP=ref_score,
                                difference_percentage_points=100 * (score - ref_score),
                                source=str(riou_path), historical_log_precision=5))
        result = dict(status='completed', finished_at=HELPERS.now(), epoch=1,
                      checkpoint_epoch=1, checkpoint=str(checkpoint),
                      checkpoint_sha256=state['checkpoint_sha256'], config=str(FROZEN),
                      loss=state['loss'], correction='absolute length difference in exponent',
                      author_formula=False, seed=42, trainval_images=11725,
                      test_images=11738, iou_thr=.5, use_07_metric=True,
                      metric=metric, test_mAP=score, mAP_percent=100 * score,
                      evaluation_source=str(evaluations[0]), comparisons=comparisons,
                      source_sha256=state['source_sha256'],
                      scope='Full test at fixed epoch 1; engineering-corrected EVCS, not raw paper EVCS or full CRL.')
        HELPERS.write_json(test_dir / 'result.json', result)
        HELPERS.write_json(DIRECTORY / 'results.json', result)
        summary = ['# EVCS 绝对值修正版：标准第 1 轮完整测试', '',
                   '独立修正方法：KFIoU＋0.006×EVCS-abs，不是论文印刷原式或完整 CRL。',
                   'Rotated FCOS / ResNet-50 / FPN；ImageNet 初始化；DIOR-R train＋val=11,725，test=11,738。',
                   '800×800，batch=2，workers=2，seed=42，原 SGD、预热与梯度裁剪。',
                   '固定使用标准第 1 轮权重；IoU=0.5，VOC 2007 的 11 点 AP。', '',
                   f'完整测试集 mAP50：**{100 * score:.4f}%**。', '',
                   '| Method | epoch 1 test mAP50 | EVCS-abs difference (pp) |',
                   '|---|---:|---:|', f'| EVCS-abs | {100 * score:.4f}% | 0 |']
        for row in comparisons:
            summary.append(f"| {row['method']} | {100 * row['test_mAP']:.4f}% | {row['difference_percentage_points']:+.4f} |")
        summary.extend(['', 'Rotated IoU 为历史日志保留 5 位小数的 mAP；其余来自原生评估 JSON。',
                        '仅比较同为第 1 轮的完整测试结果，不能据此推断第 12 轮排名。',
                        'EVCS 使用当前 FCOS 的 FPN 步长归一化训练坐标；KFIoU 为 ln 形式，中心为 FCOS 解码中心残差。',
                        '训练图片诊断与完整 test 分别记录，未自动延长至 12 轮。', ''])
        (DIRECTORY / 'results.md').write_text('\n'.join(summary), encoding='utf-8')
        (test_dir / 'results.md').write_text('\n'.join(summary), encoding='utf-8')
        state.update(status='completed_one_epoch_and_test', finished_at=HELPERS.now(), result=result)
        HELPERS.write_json(DIRECTORY / 'run.json', state)
    except Exception as error:
        state.update(status='failed', error=str(error), finished_at=HELPERS.now())
        health_path = DIRECTORY / 'training_health.json'
        if health_path.is_file():
            records = json.loads(health_path.read_text())
            if records:
                state['training_health'] = records[-1]
        HELPERS.write_json(DIRECTORY / 'run.json', state)
        raise


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
        print('One standard epoch from ImageNet; fixed epoch-1 full test; explicitly abs-corrected EVCS.')
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
            test_evaluation=True, automatic_continuation=False,
            loss=dict(cfg.model.bbox_head.loss_bbox), correction='abs length difference',
            author_formula=False, initialization='ImageNet backbone; new FCOS head',
            checkpoint_selection='fixed epoch 1', trainval_images=11725, test_images=11738,
            purpose='One standard-runner epoch of explicitly corrected EVCS on FCOS, '
                    'then full test; not the published raw EVCS or full CRL framework.'))
        env = os.environ.copy()
        env.update(PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='0')
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
