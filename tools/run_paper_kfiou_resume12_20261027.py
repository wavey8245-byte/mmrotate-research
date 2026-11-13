"""Complete pure KFIoU to cumulative epoch twelve, then evaluate full test."""
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
PREVIOUS = ROOT / 'work_dirs/dior_r_trainval/fcos_kfiou_probe1_20261021'
CHECKPOINT = PREVIOUS / 'epoch_1.pth'
DIRECTORY = ROOT / 'work_dirs/dior_r_trainval/fcos_kfiou_resume12_20261027'
FROZEN = DIRECTORY / 'snapshot/experiment.py'
FIRST_TEST = PREVIOUS / 'test_epoch_1/result.json'
RECHECK = PREVIOUS / 'fresh_process_recheck_v2/result.json'
EXPECTED_SHA = '918d4ccea7ab604902332d8f185686ab4fcc388006c6515177661f50e9b8cafe'
LOSS = dict(type='PaperKFIoUDecodedLoss', fun='ln', beta=1./9., eps=1e-6,
            min_size=1e-7, loss_weight=1.)
REFERENCES = [
    ('Rotated IoU', 'work_dirs/dior_r_trainval/fcos_riou_bs2_800/test_final/eval_20261006_092447.json'),
    ('KFIoU + EVCS-abs', 'work_dirs/dior_r_trainval/fcos_evcs_abs_resume12_retry_20261021/results.json'),
    ('ordinary BD', 'work_dirs/dior_r_trainval/fcos_paper_bd_recovery12_20261019/results.json'),
    ('AGBB + BD', 'work_dirs/dior_r_trainval/fcos_paper_agbb_bd_bs2_800_20261020/results.json'),
    ('DCAGPL resumed', 'work_dirs/dior_r_trainval/fcos_paper_dcagpl_resume12_20261021/results.json'),
    ('DCAGPL fresh', 'work_dirs/dior_r_trainval/fcos_paper_dcagpl_fresh12_20261025/results.json'),
    ('FrSIoU engineering', 'work_dirs/dior_r_trainval/frsiou_direct_resume12_20261022/results.json')]
spec = importlib.util.spec_from_file_location('kfiou_resume_helpers', ROOT / 'tools/run_dior_r_rio.py')
H = importlib.util.module_from_spec(spec)
spec.loader.exec_module(H)
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
    import torch
    previous = json.loads((PREVIOUS / 'run.json').read_text())
    if hashes(previous['source_sha256']) != previous['source_sha256']:
        raise RuntimeError('Historical experiment sources changed')
    test = json.loads(FIRST_TEST.read_text())
    check = json.loads(RECHECK.read_text())
    original = PREVIOUS / 'snapshot/experiment.py'
    if digest(original) != previous['frozen_config_sha256'] or digest(original) != check['frozen_config_sha256']:
        raise RuntimeError('Original frozen configuration changed')
    if digest(CHECKPOINT) != EXPECTED_SHA or test['checkpoint_sha256'] != EXPECTED_SHA or check['checkpoint_sha256'] != EXPECTED_SHA:
        raise RuntimeError('Epoch-one checkpoint hash mismatch')
    if test['status'] != 'completed' or test['checkpoint_epoch'] != 1 or test['test_images'] != 11738:
        raise RuntimeError('Epoch-one full-test evidence is incomplete')
    if check['status'] != 'passed' or not check['weights_finite'] or not check['same_original_hook'] or not check['same_threshold'] == .05:
        raise RuntimeError('Independent unchanged health recheck did not pass')
    if check['checkpoint_epoch'] != 1 or check['checkpoint_iteration'] != 5863 or check['training_health'][-1]['status'] != 'passed':
        raise RuntimeError('Unexpected health recheck epoch or outcome')
    cfg = mmcv.Config.fromfile(str(original))
    baseline = mmcv.Config.fromfile(str(ROOT / 'configs/dior_r/server_trainval/riou.py'))
    evcs = mmcv.Config.fromfile(str(ROOT / 'work_dirs/dior_r_trainval/fcos_evcs_abs_resume12_retry_20261021/snapshot/experiment.py'))
    for key in ('data', 'optimizer', 'optimizer_config', 'lr_config', 'seed', 'checkpoint_config'):
        if cfg[key] != baseline[key] or cfg[key] != evcs[key]:
            raise RuntimeError('Unexpected protocol difference: ' + key)
    model, reference = copy.deepcopy(cfg.model), copy.deepcopy(baseline.model)
    loss = model.bbox_head.pop('loss_bbox')
    reference.bbox_head.pop('loss_bbox')
    if model != reference or dict(loss) != LOSS or dict(test['loss']) != LOSS:
        raise RuntimeError('Expected unchanged pure-KFIoU objective and baseline FCOS')
    evcs_model = copy.deepcopy(evcs.model)
    evcs_loss = dict(evcs_model.bbox_head.pop('loss_bbox'))
    if evcs_model != model or evcs_loss.pop('type') != 'EVCSAbsCorrectionLoss' or evcs_loss.pop('evcs_weight') != .006:
        raise RuntimeError('Expected matched EVCS-abs comparison model')
    pure = dict(LOSS)
    pure.pop('type')
    if evcs_loss != pure:
        raise RuntimeError('KFIoU base differs from EVCS-abs base')
    checkpoint = torch.load(str(CHECKPOINT), map_location='cpu')
    meta, optimizer = checkpoint['meta'], checkpoint.get('optimizer', {})
    if meta.get('epoch') != 1 or meta.get('iter') != 5863 or not optimizer.get('state') or not optimizer.get('param_groups'):
        raise RuntimeError('Missing native epoch-one counters or optimizer state')
    for group in optimizer['param_groups']:
        if not math.isclose(group['lr'], .0025) or not math.isclose(group['momentum'], .9) or not math.isclose(group['weight_decay'], .0001):
            raise RuntimeError('Unexpected saved SGD settings')
    if any(not torch.isfinite(t).all() for t in checkpoint['state_dict'].values()):
        raise RuntimeError('Nonfinite checkpoint weights')
    del checkpoint
    split_root = ROOT / 'data/DIOR-R/ImageSets/Main'
    splits = {name: set((split_root / (name + '.txt')).read_text().split()) for name in ('train', 'val', 'test')}
    if [len(splits[name]) for name in ('train', 'val', 'test')] != [5862, 5863, 11738]:
        raise RuntimeError('Unexpected DIOR split sizes')
    if splits['train'] & splits['val'] or (splits['train'] | splits['val']) & splits['test']:
        raise RuntimeError('Dataset split overlap')
    cfg.runner.max_epochs = 12
    cfg.work_dir = str(DIRECTORY)
    cfg.load_from = None
    cfg.resume_from = str(CHECKPOINT)
    cfg.auto_resume = False
    cfg.custom_hooks[0]['diagnostic_iterations'] = []
    sources = list(dict.fromkeys(list(previous['source_sha256']) + ['tools/run_paper_kfiou_resume12_20261027.py']))
    evidence = [str(FIRST_TEST), str(RECHECK), str(PREVIOUS / 'run.json'), str(original)] + [rel for label, rel in REFERENCES]
    proof = dict(checkpoint_epoch=1, checkpoint_iteration=5863, checkpoint_sha256=EXPECTED_SHA,
                 optimizer_state_present=True, original_sources_unchanged=True,
                 original_training_status_preserved=previous['status'], original_online_health_passed=False,
                 unchanged_fresh_process_recheck_passed=True, recheck_source=str(RECHECK),
                 first_epoch_test_mAP=test['test_mAP'], first_epoch_test_source=str(FIRST_TEST),
                 matched_evcs_abs_KFIoU_base=True, matched_baseline_protocol=True,
                 trainval_images=11725, test_images=11738, dataset_splits_disjoint=True,
                 seed=42, cumulative_epochs=12, remaining_epochs=11, evcs_added=False,
                 rng_state_note='Native checkpoint restores model, optimizer, epoch and iteration, but not complete RNG/worker states; no bitwise uninterrupted-equivalence claim.')
    return cfg, sources, evidence, proof


def guard(state):
    if hashes(state['source_sha256']) != state['source_sha256']:
        raise RuntimeError('Experiment sources changed during execution')
    if digest(FROZEN) != state['frozen_config_sha256'] or digest(CHECKPOINT) != state['resume_checkpoint_sha256']:
        raise RuntimeError('Frozen configuration or source checkpoint changed')
    if hashes(state['evidence_sha256']) != state['evidence_sha256']:
        raise RuntimeError('Historical evidence or comparison record changed')


def worker(lock_fd):
    os.fstat(lock_fd)
    state = json.loads((DIRECTORY / 'run.json').read_text())
    state.update(worker_pid=os.getpid(), started_at=H.now())
    try:
        guard(state)
        H.run_child([sys.executable, str(ROOT / 'tools/train.py'), str(FROZEN),
                     '--resume-from', str(CHECKPOINT), '--seed', '42', '--no-validate'],
                    DIRECTORY / 'train.log', state, 'training', lock_fd)
        final = DIRECTORY / 'epoch_12.pth'
        health = json.loads((DIRECTORY / 'training_health.json').read_text())
        if not final.is_file() or [item['epoch'] for item in health] != list(range(2, 13)):
            raise RuntimeError('Expected native epoch twelve and eleven resumed epoch checks')
        if any(item['status'] != 'passed' for item in health):
            raise RuntimeError('Training-image health check failed')
        import torch
        checkpoint = torch.load(str(final), map_location='cpu')
        if checkpoint['meta'].get('epoch') != 12 or checkpoint['meta'].get('iter') != 70356:
            raise RuntimeError('Unexpected final native epoch/iteration')
        if not checkpoint.get('optimizer', {}).get('state'):
            raise RuntimeError('Final checkpoint lacks SGD state')
        if any(not torch.isfinite(t).all() for t in checkpoint['state_dict'].values()):
            raise RuntimeError('Nonfinite final weights')
        del checkpoint
        guard(state)
        state.update(training_completed_at=H.now(), checkpoint=str(final),
                     checkpoint_sha256=digest(final), training_health=health)
        H.run_child([sys.executable, str(ROOT / 'tools/test.py'), str(FROZEN), str(final),
                     '--work-dir', str(DIRECTORY / 'test_final'), '--out', str(DIRECTORY / 'test_final_predictions.pkl'),
                     '--eval', 'mAP', '--eval-options', 'iou_thr=0.5', 'use_07_metric=True'],
                    DIRECTORY / 'test.log', state, 'testing', lock_fd)
        files = list((DIRECTORY / 'test_final').glob('eval_*.json'))
        if len(files) != 1:
            raise RuntimeError('Expected exactly one full-test evaluation')
        value = float(json.loads(files[0].read_text())['metric']['mAP'])
        if not math.isfinite(value):
            raise RuntimeError('Nonfinite final mAP')
        guard(state)
        if digest(final) != state['checkpoint_sha256']:
            raise RuntimeError('Final weights changed during testing')
        comparisons = []
        for label, rel in REFERENCES:
            record = json.loads((ROOT / rel).read_text())
            reference_score = float(record['test_mAP'] if 'test_mAP' in record else record['metric']['mAP'])
            comparisons.append(dict(method=label, test_mAP=reference_score,
                                    kfiou_minus_reference_percentage_points=100*(value-reference_score), source=str(ROOT/rel)))
        evcs_score = next(item['test_mAP'] for item in comparisons if item['method']=='KFIoU + EVCS-abs')
        result = dict(status='completed', method='pure KFIoU', epoch=12, seed=42,
                      loss=LOSS, evcs_added=False, test_mAP=value, mAP_percent=100*value,
                      trainval_images=11725, test_images=11738, iou_thr=.5, use_07_metric=True,
                      checkpoint=str(final), checkpoint_sha256=state['checkpoint_sha256'],
                      resumed_from=str(CHECKPOINT), resume_epoch=1, resume_iteration=5863,
                      evaluation_source=str(files[0]), comparisons=comparisons,
                      evcs_abs_test_mAP=evcs_score, evcs_abs_minus_kfiou_percentage_points=100*(evcs_score-value),
                      first_epoch_test_mAP=state['resume_proof']['first_epoch_test_mAP'],
                      original_training_status_preserved=state['resume_proof']['original_training_status_preserved'],
                      rng_state_note=state['resume_proof']['rng_state_note'], source_sha256=state['source_sha256'],
                      center_adaptation='FCOS decoded center residuals in stride-normalized training coordinates', finished_at=H.now())
        H.write_json(DIRECTORY/'results.json',result)
        lines=['# 纯 KFIoU：累计12轮完整测试结果','',
               'Rotated FCOS / ResNet-50 / FPN；与 EVCS-abs 使用同一 KFIoU 基项，仅去除 EVCS 附加项。',
               'trainval=11725；test=11738；800×800；batch=2；workers=2；seed=42；原 SGD 与 step=[8,11]。',
               '原第1轮权重经独立进程复核通过后，从该检查点恢复模型、优化器和计数，续训第2～12轮。',
               '原在线健康检查失败记录保留；不声称恢复完整随机数或与不中断训练逐位一致。','',
               '| 实验 | 第12轮完整测试 mAP50（%） | KFIoU相对差值（百分点） |','|---|---:|---:|',
               f'| 纯 KFIoU | {100*value:.6f} | 0 |']
        lines += [f"| {item['method']} | {100*item['test_mAP']:.6f} | {item['kfiou_minus_reference_percentage_points']:+.6f} |" for item in comparisons]
        lines += ['',f'EVCS-abs 减去纯KFIoU：{100*(evcs_score-value):+.6f} 个百分点。',
                  '以上为单次实验；固定epoch_12，不按测试集选择权重，不能单次推断稳定提升。','']
        (DIRECTORY/'results.md').write_text('\n'.join(lines),encoding='utf-8')
        state.update(status='completed',finished_at=H.now(),result=result)
        H.write_json(DIRECTORY/'run.json',state)
    except Exception as error:
        state.update(status='failed',finished_at=H.now(),error=str(error))
        H.write_json(DIRECTORY/'run.json',state)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('check','start','worker','status'))
    parser.add_argument('--lock-fd',type=int)
    args=parser.parse_args()
    os.chdir(ROOT)
    if args.action=='worker':
        if args.lock_fd is None:parser.error('An inherited queue lock is required')
        worker(args.lock_fd)
        return
    if args.action=='status':
        print((DIRECTORY/'run.json').read_text())
        return
    cfg,sources,evidence,proof=configuration()
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    (torch.ones(1,device='cuda')+1).cpu()
    if args.action=='check':
        print(json.dumps(proof,indent=2))
        return
    with (ROOT/'work_dirs/dior_r_trainval/server_queue.lock').open('a') as lock:
        try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:parser.error('Another trainval experiment is running')
        if DIRECTORY.exists() and any(DIRECTORY.iterdir()):parser.error('Existing experiment will not be overwritten')
        FROZEN.parent.mkdir(parents=True,exist_ok=True)
        cfg.dump(str(FROZEN))
        for name in sources:
            relative=(ROOT/name).resolve().relative_to(ROOT)
            path=DIRECTORY/'snapshot/source'/relative
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes((ROOT/name).read_bytes())
        H.write_json(DIRECTORY/'resume_check.json',proof)
        H.write_json(DIRECTORY/'run.json',dict(
            status='starting',created_at=H.now(),target_epoch=12,completed_epochs_before_resume=1,remaining_epochs=11,
            seed=42,loss=LOSS,evcs_added=False,resume_from=str(CHECKPOINT),resume_epoch=1,resume_iteration=5863,
            resume_checkpoint_sha256=EXPECTED_SHA,optimizer_state_present=True,
            frozen_config=str(FROZEN),frozen_config_sha256=digest(FROZEN),source_sha256=hashes(sources),
            evidence_sha256=hashes(evidence),resume_proof=proof,auto_resume=False,
            final_test=True,intermediate_test=False,checkpoint_selection='fixed epoch 12',
            allocator='max_split_size_mb:128',health_guard='Original unchanged training-image/finite-metric guards; no test-based selection.'))
        env=os.environ.copy()
        env.update(PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='0',PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:128')
        env['LD_LIBRARY_PATH']='/usr/lib/wsl/lib'+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
        with (DIRECTORY/'worker.log').open('w') as log:
            process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'worker','--lock-fd',str(lock.fileno())],
                                     cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                     start_new_session=True,pass_fds=(lock.fileno(),))
        (DIRECTORY/'worker.pid').write_text(str(process.pid)+'\n')
        print(json.dumps(dict(worker_pid=process.pid,work_dir=str(DIRECTORY),target_epoch=12,remaining_epochs=11,final_test=True)))


if __name__=='__main__':
    main()
