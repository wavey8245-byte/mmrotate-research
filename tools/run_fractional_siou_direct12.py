"""Direct unchanged FrSIoU continuation from epoch 2 to total epoch 12.

Direct FrSIoU-only continuation; resume matching native
model/SGD counters and (for FrSIoU) the verified history sidecar. Fixed epoch-12
full test per arm. Never modify the running two-epoch experiment's sources.
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

ROOT=Path(__file__).resolve().parents[1]
PREVIOUS=ROOT/'work_dirs/dior_r_trainval/frsiou_standard2_controls_20261022'
DIRECTORY=ROOT/'work_dirs/dior_r_trainval/frsiou_direct_resume12_20261022'
ARMS=('frsiou_box_tangent',)
SPEC=importlib.util.spec_from_file_location('frsiou_extend_helpers',ROOT/'tools/run_dior_r_rio.py')
HELPERS=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPERS)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def hashes(paths):
    return {str(p):digest(ROOT/p) for p in paths}


def previous_state():
    state=json.loads((PREVIOUS/'run.json').read_text())
    if hashes(state['source_sha256'])!=state['source_sha256']:
        raise RuntimeError('Sources differ from the two-epoch controls')
    for name in ARMS:
        arm=state['arms'][name]
        if digest(Path(arm['frozen_config']))!=arm['frozen_config_sha256']:
            raise RuntimeError('A two-epoch frozen configuration changed')
    return state


def configuration(name,previous):
    import mmcv
    cfg=mmcv.Config.fromfile(previous['arms'][name]['frozen_config'])
    if cfg.runner.max_epochs!=2 or cfg.load_from is not None or cfg.resume_from is not None:
        raise RuntimeError('Expected native fresh two-epoch source configuration')
    if dict(cfg.model.bbox_head.loss_bbox)!=previous['arms'][name]['loss']:
        raise RuntimeError('Loss differs from two-epoch run')
    cfg.runner.max_epochs=12
    cfg.work_dir=str(DIRECTORY/name)
    cfg.load_from=None
    cfg.resume_from=str(PREVIOUS/name/'epoch_2.pth')
    cfg.auto_resume=False
    count=0
    for hook in cfg.custom_hooks:
        if hook['type']=='FrSIoUMemoryHook':
            hook['resume_checkpoint']=cfg.resume_from
            count+=1
    if count!=(1 if name=='frsiou_box_tangent' else 0):
        raise RuntimeError('Unexpected fractional history hook configuration')
    if cfg.lr_config.get('step')!=[8,11]:
        raise RuntimeError('Expected unchanged original epoch-8/11 SGD schedule')
    return cfg


def check_resume(name,cfg):
    import torch
    source=PREVIOUS/name
    result=json.loads((source/'results.json').read_text())
    state=json.loads((source/'run.json').read_text())
    health=json.loads((source/'training_health.json').read_text())
    if result['status']!='completed' or result['epoch']!=2 or result['test_images']!=11738 or state['status']!='completed':
        raise RuntimeError('Source arm is not a completed standard two-epoch experiment: '+name)
    if [x['epoch'] for x in health]!=[1,2] or any(x['status']!='passed' for x in health):
        raise RuntimeError('Source epoch health checks did not pass')
    cp=source/'epoch_2.pth'
    h=digest(cp)
    if h!=result['checkpoint_sha256']:
        raise RuntimeError('Source native epoch-2 checkpoint changed')
    saved=torch.load(str(cp),map_location='cpu')
    meta=saved.get('meta',{})
    opt=saved.get('optimizer',{})
    if meta.get('epoch')!=2 or meta.get('iter')!=11726 or not opt.get('state'):
        raise RuntimeError('Need native epoch2/iter11726 and complete optimizer state')
    if any(not math.isclose(x['lr'],.0025) or not math.isclose(x['momentum'],.9) for x in opt['param_groups']):
        raise RuntimeError('Unexpected epoch-2 optimizer settings')
    proof=dict(epoch=2,iteration=11726,optimizer_state_present=True,checkpoint=str(cp),checkpoint_sha256=h,
        source_result=str(source/'results.json'),source_result_sha256=digest(source/'results.json'),
        source_test_mAP=result['test_mAP'],rng_note='Native model/optimizer/counters resume; no complete RNG/worker snapshot.')
    del saved,opt,meta
    if name=='frsiou_box_tangent':
        from mmrotate.models.builder import build_loss
        side=cp.with_suffix('.frsiou_history.pth')
        cache=torch.load(str(side),map_location='cpu')
        if cache['checkpoint_sha256']!=h or cache['epoch']!=2 or cache['iteration']!=11726:
            raise RuntimeError('Source fractional history sidecar does not match checkpoint')
        loss=build_loss(cfg.model.bbox_head.loss_bbox)
        loss.import_history(cache['history'])
        if not loss.history or not loss.total_history_hits:
            raise RuntimeError('Fractional history missing or unused')
        proof.update(history_sidecar=str(side),history_sidecar_sha256=digest(side),
            history_signature=loss.signature(),cached_groups=len(loss.history),
            history_committed_updates=loss.committed_updates,total_history_hits=loss.total_history_hits)
        del cache,loss
    return proof


def guard(state):
    if hashes(state['source_sha256'])!=state['source_sha256']:
        raise RuntimeError('A protected continuation source changed')
    for arm in state['arms'].values():
        if digest(Path(arm['frozen_config']))!=arm['frozen_config_sha256']:
            raise RuntimeError('A frozen continuation configuration changed')
        proof=arm.get('resume_proof')
        if proof:
            for path_key,hash_key in [('checkpoint','checkpoint_sha256'),('source_result','source_result_sha256'),
                                      ('history_sidecar','history_sidecar_sha256')]:
                if path_key in proof and digest(Path(proof[path_key]))!=proof[hash_key]:
                    raise RuntimeError('A continuation source checkpoint/history/result changed')


def worker(lock_fd):
    os.fstat(lock_fd)
    state=json.loads((DIRECTORY/'run.json').read_text())
    state.update(status='queued',worker_pid=os.getpid(),waiting_since=HELPERS.now())
    HELPERS.write_json(DIRECTORY/'run.json',state)
    current=None
    try:
        # Blocking only in this detached worker; current training is untouched.
        fcntl.flock(lock_fd,fcntl.LOCK_EX)
        state.update(status='starting',queue_acquired_at=HELPERS.now())
        HELPERS.write_json(DIRECTORY/'run.json',state)
        previous=previous_state()
        if previous['arms']['frsiou_box_tangent']['status']!='completed':
            raise RuntimeError('FrSIoU epoch2 source must be complete; SIoU control is not a prerequisite')
        import mmcv
        for name in ARMS:
            current=name
            arm=state['arms'][name]
            cfg=mmcv.Config.fromfile(arm['frozen_config'])
            proof=check_resume(name,cfg)
            arm['resume_proof']=proof
            HELPERS.write_json(DIRECTORY/name/'resume_check.json',proof)
            guard(state)
            state.update(status='running',active_arm=name)
            HELPERS.write_json(DIRECTORY/'run.json',state)
            arm_dir=DIRECTORY/name
            HELPERS.DIRECTORY=arm_dir
            HELPERS.run_child([sys.executable,str(ROOT/'tools/train.py'),arm['frozen_config'],
                '--resume-from',proof['checkpoint'],'--seed','42','--no-validate'],
                arm_dir/'train.log',arm,'training',lock_fd)
            import torch
            cp=arm_dir/'epoch_12.pth'
            saved=torch.load(str(cp),map_location='cpu')
            if saved['meta'].get('epoch')!=12 or saved['meta'].get('iter')!=70356 or not saved.get('optimizer',{}).get('state'):
                raise RuntimeError('Final checkpoint lacks native epoch12/iter70356/optimizer state')
            del saved
            health=json.loads((arm_dir/'training_health.json').read_text())
            if [x['epoch'] for x in health]!=list(range(3,13)) or any(x['status']!='passed' for x in health):
                raise RuntimeError('Expected ten passed resumed-epoch training-image checks')
            h=digest(cp)
            arm.update(training_completed_at=HELPERS.now(),checkpoint=str(cp),checkpoint_sha256=h,training_health=health)
            if name=='frsiou_box_tangent':
                side=cp.with_suffix('.frsiou_history.pth')
                cache=torch.load(str(side),map_location='cpu')
                if cache['epoch']!=12 or cache['iteration']!=70356 or cache['checkpoint_sha256']!=h:
                    raise RuntimeError('Final history/model checkpoint mismatch')
                if cache['history']['committed_updates']<=proof['history_committed_updates'] or cache['history']['total_history_hits']<=proof['total_history_hits']:
                    raise RuntimeError('Resumed history did not advance')
                arm.update(final_history_sidecar=str(side),final_history_sidecar_sha256=digest(side),
                    final_cached_groups=len(cache['history']['records']),final_history_hits=cache['history']['total_history_hits'])
                del cache
            guard(state)
            HELPERS.write_json(DIRECTORY/'run.json',state)
            HELPERS.run_child([sys.executable,str(ROOT/'tools/test.py'),arm['frozen_config'],str(cp),
                '--work-dir',str(arm_dir/'test_final'),'--out',str(arm_dir/'test_predictions.pkl'),
                '--eval','mAP','--eval-options','iou_thr=0.5','use_07_metric=True'],
                arm_dir/'test.log',arm,'testing',lock_fd)
            evaluations=list((arm_dir/'test_final').glob('eval_*.json'))
            if len(evaluations)!=1:
                raise RuntimeError('Expected one full final evaluation')
            score=float(json.loads(evaluations[0].read_text())['metric']['mAP'])
            if not math.isfinite(score) or digest(cp)!=h:
                raise RuntimeError('Nonfinite score or changed final checkpoint')
            guard(state)
            result=dict(status='completed',method=name,epoch=12,test_mAP=score,mAP_percent=100*score,
                trainval_images=11725,test_images=11738,iou_thr=.5,use_07_metric=True,
                checkpoint=str(cp),checkpoint_sha256=h,resumed_from=proof['checkpoint'],
                loss=arm['loss'],author_reproduction=False,evaluation_source=str(evaluations[0]),
                resume_proof=proof,finished_at=HELPERS.now())
            baseline_path=ROOT/'work_dirs/dior_r_trainval/fcos_riou_bs2_800/test_final/eval_20261006_092447.json'
            baseline=float(json.loads(baseline_path.read_text())['metric']['mAP'])
            result.update(riou_baseline_mAP=baseline,difference_vs_riou_percentage_points=100*(score-baseline))
            HELPERS.write_json(arm_dir/'results.json',result)
            arm.update(status='completed',result=result,finished_at=HELPERS.now())
            HELPERS.write_json(arm_dir/'run.json',arm)
            HELPERS.write_json(DIRECTORY/'run.json',state)
        fr=state['arms']['frsiou_box_tangent']['result']['test_mAP']
        result=dict(state['arms']['frsiou_box_tangent']['result'])
        comparisons=[]
        for label,relative,key in [
            ('Rotated IoU','work_dirs/dior_r_trainval/fcos_riou_bs2_800/test_final/eval_20261006_092447.json','metric'),
            ('ordinary BD','work_dirs/dior_r_trainval/fcos_paper_bd_recovery12_20261019/results.json','test_mAP'),
            ('AGBB + BD','work_dirs/dior_r_trainval/fcos_paper_agbb_bd_bs2_800_20261020/results.json','test_mAP'),
            ('DCAGPL','work_dirs/dior_r_trainval/fcos_paper_dcagpl_resume12_20261021/results.json','test_mAP'),
            ('KFIoU + EVCS-abs','work_dirs/dior_r_trainval/fcos_evcs_abs_resume12_retry_20261021/results.json','test_mAP')]:
            ref=json.loads((ROOT/relative).read_text())
            score=float(ref['metric']['mAP'] if key=='metric' else ref[key])
            comparisons.append(dict(method=label,test_mAP=score,difference_percentage_points=100*(fr-score),source=str(ROOT/relative)))
        result['comparisons']=comparisons
        HELPERS.write_json(DIRECTORY/'results.json',result)
        lines=['# FrSIoU 工程历史版：固定第12轮完整测试','',
            '直接从标准第2轮恢复模型、SGD、计数和匹配历史侧文件，续训第3-12轮。',
            'FrSIoU使用历史框切线代理，不是作者代码严格复现。',
            '参数、batch、图片尺寸、SGD调度保持不变；原生检查点缺少完整RNG/工作进程状态。','',
            '| 方案 | 第12轮完整test mAP50 | FrSIoU减去对照，百分点 |','|---|---:|---:|',
            f'| FrSIoU 工程历史版 | {fr*100:.4f}% | 0 |']
        for row in comparisons:
            lines.append(f"| {row['method']} | {row['test_mAP']*100:.4f}% | {row['difference_percentage_points']:+.4f} |")
        lines+=['','单次seed42、DIOR-R trainval11725/test11738，固定末轮，不按test选权重。',
            'SIoU对照已按用户要求停止，尚无匹配的SIoU第12轮结果，不能单独归因历史机制。','']
        (DIRECTORY/'results.md').write_text('\n'.join(lines),encoding='utf-8')
        state.update(status='completed',active_arm=None,result=result,finished_at=HELPERS.now())
        HELPERS.write_json(DIRECTORY/'run.json',state)
    except Exception as error:
        if current is not None:
            state['arms'][current].update(status='failed',error=str(error))
            HELPERS.write_json(DIRECTORY/current/'run.json',state['arms'][current])
        state.update(status='failed',error=str(error),finished_at=HELPERS.now())
        HELPERS.write_json(DIRECTORY/'run.json',state)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('check','start','worker','status'))
    parser.add_argument('--lock-fd',type=int)
    args=parser.parse_args()
    os.chdir(ROOT)
    if args.action=='worker':
        if args.lock_fd is None:
            parser.error('Worker requires inherited queue descriptor')
        worker(args.lock_fd)
        return
    if args.action=='status':
        print((DIRECTORY/'run.json').read_text())
        return
    previous=previous_state()
    configs={name:configuration(name,previous) for name in ARMS}
    ready={}
    for name,cfg in configs.items():
        result=PREVIOUS/name/'results.json'
        ready[name]=check_resume(name,cfg) if result.is_file() else dict(status='awaiting_prior_epoch2_and_test')
    if args.action=='check':
        print(json.dumps(dict(target_epoch=12,resume_epoch=2,remaining_epochs=10,arms=ready,
            prior_status=previous['status'],resume_source=str(PREVIOUS/'frsiou_box_tangent')),indent=2))
        return
    DIRECTORY.mkdir(exist_ok=False)
    sources=list(dict.fromkeys(list(previous['source_sha256'])+['tools/run_fractional_siou_direct12.py']))
    for name in sources:
        dest=DIRECTORY/'snapshot/source'/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes((ROOT/name).read_bytes())
    arms={}
    for name,cfg in configs.items():
        frozen=DIRECTORY/name/'snapshot/experiment.py'
        frozen.parent.mkdir(parents=True,exist_ok=True)
        cfg.dump(str(frozen))
        arms[name]=dict(status='queued',target_epoch=12,completed_epochs_before_resume=2,remaining_epochs=10,
            frozen_config=str(frozen),frozen_config_sha256=digest(frozen),loss=dict(cfg.model.bbox_head.loss_bbox))
    state=dict(status='queued',created_at=HELPERS.now(),arms=arms,active_arm=None,source_sha256=hashes(sources),
        resume_source=str(PREVIOUS/'frsiou_box_tangent'),target_epoch=12,resume_epoch=2,remaining_epochs_per_arm=10,
        seed=42,trainval_images=11725,test_images=11738,final_test_per_arm=True,
        author_reproduction=False,preflight_resume_checks=ready,
        resume_note='Restore model,optimizer,counters and matching FrSIoU history; retain original SGD schedule without warmup restart.')
    HELPERS.write_json(DIRECTORY/'run.json',state)
    env=os.environ.copy()
    env.update(PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='0',PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:128')
    env['LD_LIBRARY_PATH']='/usr/lib/wsl/lib'+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    with (ROOT/'work_dirs/dior_r_trainval/server_queue.lock').open('a') as lock:
        with (DIRECTORY/'worker.log').open('w') as output:
            child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'worker','--lock-fd',str(lock.fileno())],
                cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,
                start_new_session=True,pass_fds=(lock.fileno(),))
    (DIRECTORY/'worker.pid').write_text(str(child.pid)+'\n')
    print(json.dumps(dict(status='queued',worker_pid=child.pid,work_dir=str(DIRECTORY),target_epoch=12,
        remaining_epochs_per_arm=10,resume_source=str(PREVIOUS/'frsiou_box_tangent'))))


if __name__=='__main__':
    main()
