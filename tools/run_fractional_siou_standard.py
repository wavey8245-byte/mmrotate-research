"""Two standard epochs each: FrSIoU engineering box tangent, matched SIoU.

Epoch 1 populates per-object history; epoch 2 exercises prior-visit memory.
Each arm is initialized fresh, retains the common FCOS/SGD protocol, and
gets one full-test evaluation at fixed epoch 2. This is not author code.
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
DIRECTORY=ROOT/'work_dirs/dior_r_trainval/frsiou_standard2_controls_20261022'
PROOF=ROOT/'work_dirs/dior_r_trainval/frsiou_repeated_pool_200_20261022/report.json'
ARMS={'frsiou_box_tangent':'paper_frsiou_box_tangent.py','siou':'paper_frsiou_siou_control.py'}
SPEC=importlib.util.spec_from_file_location('frsiou_run_helpers',ROOT/'tools/run_dior_r_rio.py')
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


def verify_proof():
    proof=json.loads(PROOF.read_text())
    if proof['status']!='completed_short_controls' or proof['steps_per_arm']!=200 or not proof['sources_unchanged']:
        raise RuntimeError('Completed repeated-pool 200-step history controls required')
    if hashes(proof['source_sha256'])!=proof['source_sha256']:
        raise RuntimeError('A source changed after the short controls')
    reference=proof['arms']['siou']['augmented_batch_sha256']
    for name in ('siou','frsiou_values','frsiou_box_tangent'):
        arm=proof['arms'][name]
        if arm['steps_completed']!=200 or arm['status']!='completed_short_probe' or arm['augmented_batch_sha256']!=reference:
            raise RuntimeError('Incomplete or mismatched short arm: '+name)
        if name!='siou' and not arm.get('total_history_hit_groups',0):
            raise RuntimeError('History was not exercised in '+name)
    return proof


def configuration(name,proof):
    import mmcv
    cfg=mmcv.Config.fromfile(str(ROOT/'configs/dior_r/server_trainval'/ARMS[name]))
    baseline=mmcv.Config.fromfile(str(ROOT/'configs/dior_r/server_trainval/riou.py'))
    for key in ('data','optimizer','optimizer_config','lr_config','seed','load_from','resume_from'):
        if cfg[key]!=baseline[key]:
            raise RuntimeError('Unexpected training protocol difference: '+key)
    model,reference=copy.deepcopy(cfg.model),copy.deepcopy(baseline.model)
    loss=model.bbox_head.pop('loss_bbox')
    reference.bbox_head.pop('loss_bbox')
    reference.bbox_head.type='FractionalRotatedFCOSHead'
    if model!=reference or dict(loss)!=proof['arms'][name]['loss']:
        raise RuntimeError('Full model/loss differs from verified adapter')
    if cfg.load_from is not None or cfg.resume_from is not None:
        raise RuntimeError('Both standard arms must initialize fresh')
    cfg.runner.max_epochs=2
    cfg.work_dir=str(DIRECTORY/name)
    cfg.auto_resume=False
    health_data=copy.deepcopy(cfg.data.test)
    health_data.ann_file='data/DIOR-R/ImageSets/Main/train.txt'
    health_data.img_prefix='data/DIOR-R/JPEGImages-trainval/'
    health_data.test_mode=True
    imports=list(cfg.custom_imports['imports'])+['mmrotate.core.rio_training_health']
    cfg.custom_imports=dict(imports=list(dict.fromkeys(imports)),allow_failed_imports=False)
    cfg.custom_hooks=list(cfg.get('custom_hooks',[]))+[dict(
        type='RiOTrainingHealthHook',dataset_cfg=health_data,images=128,seed=42,
        minimum_map=.05,minimum_fraction=.5,diagnostic_iterations=[],priority='LOW')]
    return cfg


def guard(state):
    if hashes(state['source_sha256'])!=state['source_sha256']:
        raise RuntimeError('Fractional experiment sources changed')
    if digest(PROOF)!=state['short_proof_sha256']:
        raise RuntimeError('Short proof changed during full controls')
    for name,arm in state['arms'].items():
        if digest(Path(arm['frozen_config']))!=arm['frozen_config_sha256']:
            raise RuntimeError('Frozen config changed: '+name)


def worker(lock_fd):
    os.fstat(lock_fd)
    state=json.loads((DIRECTORY/'run.json').read_text())
    state.update(worker_pid=os.getpid(),started_at=HELPERS.now())
    current=None
    try:
        for name in ARMS:
            current=name
            arm=state['arms'][name]
            arm_dir=DIRECTORY/name
            frozen=Path(arm['frozen_config'])
            guard(state)
            state.update(status='running',active_arm=name)
            HELPERS.write_json(DIRECTORY/'run.json',state)
            HELPERS.DIRECTORY=arm_dir
            HELPERS.run_child([sys.executable,str(ROOT/'tools/train.py'),str(frozen),
                '--seed','42','--no-validate'],arm_dir/'train.log',arm,'training',lock_fd)
            import torch
            checkpoint=arm_dir/'epoch_2.pth'
            health=json.loads((arm_dir/'training_health.json').read_text())
            if [x['epoch'] for x in health]!=[1,2] or any(x['status']!='passed' for x in health):
                raise RuntimeError('Expected two successful native epoch health checks')
            saved=torch.load(str(checkpoint),map_location='cpu')
            if saved['meta'].get('epoch')!=2 or saved['meta'].get('iter')!=11726 or not saved.get('optimizer',{}).get('state'):
                raise RuntimeError('Invalid standard epoch-2 checkpoint')
            del saved
            checkpoint_hash=digest(checkpoint)
            arm.update(training_completed_at=HELPERS.now(),checkpoint=str(checkpoint),
                checkpoint_sha256=checkpoint_hash,training_health=health)
            if name=='frsiou_box_tangent':
                side=checkpoint.with_suffix('.frsiou_history.pth')
                cache=torch.load(str(side),map_location='cpu')
                if cache['epoch']!=2 or cache['iteration']!=11726 or cache['checkpoint_sha256']!=checkpoint_hash:
                    raise RuntimeError('Fractional sidecar/checkpoint mismatch')
                if not cache['history']['total_history_hits'] or not cache['history']['records']:
                    raise RuntimeError('Second epoch failed to exercise per-object history')
                arm.update(history_sidecar=str(side),history_sidecar_sha256=digest(side),
                    cached_groups=len(cache['history']['records']),total_history_hits=cache['history']['total_history_hits'])
                del cache
            guard(state)
            HELPERS.write_json(DIRECTORY/'run.json',state)
            HELPERS.run_child([sys.executable,str(ROOT/'tools/test.py'),str(frozen),str(checkpoint),
                '--work-dir',str(arm_dir/'test_epoch_2'),'--out',str(arm_dir/'test_predictions.pkl'),
                '--eval','mAP','--eval-options','iou_thr=0.5','use_07_metric=True'],
                arm_dir/'test.log',arm,'testing',lock_fd)
            evaluations=list((arm_dir/'test_epoch_2').glob('eval_*.json'))
            if len(evaluations)!=1:
                raise RuntimeError('Expected one complete test evaluation per arm')
            score=float(json.loads(evaluations[0].read_text())['metric']['mAP'])
            if not math.isfinite(score) or digest(checkpoint)!=checkpoint_hash:
                raise RuntimeError('Nonfinite score or changed checkpoint')
            guard(state)
            result=dict(status='completed',method=name,epoch=2,test_images=11738,trainval_images=11725,
                test_mAP=score,mAP_percent=100*score,iou_thr=.5,use_07_metric=True,
                loss=arm['loss'],author_reproduction=False,
                checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_hash,
                evaluation_source=str(evaluations[0]),finished_at=HELPERS.now())
            HELPERS.write_json(arm_dir/'results.json',result)
            arm.update(status='completed',result=result,finished_at=HELPERS.now())
            HELPERS.write_json(arm_dir/'run.json',arm)
            HELPERS.write_json(DIRECTORY/'run.json',state)
        fr=state['arms']['frsiou_box_tangent']['result']['test_mAP']
        siou=state['arms']['siou']['result']['test_mAP']
        result=dict(status='completed_two_epoch_controls',epochs_per_arm=2,
            test_images=11738,trainval_images=11725,seed=42,
            frsiou_engineering_test_mAP=fr,siou_control_test_mAP=siou,
            difference_percentage_points=100*(fr-siou),author_reproduction=False,
            historical_gradient='Explicit normalized-box local tangent; no old network Jacobian replay.',
            arms={k:v['result'] for k,v in state['arms'].items()},finished_at=HELPERS.now())
        HELPERS.write_json(DIRECTORY/'results.json',result)
        summary=['# FrSIoU 工程历史梯度版：标准两轮对照','',
            'Raw FPN 语义，alpha=.3、beta=.5、lambda=.3；工程选择 nu=.5、M=2。',
            '采用式(12)一致的 G-L 系数；按图片/GT行/FPN层保存历史，加权聚合正样本。',
            '历史框梯度使用局部切线传递，没有重放旧网络参数雅可比；不是作者实现的严格复现。',
            '两组均为标准 train+val 两轮，第一轮建立历史，第二轮使用历史；固定 epoch2 完整 test。','',
            '| 方案 | 第2轮完整 test mAP50 |','|---|---:|',
            f'| FrSIoU 工程历史框梯度版 | {fr*100:.4f}% |',
            f'| 同适配器 SIoU 对照 | {siou*100:.4f}% |','',
            f'差值：{100*(fr-siou):+.4f} 个百分点。',
            '单次 seed42；不能与12轮结果直接比较，也不能等同论文原模型/原任务。','']
        (DIRECTORY/'results.md').write_text('\n'.join(summary),encoding='utf-8')
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
            parser.error('Worker requires inherited experiment lock')
        worker(args.lock_fd)
        return
    if args.action=='status':
        print((DIRECTORY/'run.json').read_text())
        return
    proof=verify_proof()
    configs={name:configuration(name,proof) for name in ARMS}
    split=ROOT/'data/DIOR-R/ImageSets/Main'
    ids={key:set((split/(key+'.txt')).read_text().split()) for key in ('train','val','test')}
    if [len(ids[k]) for k in ('train','val','test')]!=[5862,5863,11738] or ids['train']&ids['val'] or (ids['train']|ids['val'])&ids['test']:
        raise RuntimeError('Unexpected or overlapping dataset splits')
    if args.action=='check':
        print(json.dumps(dict(arms=list(ARMS),epochs_per_arm=2,fresh_initialization=True,
            first_epoch='Populate per-GT/FPN history',second_epoch='Use prior-visit memory',
            full_test_per_arm=True,author_reproduction=False),indent=2))
        return
    lock_path=ROOT/'work_dirs/dior_r_trainval/server_queue.lock'
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('Another experiment holds the shared queue lock')
        DIRECTORY.mkdir(exist_ok=False)
        sources=list(dict.fromkeys(list(proof['source_sha256'])+[
            'tools/run_fractional_siou_standard.py','tools/train.py','tools/test.py',
            'tools/run_dior_r_rio.py','mmrotate/core/rio_training_health.py',
            'data/DIOR-R/ImageSets/Main/train.txt','data/DIOR-R/ImageSets/Main/val.txt',
            'data/DIOR-R/ImageSets/Main/test.txt']))
        for name in sources:
            dest=DIRECTORY/'snapshot/source'/name
            dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes((ROOT/name).read_bytes())
        arms={}
        for name,cfg in configs.items():
            frozen=DIRECTORY/name/'snapshot/experiment.py'
            frozen.parent.mkdir(parents=True,exist_ok=True)
            cfg.dump(str(frozen))
            arms[name]=dict(status='queued',target_epochs=2,loss=dict(cfg.model.bbox_head.loss_bbox),
                frozen_config=str(frozen),frozen_config_sha256=digest(frozen))
        state=dict(status='starting',created_at=HELPERS.now(),arms=arms,active_arm=None,
            epochs_per_arm=2,source_sha256=hashes(sources),short_proof_sha256=digest(PROOF),
            author_reproduction=False,seed=42,trainval_images=11725,test_images=11738,
            interpretation='GL Eq12, per-GT/FPN weighted mean, nu=.5,M=2,lambda=.3,raw FPN, explicit historical box tangent')
        HELPERS.write_json(DIRECTORY/'run.json',state)
        env=os.environ.copy()
        env.update(PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='0',
            PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:128')
        env['LD_LIBRARY_PATH']='/usr/lib/wsl/lib'+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
        with (DIRECTORY/'worker.log').open('w') as output:
            child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'worker','--lock-fd',str(lock.fileno())],
                cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,
                start_new_session=True,pass_fds=(lock.fileno(),))
        (DIRECTORY/'worker.pid').write_text(str(child.pid)+'\n')
        print(json.dumps(dict(worker_pid=child.pid,work_dir=str(DIRECTORY),arms=list(ARMS),epochs_per_arm=2,full_test_per_arm=True)))


if __name__=='__main__':
    main()
