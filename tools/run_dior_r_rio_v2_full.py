"""Fresh 12-epoch FCOS/v2 loss run and automatic fixed-checkpoint full test."""
import argparse
import copy
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CONFIG = 'configs/dior_r/server_trainval/rio_v2_adamw_full12.py'
DIRECTORY = ROOT / 'work_dirs/dior_r_trainval/fcos_rio_v2_adamw_full12_20261026'
FROZEN = DIRECTORY / 'snapshot/experiment.py'
PREFLIGHT = ROOT / 'work_dirs/dior_r/rio_v2_check_20261023/preflight.json'
spec = importlib.util.spec_from_file_location('rio_full_helpers', ROOT / 'tools/run_dior_r_rio.py')
H = importlib.util.module_from_spec(spec)
spec.loader.exec_module(H)
H.DIRECTORY = DIRECTORY


def configuration():
    import mmcv
    from mmcv.utils import import_modules_from_strings
    from mmrotate.models import build_detector
    previous = json.loads(PREFLIGHT.read_text())
    if previous['status'] != 'passed' or previous['steps'] != 1000:
        raise RuntimeError('The completed v2 1000-step check is required')
    if H.hashes(previous['source_sha256']) != previous['source_sha256']:
        raise RuntimeError('Sources changed since the v2 short check')
    original = mmcv.Config.fromfile(str(ROOT / previous['config']))
    if hashlib.sha256(original.pretty_text.encode()).hexdigest() != previous['resolved_config_sha256']:
        raise RuntimeError('Short-check configuration changed')
    cfg = mmcv.Config.fromfile(str(ROOT / CONFIG))
    left, right = copy.deepcopy(cfg.to_dict()), copy.deepcopy(original.to_dict())
    left.pop('work_dir')
    right.pop('work_dir')
    if left != right:
        raise RuntimeError('Full run differs from checked configuration beyond output path')
    if cfg.load_from is not None or cfg.resume_from is not None or cfg.get('auto_resume', False):
        raise RuntimeError('Full experiment must start with a fresh detector and optimizer')
    if cfg.runner.max_epochs != 12 or cfg.seed != 42 or cfg.data.samples_per_gpu != 2:
        raise RuntimeError('Unexpected full-training protocol')
    if (ROOT / cfg.work_dir).resolve() != DIRECTORY:
        raise RuntimeError('Unexpected output directory')
    split_root = ROOT / 'data/DIOR-R/ImageSets/Main'
    splits = {name: set((split_root / (name + '.txt')).read_text().split())
              for name in ('train', 'val', 'test')}
    if [len(splits[name]) for name in ('train', 'val', 'test')] != [5862, 5863, 11738]:
        raise RuntimeError('Unexpected DIOR split sizes')
    if splits['train'] & splits['val'] or (splits['train'] | splits['val']) & splits['test']:
        raise RuntimeError('Overlapping dataset splits')
    import_modules_from_strings(**cfg.custom_imports)
    model = build_detector(cfg.model)
    if model.bbox_head.__class__.__name__ != 'RiOV2FCOSHead':
        raise RuntimeError('Incorrect head')
    if any(p.requires_grad for p in model.bbox_head.conv_centerness.parameters()):
        raise RuntimeError('Centerness branch must be frozen')
    sources = sorted(set(previous['source_sha256']) | {
        CONFIG, 'tools/run_dior_r_rio_v2_full.py', 'tools/run_dior_r_rio.py',
        'tools/train.py', 'tools/test.py', 'mmrotate/datasets/dior.py'})
    return cfg, previous, sources


def guard(state):
    if H.hashes(state['source_sha256']) != state['source_sha256']:
        raise RuntimeError('Experiment source changed')
    if H.hashes([str(FROZEN)])[str(FROZEN)] != state['frozen_config_sha256']:
        raise RuntimeError('Frozen experiment configuration changed')


def worker(lock_fd):
    os.fstat(lock_fd)
    state = json.loads((DIRECTORY / 'run.json').read_text())
    state.update(worker_pid=os.getpid(), started_at=H.now())
    try:
        guard(state)
        H.run_child([sys.executable, str(ROOT / 'tools/train.py'), str(FROZEN),
                     '--seed', '42', '--no-validate'],
                    DIRECTORY / 'train.log', state, 'training', lock_fd)
        guard(state)
        checkpoint = DIRECTORY / 'epoch_12.pth'
        if not checkpoint.is_file():
            raise RuntimeError('Training ended without epoch_12.pth')
        import torch
        saved = torch.load(str(checkpoint), map_location='cpu')
        if saved['meta'].get('epoch') != 12 or saved['meta'].get('iter') != 70356:
            raise RuntimeError('Unexpected final epoch/iteration')
        if not saved.get('optimizer', {}).get('state'):
            raise RuntimeError('Final checkpoint lacks optimizer state')
        if any(not torch.isfinite(x).all() for x in saved['state_dict'].values()):
            raise RuntimeError('Nonfinite final model weights')
        del saved
        H.run_child([sys.executable, str(ROOT / 'tools/test.py'), str(FROZEN), str(checkpoint),
                     '--work-dir', str(DIRECTORY / 'test_final'),
                     '--out', str(DIRECTORY / 'test_final_predictions.pkl'),
                     '--eval', 'mAP'],
                    DIRECTORY / 'test.log', state, 'testing', lock_fd)
        guard(state)
        evaluations = sorted((DIRECTORY / 'test_final').glob('eval_*.json'))
        if len(evaluations) != 1:
            raise RuntimeError('Expected one fixed-checkpoint test evaluation')
        score = float(json.loads(evaluations[0].read_text())['metric']['mAP'])
        if not math.isfinite(score):
            raise RuntimeError('Nonfinite test mAP')
        result = dict(config=CONFIG, checkpoint=str(checkpoint), epoch=12,
                      iterations=70356, test_mAP=score, trainval_images=11725,
                      test_images=11738, seed=42, initialization='fresh ImageNet backbone',
                      classification='VFL, KLD-derived detached quality',
                      regression_weights=dict(kld=5, spatial_l1=5, sp_l1=5),
                      regression_weighting='uniform positive points', centerness=False,
                      full_rio_v2_reproduction=False, source_sha256=state['source_sha256'],
                      evaluation_source=str(evaluations[0]))
        H.write_json(DIRECTORY / 'results.json', result)
        (DIRECTORY / 'results.md').write_text(
            '# FCOS + RiO-DETRv2 loss adaptation\n\n'
            f'Fixed epoch 12 test AP50: **{100 * score:.4f}%**.\n\n'
            'VFL + 5*KLD + 5*spatial L1 + 5*SP-L1; uniform positive-point regression.\n'
            'Trainval 11,725 / test 11,738; seed 42; fresh 12-epoch run.\n'
            'FCOS outputs and assignment; no FDR or teacher distillation.\n', encoding='utf-8')
        state.update(status='completed', finished_at=H.now(), result=result)
        H.write_json(DIRECTORY / 'run.json', state)
    except Exception as error:
        state.update(status='failed', finished_at=H.now(), error=str(error))
        H.write_json(DIRECTORY / 'run.json', state)
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
            parser.error('worker requires inherited experiment lock')
        worker(args.lock_fd)
        return
    cfg, previous, sources = configuration()
    if args.action == 'check':
        print(json.dumps(dict(config=CONFIG, epochs=12, source_files=len(sources),
                              short_check_AP50=previous['train_image_mAP'],
                              fresh=True, final_test_images=11738)))
        return
    if shutil.disk_usage(ROOT).free < 8 * 1024**3:
        raise RuntimeError('Less than 8 GiB free for full-run checkpoints')
    gpu = subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi', '--id=0',
        '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True)
    if int(gpu.strip()) < 4096:
        raise RuntimeError('GPU currently has less than 4096 MiB free')
    lock_path = ROOT / 'work_dirs/dior_r_trainval/server_queue.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        DIRECTORY.mkdir(parents=True, exist_ok=False)
        FROZEN.parent.mkdir(parents=True)
        cfg.dump(str(FROZEN))
        for name in sources:
            output = DIRECTORY / 'snapshot/source' / name
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes((ROOT / name).read_bytes())
        H.write_json(DIRECTORY / 'preflight.json', previous)
        H.write_json(DIRECTORY / 'run.json', dict(
            status='starting', config=CONFIG, frozen_config=str(FROZEN),
            created_at=H.now(), target_epoch=12, seed=42, fresh=True,
            load_from=None, resume_from=None, short_check_resume=False,
            final_test=True, intermediate_test=False, trainval_images=11725,
            test_images=11738, source_sha256=H.hashes(sources),
            frozen_config_sha256=H.hashes([str(FROZEN)])[str(FROZEN)],
            health_guard='Nonfinite metrics or sustained gradient explosion; no test AP stopping',
            adaptation='FCOS outputs/assignment; VFL + uniformly weighted KLD/L1/SP-L1; '
                       'no Centerness, FDR, decoder auxiliary losses or distillation'))
        env = os.environ.copy()
        env.update(PYTHONUNBUFFERED='1', CUDA_VISIBLE_DEVICES='0')
        env['LD_LIBRARY_PATH'] = '/usr/lib/wsl/lib' + (
            ':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
        with (DIRECTORY / 'worker.log').open('w') as output:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                'worker', '--lock-fd', str(lock.fileno())], cwd=ROOT, env=env,
                stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                start_new_session=True, pass_fds=(lock.fileno(),))
        (DIRECTORY / 'worker.pid').write_text(str(process.pid) + '\n')
        print(json.dumps(dict(worker_pid=process.pid, work_dir=str(DIRECTORY),
                              target_epoch=12, fresh=True, final_test=True)))


if __name__ == '__main__':
    main()
