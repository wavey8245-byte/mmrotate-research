"""Queue the user-selected FCOS/v2 loss adaptation on an available GPU."""
import argparse
import ast
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CONFIG = 'configs/dior_r/server_trainval/rio_v2_adamw.py'
FOLDER = ROOT / 'work_dirs/dior_r/rio_v2_check_20261023'
REPORT = FOLDER / 'preflight.json'
STEPS = 1000


def config_sources(path):
    result = {path}
    for node in ast.parse((ROOT / path).read_text()).body:
        if isinstance(node, ast.Assign) and any(
                isinstance(x, ast.Name) and x.id == '_base_' for x in node.targets):
            bases = ast.literal_eval(node.value)
            for base in ([bases] if isinstance(bases, str) else bases):
                parent = (ROOT / path).parent / base
                result.update(config_sources(str(parent.resolve().relative_to(ROOT))))
    return result


def source_hashes():
    names = config_sources(CONFIG) | {
        'mmrotate/models/dense_heads/rio_v2_fcos_head.py',
        'mmrotate/models/losses/rio_v2_varifocal_loss.py',
        'mmrotate/models/dense_heads/rotated_fcos_head.py',
        'mmrotate/models/dense_heads/rotated_anchor_free_head.py',
        'mmrotate/models/losses/rotated_rio_loss.py',
        'tools/check_dior_r_rio.py', str(Path(__file__).resolve().relative_to(ROOT))}
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in sorted(names)}


def save(state):
    state['updated_at'] = datetime.now().astimezone().isoformat()
    temporary = FOLDER / 'run.json.tmp'
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    temporary.replace(FOLDER / 'run.json')


def gpu_status():
    result = subprocess.run([
        '/usr/lib/wsl/lib/nvidia-smi', '--id=0',
        '--query-gpu=memory.free,memory.total,utilization.gpu',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    free, total, utilization = [int(x.strip()) for x in result.stdout.strip().split(',')]
    return dict(free_mib=free, total_mib=total, utilization_percent=utilization)


def check_configuration():
    import mmcv
    from mmcv.utils import import_modules_from_strings
    from mmrotate.models import build_detector
    cfg = mmcv.Config.fromfile(CONFIG)
    import_modules_from_strings(**cfg.custom_imports)
    model = build_detector(cfg.model)
    assert model.bbox_head.__class__.__name__ == 'RiOV2FCOSHead'
    assert not any(x.requires_grad for x in model.bbox_head.conv_centerness.parameters())
    loss = model.bbox_head.loss_bbox
    assert (loss.kld_weight, loss.spatial_weight, loss.angle_weight) == (5., 5., 5.)
    return cfg


def worker():
    state = json.loads((FOLDER / 'run.json').read_text())
    state['worker_pid'] = os.getpid()
    try:
        consecutive_ready = 0
        while consecutive_ready < 2:
            if source_hashes() != state['source_sha256']:
                raise RuntimeError('Queued experiment sources changed; refusing a mixed run')
            gpu = gpu_status()
            ready = gpu['free_mib'] >= state['minimum_free_mib'] and gpu['utilization_percent'] < 10
            consecutive_ready = consecutive_ready + 1 if ready else 0
            state.update(status='waiting_for_gpu', gpu=gpu)
            save(state)
            if consecutive_ready < 2:
                time.sleep(30)
        state.update(status='training', started_at=datetime.now().astimezone().isoformat())
        save(state)
        spec = importlib.util.spec_from_file_location('rio_existing_preflight',
                                                    ROOT / 'tools/check_dior_r_rio.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # Reuse the established sampler, optimizer, clipping, evaluation and
        # unchanged learning gate. Include every new implementation/config in
        # its provenance and diagnostic checkpoint, rather than losing imports.
        module.source_hashes = lambda config=CONFIG: source_hashes()
        sys.argv = [str(ROOT / 'tools/check_dior_r_rio.py'), '--config', CONFIG,
                    '--steps', str(STEPS), '--report', str(REPORT)]
        module.main()
        result = json.loads(REPORT.read_text())
        state.update(status='completed', result_status=result['status'],
                     train64_mAP=result['train_image_mAP'], report=str(REPORT),
                     finished_at=datetime.now().astimezone().isoformat())
        save(state)
    except BaseException as error:
        state.update(status='failed', error=str(error),
                     finished_at=datetime.now().astimezone().isoformat())
        if REPORT.exists():
            result = json.loads(REPORT.read_text())
            state.update(result_status=result.get('status'), report=str(REPORT),
                         train64_mAP=result.get('train_image_mAP'))
        save(state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'start', 'worker', 'status'))
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.action == 'status':
        print((FOLDER / 'run.json').read_text())
        return
    if args.action == 'worker':
        worker()
        return
    cfg = check_configuration()
    if args.action == 'check':
        print(json.dumps(dict(config=CONFIG, source_files=len(source_hashes()),
                              gpu=gpu_status(), steps=STEPS)))
        return
    FOLDER.mkdir(parents=True, exist_ok=False)
    hashes = source_hashes()
    for name in hashes:
        output = FOLDER / 'snapshot/source' / name
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes((ROOT / name).read_bytes())
    cfg.dump(str(FOLDER / 'snapshot/experiment.py'))
    save(dict(status='waiting_for_gpu', source_sha256=hashes, steps=STEPS,
              minimum_free_mib=4800, config=CONFIG, test_evaluation=False,
              scope='FCOS outputs/assignment; VFL + uniformly weighted KLD/L1/SP-L1; '
                    '1000-step training-subset check, not a full epoch or test evaluation',
              created_at=datetime.now().astimezone().isoformat()))
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED='1', CUDA_VISIBLE_DEVICES='0')
    env['LD_LIBRARY_PATH'] = '/usr/lib/wsl/lib' + (
        ':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    with (FOLDER / 'worker.log').open('w') as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'worker'],
                                   cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps(dict(worker_pid=process.pid, output=str(FOLDER))))


if __name__ == '__main__':
    main()
