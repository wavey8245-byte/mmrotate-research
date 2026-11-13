"""Queue matched 1000-step component or auxiliary-weight checks."""
import argparse
import copy
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'work_dirs/dior_r/rio_adamw_components_20261018'
REFERENCE = ROOT / 'work_dirs/dior_r/rio_adamw_preflight_20261018.json'
DEPENDENCY = ROOT / 'work_dirs/dior_r_trainval/fcos_riou_adamw_probe1_20261018/test_epoch_1/run.json'
ARMS = dict(kld='configs/dior_r/server_trainval/rio_adamw_kld.py',
            kld_l1='configs/dior_r/server_trainval/rio_adamw_kld_l1.py')
EXPECTED_WEIGHTS = dict(kld=(0., 0.), kld_l1=(5., 0.))


def configure_suite(suite):
    global FOLDER, ARMS, EXPECTED_WEIGHTS
    if suite == 'weights':
        FOLDER = ROOT / 'work_dirs/dior_r/rio_adamw_weights_20261022'
        ARMS = dict(spatial05='configs/dior_r/server_trainval/rio_adamw_spatial05.py',
                    angle05='configs/dior_r/server_trainval/rio_adamw_angle05.py',
                    aux05='configs/dior_r/server_trainval/rio_adamw_aux05.py')
        EXPECTED_WEIGHTS = dict(spatial05=(.5, 5.), angle05=(5., .5), aux05=(.5, .5))
    elif suite != 'components':
        raise ValueError(f'Unknown suite: {suite}')


def save(state):
    state['updated_at'] = datetime.now().astimezone().isoformat()
    temporary = FOLDER / 'run.json.tmp'
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    temporary.replace(FOLDER / 'run.json')


def validate():
    import mmcv
    reference = json.loads(REFERENCE.read_text())
    if reference['steps'] != 1000 or reference['status'] != 'passed':
        raise RuntimeError('Expected the preserved three-term 1000-step reference')
    for path, digest in reference['source_sha256'].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Reference source changed: {path}')
    baseline = mmcv.Config.fromfile(reference['config'])
    if hashlib.sha256(baseline.pretty_text.encode()).hexdigest() != reference['resolved_config_sha256']:
        raise RuntimeError('Resolved reference configuration changed')
    sources = set(reference['source_sha256']) | set(ARMS.values()) | {
        'configs/dior_r/server_trainval/riou_adamw.py',
        'configs/dior_r/server_trainval/riou.py', str(Path(__file__).relative_to(ROOT))}
    for name, config in ARMS.items():
        candidate = mmcv.Config.fromfile(config)
        left, right = copy.deepcopy(candidate.to_dict()), copy.deepcopy(baseline.to_dict())
        weights = left['model']['bbox_head']['loss_bbox']
        expected = EXPECTED_WEIGHTS[name]
        if (weights['spatial_weight'], weights['angle_weight']) != expected:
            raise RuntimeError(f'Unexpected ablation weights: {name}')
        for value in (left, right):
            value.pop('work_dir')
            value['model']['bbox_head']['loss_bbox'].pop('spatial_weight')
            value['model']['bbox_head']['loss_bbox'].pop('angle_weight')
        if left != right:
            raise RuntimeError(f'Unmatched training protocol: {name}')
    return {str(p): hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sorted(sources)}


def worker():
    state = json.loads((FOLDER / 'run.json').read_text())
    try:
        if validate() != state['source_sha256']:
            raise RuntimeError('Ablation inputs changed after queueing')
        while True:
            dependency = json.loads(DEPENDENCY.read_text())
            if dependency['status'] == 'completed':
                break
            if dependency['status'] == 'failed':
                raise RuntimeError('Control evaluation failed; inspect it before continuing')
            time.sleep(30)
        state['control_test_mAP'] = dependency['metric']['mAP']
        for name, config in ARMS.items():
            if validate() != state['source_sha256']:
                raise RuntimeError('Ablation sources changed')
            state.update(status='training', arm=name)
            save(state)
            report = FOLDER / (name + '.json')
            with (FOLDER / (name + '.log')).open('w') as output:
                process = subprocess.Popen([sys.executable, str(ROOT / 'tools/check_dior_r_rio.py'),
                    '--config', config, '--steps', '1000', '--report', str(report)],
                    cwd=ROOT, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT)
                state['child_pid'] = process.pid
                save(state)
                code = process.wait()
            result = json.loads(report.read_text())
            # A completed experiment can fail the unchanged learning gate.
            # Preserve that outcome; never relabel it as a passed stability check.
            if not result['history'] or result['history'][-1]['step'] != 1000:
                raise RuntimeError(f'{name} stopped before step 1000: {result.get("error")}')
            if not math.isfinite(result.get('train_image_mAP', float('nan'))):
                raise RuntimeError(f'{name} has no finite learning evaluation')
            reference = json.loads(REFERENCE.read_text())
            for metric in ('loss_cls', 'bbox_kld'):
                if not math.isclose(result['history'][0][metric],
                                    reference['history'][0][metric], rel_tol=1e-5, abs_tol=1e-6):
                    raise RuntimeError(f'{name} has an unmatched first-batch {metric}')
            state['results'][name] = dict(returncode=code, status=result['status'],
                train64_mAP=result['train_image_mAP'], report=str(report),
                kld_weight=5., spatial_weight=EXPECTED_WEIGHTS[name][0],
                angle_weight=EXPECTED_WEIGHTS[name][1])
            save(state)
        state.update(status='completed', finished_at=datetime.now().astimezone().isoformat())
        save(state)
    except Exception as error:
        state.update(status='failed', error=str(error))
        save(state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'start', 'worker', 'status'))
    parser.add_argument('--suite', choices=('components', 'weights'), default='components')
    args = parser.parse_args()
    os.chdir(ROOT)
    configure_suite(args.suite)
    if args.action == 'status':
        print((FOLDER / 'run.json').read_text())
        return
    if args.action == 'worker':
        worker()
        return
    hashes = validate()
    if args.action == 'check':
        print('Matched manual sampler, seed 42, batch 2, 1000 updates and the same first 64 train images.')
        return
    FOLDER.mkdir(parents=True, exist_ok=False)
    for path in hashes:
        output = FOLDER / 'snapshot/source' / path
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes((ROOT / path).read_bytes())
    reference = json.loads(REFERENCE.read_text())
    (FOLDER / 'three_term_reference.json').write_bytes(REFERENCE.read_bytes())
    save(dict(status='waiting_for_control_test', source_sha256=hashes, suite=args.suite,
        created_at=datetime.now().astimezone().isoformat(), dependency=str(DEPENDENCY),
        steps_per_arm=1000, test_evaluation=False,
        scope='Training-subset ablation; not a complete epoch or final test comparison.',
        results=dict(kld_l1_sp=dict(status=reference['status'],
                                  train64_mAP=reference['train_image_mAP'], report=str(REFERENCE)))))
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED='1', CUDA_VISIBLE_DEVICES='0')
    env['LD_LIBRARY_PATH'] = '/usr/lib/wsl/lib' + (':' + env['LD_LIBRARY_PATH']
                                                 if env.get('LD_LIBRARY_PATH') else '')
    with (FOLDER / 'worker.log').open('w') as output:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
            'worker', '--suite', args.suite],
            cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=output,
            stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps(dict(worker_pid=process.pid, output=str(FOLDER))), flush=True)


if __name__ == '__main__':
    main()
