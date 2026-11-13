"""Create a portable source bundle without datasets, checkpoints or local UI files."""
import argparse
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import subprocess
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
EXTRAS = [
    'configs/dior_r',
    'mmrotate/datasets/dior.py',
    'mmrotate/models/losses/rotated_smooth_l1_loss.py',
    'mmrotate/models/losses/rotated_wf_iou_loss.py',
    'tools/check_dior_r.py', 'tools/run_dior_r.sh', 'tools/queue_dior_r.py',
    'tools/check_dior_r_trainval.py', 'tools/run_dior_r_trainval.sh',
    'tools/setup_dior_r_server_env.sh',
    'tools/summarize_dior_r_trainval.py', 'tools/package_dior_r_trainval.py',
    'tools/diagnose_dior_r_smooth_l1.py',
    'tests/test_models/test_dior_r_experiment_losses.py',
    'tests/test_tools/test_queue_dior_r.py', 'tests/test_tools/test_trainval_runner.py',
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='dist/mmrotate034_dior_trainval_source.tar.gz')
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if output.exists():
        raise RuntimeError(f'Output already exists: {output}')
    names = set(subprocess.check_output(
        ['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')) - {''}
    for extra in EXTRAS:
        path = ROOT / extra
        candidates = path.rglob('*') if path.is_dir() else [path]
        for candidate in candidates:
            if candidate.is_file() and '__pycache__' not in candidate.parts:
                names.add(candidate.relative_to(ROOT).as_posix())
    names = sorted(name for name in names if (ROOT / name).is_file()
        and not name.startswith(('data/', 'work_dirs/', '.git/', '.vscode/', 'dist/'))
        and '__pycache__' not in Path(name).parts and not name.endswith('.pyc'))
    manifest = dict(protocol=dict(trainval=11725, test=11738,
        final_checkpoint='epoch_12.pth'), versions={}, files={})
    for distribution in ('torch', 'mmcv-full', 'mmdet', 'mmrotate'):
        manifest['versions'][distribution] = importlib.metadata.version(distribution)
    for name in names:
        manifest['files'][name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    payload = (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, 'w:gz') as archive:
        for name in names:
            archive.add(ROOT / name, arcname='mmrotate034-src/' + name, recursive=False)
        info = tarfile.TarInfo('mmrotate034-src/SERVER_SOURCE_MANIFEST.json')
        info.size = len(payload)
        info.mtime = int(time.time())
        info.mode = 0o644
        archive.addfile(info, io.BytesIO(payload))
    print(json.dumps(dict(output=str(output), source_files=len(names),
                         size_mib=round(output.stat().st_size / 1024**2, 2)), indent=2))


if __name__ == '__main__':
    main()
