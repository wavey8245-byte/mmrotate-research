"""Checkpoint validation and JSON output for the reproduction commands."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix+'.pending')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n',encoding='utf8')
    pending.replace(path)


def sha(path):
    with Path(path).open('rb') as f:
        h = hashlib.sha256()
        for block in iter(lambda:f.read(2**20),b''):
            h.update(block)
    return h.hexdigest()


def reference_records(checkpoints):
    return {name:dict(checkpoint=str(path), checkpoint_sha256=sha(path), status='completed',
                      epoch=36,iter=11700) for name,path in checkpoints.items()}


def reference_state(path):
    import torch

    ckpt = torch.load(path,map_location='cpu')
    meta = ckpt.get('meta',{})
    if (meta.get('epoch'),meta.get('iter'))!=(36,11700):
        raise ValueError(f'Expected an official epoch 36 / step 11700 checkpoint: {path}')
    return ckpt['state_dict']


def checkpoint_path(mode, destination):
    return destination/mode/'epoch_3.pth'
