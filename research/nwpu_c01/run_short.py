"""Run the paired frozen-classifier trials and their official final evaluation."""
import argparse
import subprocess
import sys
from pathlib import Path

from . import classifier, rank
from .evaluate import evaluate
from .runtime import checkpoint_path, now, save

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT/'configs/nwpu_r/c01_36/FIELD_P3.py'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('protocol',choices=('classifier','rank'))
    parser.add_argument('--p3',required=True,type=Path,help='Official FIELD_P3 epoch 36 checkpoint')
    parser.add_argument('--agree',type=Path,help='Official FIELD_AGREE epoch 36 checkpoint for distillation')
    parser.add_argument('--work-dir',required=True,type=Path)
    parser.add_argument('--config',default=DEFAULT_CONFIG,type=Path,
                        help='FIELD_P3 config, with dataset paths resolved from the repository root')
    parser.add_argument('--smoke',action='store_true',help='Train one paired batch and record the preflight checks')
    args = parser.parse_args()
    if Path.cwd().resolve()!=ROOT:
        parser.error('Run from the repository root')
    if args.protocol=='classifier' and args.agree is None:
        parser.error('--agree is required for classifier distillation')
    source = {'FIELD_P3':args.p3.resolve(strict=True)}
    if args.agree is not None:
        source['FIELD_AGREE'] = args.agree.resolve(strict=True)
    destination = args.work_dir.resolve()
    destination.mkdir(parents=True,exist_ok=True)
    status_path = destination/'status.json'
    if status_path.exists():
        parser.error('Choose a fresh --work-dir')
    config = args.config.resolve(strict=True)
    module = classifier if args.protocol=='classifier' else rank
    state = dict(status='preparing',started_at=now(),modes=module.MODES)
    save(status_path,state)
    try:
        module.train(config,state,destination,source,smoke=args.smoke)
        if args.smoke:
            state.update(status='smoke_passed',finished_at=now())
            return
        import torch

        torch.cuda.empty_cache()
        for mode in module.MODES:
            ckpt = checkpoint_path(mode,destination)
            preds = ckpt.parent/'predictions.pkl'
            state.update(status='evaluating',current=mode,updated_at=now())
            save(status_path,state)
            subprocess.run([sys.executable,str(ROOT/'tools/test.py'),str(config),str(ckpt),'--out',str(preds)],check=True)
            save(ckpt.parent/'metrics.json',evaluate(config,preds))
        state.update(status='completed',finished_at=now(),current=None)
    except Exception as exc:
        state.update(status='failed',finished_at=now(),error=str(exc))
        raise
    finally:
        save(status_path,state)


if __name__=='__main__':
    main()
