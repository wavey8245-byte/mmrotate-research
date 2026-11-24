"""Score an official NWPU-R prediction pickle and write overall and shape AP."""
import argparse
from pathlib import Path

from .evaluate import evaluate
from .runtime import save


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config',type=Path)
    parser.add_argument('predictions',type=Path)
    parser.add_argument('metrics',type=Path)
    args = parser.parse_args()
    save(args.metrics,evaluate(args.config,args.predictions))
