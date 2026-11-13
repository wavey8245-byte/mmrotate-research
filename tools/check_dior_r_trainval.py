"""Check the train+val/test benchmark protocol without allocating the GPU."""
import argparse
import json
from pathlib import Path

import mmcv
import numpy as np

from mmrotate.datasets import build_dataset


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / 'configs/dior_r/server_trainval'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', default='work_dirs/dior_r_trainval/protocol_check.json')
    args = parser.parse_args()
    baseline = mmcv.Config.fromfile(str(CONFIG_DIR / 'riou.py'))
    for variant in ('smooth_l1', 'wf', 'riou_linear'):
        cfg = mmcv.Config.fromfile(str(CONFIG_DIR / f'{variant}.py'))
        for field in ('data', 'optimizer', 'optimizer_config', 'runner',
                      'lr_config', 'evaluation', 'seed'):
            assert cfg[field] == baseline[field], (variant, field)
    assert baseline.data.samples_per_gpu == 2
    assert baseline.optimizer.lr == .0025
    assert baseline.runner.max_epochs == 12
    assert baseline.evaluation.save_best is None
    assert baseline.data.val.ann_file == baseline.data.test.ann_file
    assert baseline.data.val.img_prefix == baseline.data.test.img_prefix
    trainval = build_dataset(baseline.data.train)
    assert len(trainval.datasets) == 2
    parts = trainval.datasets
    ids = [[info['id'] for info in dataset.data_infos] for dataset in parts]
    assert [len(part) for part in ids] == [5862, 5863]
    assert not (set(ids[0]) & set(ids[1]))
    merged = ids[0] + ids[1]
    assert len(trainval) == len(set(merged)) == 11725
    test_cfg = baseline.data.test.copy()
    test_cfg.test_mode = True
    test = build_dataset(test_cfg)
    test_ids = [info['id'] for info in test.data_infos]
    assert len(test) == len(set(test_ids)) == 11738
    assert not (set(merged) & set(test_ids))
    # Same grouped sampler as single-GPU MMDetection 2.x training.
    groups = np.bincount(trainval.flag)
    batch_size = baseline.data.samples_per_gpu
    batches = sum((int(n) + batch_size - 1) // batch_size for n in groups)
    report = dict(train=5862, val_added_to_training=5863, trainval=len(trainval),
        evaluation_test=len(test), train_test_disjoint=True,
        val_is_test=True, independent_validation=False,
        batch_size=batch_size, lr=baseline.optimizer.lr,
        max_epochs=baseline.runner.max_epochs,
        final_checkpoint='epoch_12.pth', batches_per_epoch=batches,
        invalid_trainval=sum(len(part.invalid_annotations) for part in parts),
        invalid_test=len(test.invalid_annotations),
        objects_trainval=sum(len(info['ann']['labels'])
                            for part in parts for info in part.data_infos),
        objects_test=sum(len(info['ann']['labels']) for info in test.data_infos))
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
