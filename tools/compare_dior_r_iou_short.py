"""Controlled short training: original WF versus native linear/log IoU.

Identical initialized weights, augmented batches, SGD, warmup and trainval.
Diagnostics use training samples only; no test-based tuning or full runs.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

import mmcv
import numpy as np
import torch
from mmcv.ops import box_iou_rotated
from mmcv.parallel import collate, scatter
from mmcv.utils import get_logger
from mmdet.datasets import build_dataloader

from mmrotate.datasets import build_dataset
from mmrotate.models import build_detector
from mmrotate.models.builder import build_loss


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--arms', nargs='+', choices=['log_iou', 'linear_iou',
                        'original_wf', 'wf_aux_log'],
                        default=['log_iou', 'linear_iou', 'original_wf'])
    parser.add_argument('--report', default='work_dirs/dior_r_trainval/iou_short_control.json')
    args = parser.parse_args()
    if not 1 <= args.steps <= 2000:
        parser.error('Short controls require 1..2000 steps')
    destination = Path(args.report)
    if destination.exists():
        parser.error(f'Report already exists: {destination}')
    torch.set_num_threads(2)
    cfg = mmcv.Config.fromfile('configs/dior_r/server_trainval/riou.py')
    dataset = build_dataset(cfg.data.train)
    seed(42)
    model = build_detector(cfg.model).cuda()
    get_logger('mmcv').setLevel(40)
    model.init_weights()
    initial = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    initial_digest = hashlib.sha256()
    for key, value in initial.items():
        initial_digest.update(key.encode())
        initial_digest.update(value.numpy().tobytes())
    # Cache identically augmented training anchors before starting any arm.
    seed(63)
    ids = np.random.RandomState(63).choice(len(dataset), 16, replace=False)
    anchor_samples = [dataset[int(i)] for i in ids]
    report = dict(steps=args.steps, seed=42, trainval=len(dataset),
                  initial_state_sha256=initial_digest.hexdigest(),
                  anchor_train_ids=ids.tolist(), arms={})
    arms = {
        'log_iou': dict(type='RotatedIoULoss', mode='log', loss_weight=1.),
        'linear_iou': dict(type='RotatedIoULoss', mode='linear', loss_weight=1.),
        'original_wf': dict(type='RotatedWFIoULoss', tau=.7, eps=1e-7, loss_weight=1.),
        'wf_aux_log': dict(type='RotatedIoUWithWFAuxLoss', wf_weight=.1,
                           tau=.7, eps=1e-6, loss_weight=1.),
    }
    baseline_filenames = None
    baseline_hashes = None
    for arm in args.arms:
        loss_cfg = arms[arm]
        seed(42)
        model.bbox_head.loss_bbox = build_loss(loss_cfg)
        model.load_state_dict(initial, strict=True)
        model.train()
        captured = {}
        original = model.bbox_head.loss_bbox.forward

        def capture(pred, target, *a, **kw):
            captured.update(pred=pred.detach(), target=target.detach())
            return original(pred, target, *a, **kw)

        model.bbox_head.loss_bbox.forward = capture
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.optimizer.lr,
                                    momentum=cfg.optimizer.momentum,
                                    weight_decay=cfg.optimizer.weight_decay)
        loader = build_dataloader(dataset, samples_per_gpu=2, workers_per_gpu=2,
                                  num_gpus=1, dist=False, shuffle=True, seed=42)
        history, filenames, hashes = [], [], {}

        def measure():
            pred, target = captured['pred'], captured['target']
            iou = box_iou_rotated(pred, target, aligned=True).reshape(-1)
            angle = ((pred[:, 4] + (pred[:, 2] < pred[:, 3]).float()*np.pi/2
                      - target[:, 4] + np.pi/2) % np.pi - np.pi/2).abs()
            return dict(positives=len(pred), mean_iou=float(iou.mean()),
                        iou_ge_half_fraction=float((iou >= .5).float().mean()),
                        mean_angle_error_degrees=float(angle.mean()*180/np.pi),
                        zero_wh_fraction=float(((pred[:, 2] <= 0) |
                                                (pred[:, 3] <= 0)).float().mean()))

        for step, cpu_batch in enumerate(loader):
            if step >= args.steps:
                break
            batch = scatter(cpu_batch, [0])[0]
            filenames.append([item['filename'] for item in batch['img_metas']])
            if step % 50 == 0:
                hashes[str(step)] = hashlib.sha256(
                    batch['img'].detach().cpu().numpy().tobytes()).hexdigest()
            ratio = 1/3 + 2/3 * min(step/500, 1)
            optimizer.param_groups[0]['lr'] = cfg.optimizer.lr * ratio
            optimizer.zero_grad()
            output = model.train_step(batch, optimizer)
            output['loss'].backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 35)
            if not torch.isfinite(grad):
                raise RuntimeError(f'{arm}: nonfinite gradient at step {step+1}')
            optimizer.step()
            if step == 0 or (step+1) % 50 == 0:
                record = dict(step=step+1, **output['log_vars'], grad_norm=float(grad),
                              angle_scale=float(model.bbox_head.scale_angle.scale),
                              **measure())
                history.append(record)
                print(arm, json.dumps(record), flush=True)
            del output, batch, cpu_batch
        del loader
        if baseline_filenames is None:
            baseline_filenames, baseline_hashes = filenames, hashes
        else:
            assert filenames == baseline_filenames, 'Different sample ordering'
            assert hashes == baseline_hashes, 'Different input augmentation'
        anchors = []
        with torch.no_grad():
            for i in range(0, len(anchor_samples), 2):
                batch = scatter(collate(anchor_samples[i:i+2], samples_per_gpu=2), [0])[0]
                model(return_loss=True, **batch)
                anchors.append(measure())
        count = sum(value['positives'] for value in anchors)
        anchor_stats = {key: sum(value[key]*value['positives'] for value in anchors)/count
                        for key in anchors[0] if key != 'positives'}
        report['arms'][arm] = dict(loss=loss_cfg, history=history,
                                   identical_input_batches=True,
                                   augmented_input_hashes=hashes,
                                   anchor_positives=count, anchor_metrics=anchor_stats)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + '\n')
        print('RESULT', arm, json.dumps(anchor_stats), flush=True)
        # Prevent a later arm from retaining the previous closure/module graph.
        model.bbox_head.loss_bbox.forward = original
    print('COMPLETE', destination, flush=True)


if __name__ == '__main__':
    main()
