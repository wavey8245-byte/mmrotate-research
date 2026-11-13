"""Read-only WF checkpoint diagnostics on identical real training batches.

No optimizer steps, checkpoint changes, or test-based model selection.
"""
import argparse
import json
import random
from pathlib import Path

import mmcv
import numpy as np
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmcv.ops import box_iou_rotated, diff_iou_rotated_2d

from mmrotate.datasets import build_dataset
from mmrotate.models import build_detector
from mmrotate.models.losses.rotated_wf_iou_loss import (
    rotated_wf_geometry, wf_weighting_factor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batches', type=int, default=8)
    parser.add_argument('--report', default='work_dirs/dior_r_trainval/wf_diagnosis.json')
    args = parser.parse_args()
    torch.set_num_threads(2)
    cfg = mmcv.Config.fromfile('configs/dior_r/server_trainval/wf.py')
    # Remove augmentation for a repeatable checkpoint comparison.
    for transform in cfg.data.train.pipeline:
        if transform.type == 'RRandomFlip':
            transform.flip_ratio = [0., 0., 0.]
    dataset = build_dataset(cfg.data.train)
    sample_ids = np.random.RandomState(63).choice(
        len(dataset), args.batches * 2, replace=False)
    report = dict(sample_ids=sample_ids.tolist(), batches=args.batches, arms={})
    arms = {
        'trainval_wf_epoch1': 'work_dirs/dior_r_trainval/fcos_wf_bs2_800/epoch_1.pth',
        'trainval_wf_epoch4': 'work_dirs/dior_r_trainval/fcos_wf_bs2_800/epoch_4.pth',
        'trainval_wf_epoch12': 'work_dirs/dior_r_trainval/fcos_wf_bs2_800/epoch_12.pth',
        'old_split_wf_epoch12': 'work_dirs/dior_r/fcos_wf_bs2_800/epoch_12.pth',
        'trainval_log_iou_epoch12': 'work_dirs/dior_r_trainval/fcos_riou_bs2_800/epoch_12.pth',
    }
    model = build_detector(cfg.model).cuda()
    captured = {}
    original = model.bbox_head.loss_bbox.forward

    def capture(pred, target, *a, **kw):
        captured.update(pred=pred, target=target, weight=kw['weight'])
        return original(pred, target, *a, **kw)

    model.bbox_head.loss_bbox.forward = capture
    for arm, checkpoint in arms.items():
        load_checkpoint(model, checkpoint, map_location='cpu', strict=True)
        model.train()
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        records = []
        for i in range(args.batches):
            batch = scatter(collate([dataset[int(j)] for j in sample_ids[2*i:2*i+2]],
                                    samples_per_gpu=2), [0])[0]
            losses = model(return_loss=True, **batch)
            pred, target = captured['pred'], captured['target']
            gradient = torch.autograd.grad(losses['loss_bbox'], pred)[0]
            native_pred = pred.detach().clone().requires_grad_()
            native_iou = diff_iou_rotated_2d(
                native_pred[None], target.detach()[None]).reshape(-1)
            weights = captured['weight'].detach()
            native_loss = ((1-native_iou) * weights).sum() / weights.sum().clamp(min=1e-6)
            native_gradient = torch.autograd.grad(native_loss, native_pred)[0]
            with torch.no_grad():
                iou, ratio = rotated_wf_geometry(pred, target)
                factor = wf_weighting_factor(iou, ratio)
                raw = (1 - factor * iou).clamp(min=0)
                ref = box_iou_rotated(pred, target, aligned=True).reshape(-1)
                records.append(dict(
                    positives=len(pred), iou=iou.cpu(), ratio=ratio.cpu(),
                    factor=factor.cpu(), raw=raw.cpu(),
                    zero_wh=((pred[:, 2] <= 0) | (pred[:, 3] <= 0)).cpu(),
                    zero_gradient=(gradient.abs().sum(-1) == 0).cpu(),
                    opposes_iou=((gradient * native_gradient).sum(-1) < -1e-12).cpu(),
                    angle_error=((pred[:, 4] + (pred[:, 2] < pred[:, 3]).float()*np.pi/2
                                  - target[:, 4] + np.pi/2) % np.pi - np.pi/2).abs().cpu(),
                    iou_error=(iou-ref).abs().cpu(),
                    losses={k: float(v.detach()) for k, v in losses.items()}))
            del losses, batch, captured['pred'], captured['target'], captured['weight']
        values = {key: torch.cat([r[key] for r in records]) for key in records[0]
                  if key not in ('positives', 'losses')}
        iou, raw = values['iou'], values['raw']
        stats = dict(
            checkpoint=checkpoint, positives=len(iou),
            mean_iou=float(iou.mean()), median_iou=float(iou.median()),
            iou_ge_half_fraction=float((iou >= .5).float().mean()),
            no_overlap_fraction=float((iou == 0).float().mean()),
            zero_wh_fraction=float(values['zero_wh'].float().mean()),
            zero_loss_fraction=float((raw == 0).float().mean()),
            bad_iou_zero_loss_fraction=float(((iou < .3) & (raw == 0)).float().mean()),
            zero_box_gradient_fraction=float(values['zero_gradient'].float().mean()),
            wf_gradient_opposes_iou_fraction=float(values['opposes_iou'].float().mean()),
            mean_canonical_angle_error_degrees=float(values['angle_error'].mean()*180/np.pi),
            max_iou_reference_error=float(values['iou_error'].max()),
            reg_bias=model.bbox_head.conv_reg.bias.detach().cpu().tolist(),
            angle_scale=float(model.bbox_head.scale_angle.scale),
            distance_scales=[float(s.scale) for s in model.bbox_head.scales],
            mean_losses={k: sum(r['losses'][k] for r in records)/len(records)
                         for k in records[0]['losses']})
        report['arms'][arm] = stats
        print(arm, json.dumps(stats), flush=True)
        destination = Path(args.report)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + '\n')
    # Simple controlled geometry: identical centered 10:2 rectangles, only
    # relative angle changes. Check whether poor localization can minimize WF.
    sweep = []
    target = torch.tensor([[0., 0., 10., 2., 0.]], device='cuda')
    for degrees in (0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 60, 75, 90):
        pred = target.clone()
        pred[:, 4] = degrees*np.pi/180
        iou, ratio = rotated_wf_geometry(pred, target)
        factor = wf_weighting_factor(iou, ratio)
        sweep.append(dict(angle_degrees=degrees, iou=float(iou), ratio=float(ratio),
                          weight=float(factor), loss=float((1-factor*iou).clamp(min=0))))
    report['centered_elongated_angle_sweep'] = sweep
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print('SWEEP', json.dumps(sweep), flush=True)
    print('REPORT', args.report, flush=True)


if __name__ == '__main__':
    main()
