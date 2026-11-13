"""Controlled short runs to diagnose decoded Smooth L1 regression failure.

Uses real 800x800 batches with the same initial weights, sampling, augmentations,
SGD and warmup in each arm. Does not overwrite any experiment checkpoint.
"""
import argparse
import json
import math
import random
from pathlib import Path
import types

import mmcv
import numpy as np
import torch
from mmcv.parallel import collate, scatter

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
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--val-images', type=int, default=64)
    parser.add_argument('--arms', nargs='+', choices=[
        'decoded_raw', 'decoded_periodic', 'encoded_periodic', 'configured'], default=[
        'decoded_raw', 'decoded_periodic', 'encoded_periodic'])
    parser.add_argument('--report', default='work_dirs/dior_r/smooth_l1_diagnosis.json')
    args = parser.parse_args()
    if args.steps < 1 or args.val_images < 1:
        parser.error('steps and val-images must be positive')
    torch.set_num_threads(2)
    cfg = mmcv.Config.fromfile(
        'configs/dior_r/rotated_fcos_r50_fpn_1x_dior_r_le90_smooth_l1_bs2.py')
    configured_loss = build_loss(cfg.model.bbox_head.loss_bbox)
    # Always diagnose the original loss, even after updating the production config.
    cfg.model.bbox_head.loss_bbox.type = 'RotatedSmoothL1Loss'
    train = build_dataset(cfg.data.train)
    val = build_dataset(cfg.data.val)
    val.data_infos = val.data_infos[:args.val_images]
    seed(42)
    model = build_detector(cfg.model).cuda()
    model.init_weights()
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    original_loss = model.bbox_head.loss_bbox.forward
    legacy_loss = model.bbox_head.loss_bbox
    original_decode = model.bbox_head.bbox_coder.decode
    indices = np.random.RandomState(42).permutation(len(train))
    reports = {}

    def periodic_loss(self, pred, target, *a, **kw):
        delta = pred - target
        angle = (delta[:, 4:] + math.pi / 2) % math.pi - math.pi / 2
        residual = torch.cat([delta[:, :4], angle], dim=1)
        return original_loss(residual, torch.zeros_like(residual), *a, **kw)

    for arm in args.arms:
        seed(42)
        model.load_state_dict(initial)
        model.bbox_head.loss_bbox = configured_loss if arm == 'configured' else legacy_loss
        if arm != 'configured':
            legacy_loss.forward = (original_loss if arm == 'decoded_raw'
                else types.MethodType(periodic_loss, legacy_loss))
        model.bbox_head.bbox_coder.decode = (lambda points, distances: distances
            if arm == 'encoded_periodic' else original_decode(points, distances))
        options = dict(cfg.optimizer)
        options.pop('type')
        optimizer = torch.optim.SGD(model.parameters(), **options)
        model.train()
        history = []
        for step in range(args.steps):
            sample_ids = indices[(2 * step + np.arange(2)) % len(indices)]
            batch = scatter(collate([train[int(i)] for i in sample_ids],
                                    samples_per_gpu=2), [0])[0]
            ratio = 1 / 3 + 2 / 3 * min(step / 500, 1)
            optimizer.param_groups[0]['lr'] = cfg.optimizer.lr * ratio
            optimizer.zero_grad()
            output = model.train_step(batch, optimizer)
            output['loss'].backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 35)
            if not torch.isfinite(grad):
                raise RuntimeError(f'{arm}: nonfinite gradient at {step}')
            optimizer.step()
            if step == 0 or (step + 1) % 50 == 0:
                record = dict(step=step + 1, **output['log_vars'],
                    grad_norm=float(grad), angle_scale=float(model.bbox_head.scale_angle.scale))
                history.append(record)
                print(arm, record, flush=True)
            del batch, output
        model.bbox_head.bbox_coder.decode = original_decode
        model.eval()
        results = []
        with torch.no_grad():
            for i in range(len(val)):
                batch = scatter(collate([val[i]], samples_per_gpu=1), [0])[0]
                results += model(return_loss=False, rescale=True, **batch)
        metric = val.evaluate(results, nproc=2, use_07_metric=True)
        detections = [r for image in results for r in image if len(r)]
        boxes = np.concatenate(detections) if detections else np.empty((0, 6))
        reports[arm] = dict(history=history, val_images=len(val), metrics=metric,
            loss_type=type(model.bbox_head.loss_bbox).__name__,
            loss_definition=type(model.bbox_head.loss_bbox).__doc__,
            detections=len(boxes), positive_wh_fraction=float(
                ((boxes[:, 2] > 0) & (boxes[:, 3] > 0)).mean()) if len(boxes) else 0)
        print('RESULT', arm, reports[arm], flush=True)
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(reports, indent=2) + '\n')


if __name__ == '__main__':
    main()
