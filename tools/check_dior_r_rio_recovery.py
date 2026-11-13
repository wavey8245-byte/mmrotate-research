"""Controlled training-only checks of RiO angular parameterization recovery."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import types

import mmcv
from mmcv.parallel import collate, scatter
import numpy as np
import torch

from mmrotate.datasets import build_dataset
from mmrotate.models import build_detector


def seed():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--arms', nargs='+', default=['fixed_scale', 'bounded'],
                        choices=['original', 'fixed_scale', 'bounded',
                                 'kld_only', 'kld_l1', 'sp_reduced'])
    parser.add_argument('--angle-limit', type=float, default=10000.)
    parser.add_argument('--report', default='work_dirs/dior_r/rio_recovery_20261016.json')
    args = parser.parse_args()
    torch.set_num_threads(2)
    seed()
    cfg = mmcv.Config.fromfile('configs/dior_r/server_trainval/rio.py')
    train = build_dataset(cfg.data.train)
    infer_cfg = copy.deepcopy(cfg.data.test)
    infer_cfg.ann_file = 'data/DIOR-R/ImageSets/Main/train.txt'
    infer_cfg.img_prefix = 'data/DIOR-R/JPEGImages-trainval/'
    infer_cfg.test_mode = True
    inference = build_dataset(infer_cfg)
    inference.data_infos = inference.data_infos[:64]
    model = build_detector(cfg.model).cuda()
    model.init_weights()
    initial = {name: value.detach().cpu().clone()
               for name, value in model.state_dict().items()}
    head = model.bbox_head
    original_forward = head.forward_single
    current = {}

    def angular_forward(self, *values, **kwargs):
        outputs = list(original_forward(*values, **kwargs))
        raw = outputs[2]
        current['raw'].append(raw.detach().reshape(-1))
        if current['arm'] == 'bounded':
            outputs[2] = math.pi * (raw.sigmoid() - .5)
        current['angles'].append(outputs[2].detach().reshape(-1))
        if not self.training:
            current['regression'] = torch.cat([
                value.detach().reshape(-1) for levels in outputs[1:3]
                for value in levels])
        return tuple(outputs)

    head.forward_single = types.MethodType(angular_forward, head)
    indices = np.random.RandomState(42).permutation(len(train))
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = dict(status='running', steps=args.steps,
                  note='Training images only, same initialization/batches/SGD in each arm.',
                  purpose='Component diagnosis; this does not authorize an automatic full run.',
                  angle_limit=args.angle_limit,
                  source_sha256={name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                    for name in ('mmrotate/models/dense_heads/rotated_fcos_head.py',
                                 'mmrotate/models/losses/rotated_rio_loss.py',
                                 'tools/check_dior_r_rio_recovery.py')}, arms={})

    def save():
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(path)

    save()
    for arm in args.arms:
        seed()
        model.load_state_dict(initial)
        head.is_scale_angle = arm not in ('fixed_scale', 'bounded')
        head.loss_bbox.kld_weight = 5.
        head.loss_bbox.spatial_weight = 0. if arm == 'kld_only' else 5.
        head.loss_bbox.angle_weight = (0. if arm in ('kld_only', 'kld_l1')
                                       else .2 if arm == 'sp_reduced' else 5.)
        model.train()
        options = dict(cfg.optimizer)
        options.pop('type')
        optimizer = torch.optim.SGD(model.parameters(), **options)
        record = dict(status='running', history=[], train_image_metrics=None,
                      loss_weights=[head.loss_bbox.kld_weight,
                                    head.loss_bbox.spatial_weight,
                                    head.loss_bbox.angle_weight],
                      learnable_angle_scale=head.is_scale_angle)
        report['arms'][arm] = record
        gradients = []
        for step in range(args.steps):
            current.update(arm=arm, raw=[], angles=[])
            sample_ids = indices[(2 * step + np.arange(2)) % len(indices)]
            batch = scatter(collate([train[int(i)] for i in sample_ids],
                                    samples_per_gpu=2), [0])[0]
            optimizer.param_groups[0]['lr'] = cfg.optimizer.lr * (
                1 / 3 + 2 / 3 * min(step / 500, 1))
            optimizer.zero_grad()
            output = model.train_step(batch, optimizer)
            output['loss'].backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 35)
            gradients.append(float(norm))
            raw = torch.cat(current['raw'])
            angles = torch.cat(current['angles'])
            if not torch.isfinite(norm) or not torch.isfinite(output['loss']):
                record.update(status='failed', error='Nonfinite loss or gradient')
                break
            optimizer.step()
            if step == 0 or (step + 1) % 50 == 0:
                item = dict(step=step + 1, **output['log_vars'],
                            grad_norm=float(norm),
                            raw_angle_p95=float(torch.quantile(raw.abs(), .95)),
                            angle_p95=float(torch.quantile(angles.abs(), .95)),
                            angle_scale=float(head.scale_angle.scale))
                record['history'].append(item)
                print(arm, json.dumps(item), flush=True)
                save()
                if item['angle_p95'] > args.angle_limit:
                    record.update(status='failed', error='Angular output runaway')
                    break
            del batch, output, raw, angles
        record['last200_grad_median'] = float(np.median(gradients[-200:]))
        record['last200_grad_p95'] = float(np.percentile(gradients[-200:], 95))
        model.eval()
        results = []
        responses = []
        with torch.no_grad():
            for i in range(len(inference)):
                current.update(raw=[], angles=[])
                batch = scatter(collate([inference[i]], samples_per_gpu=1), [0])[0]
                results.extend(model(return_loss=False, rescale=True, **batch))
                if i < 2:
                    responses.append(current['regression'].cpu())
        record['train_image_metrics'] = inference.evaluate(
            results, nproc=2, use_07_metric=True)
        detections = [boxes for image in results for boxes in image if len(boxes)]
        boxes = np.concatenate(detections) if detections else np.empty((0, 6))
        record['detections'] = len(boxes)
        record['positive_wh_fraction'] = float(
            ((boxes[:, 2] > 0) & (boxes[:, 3] > 0)).mean()) if len(boxes) else 0
        record['regression_image_response'] = float(
            (responses[1] - responses[0]).abs().mean()
            / (responses[0].abs().mean() + 1e-6))
        if record['status'] == 'running':
            record['short_check_passed'] = bool(record['last200_grad_median'] < 350
                                and record['last200_grad_p95'] < 3500
                                and record['positive_wh_fraction'] >= .95
                                and record['train_image_metrics']['mAP'] > .01
                                and record['regression_image_response'] > 1e-5)
            record['status'] = 'completed'
        checkpoints = path.with_suffix('')
        checkpoints.mkdir(exist_ok=True)
        checkpoint = checkpoints / f'{arm}_step_{step + 1}.pth'
        torch.save(dict(state_dict={name: value.detach().cpu()
                                    for name, value in model.state_dict().items()},
                        steps=step + 1, arm=arm, loss_weights=record['loss_weights'],
                        source_sha256=report['source_sha256']), checkpoint)
        record['diagnostic_checkpoint'] = str(checkpoint)
        print('RESULT', arm, json.dumps(record), flush=True)
        save()
        del optimizer
    report['status'] = 'completed'
    save()


if __name__ == '__main__':
    main()
