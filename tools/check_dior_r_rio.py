"""Real-image stability check for the RiO loss; no test-set model selection."""
import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random

import mmcv
import numpy as np
import torch
from mmcv.parallel import collate, scatter

from mmrotate.datasets import build_dataset
from mmrotate.models import build_detector


ROOT = Path(__file__).resolve().parents[1]
CONFIG = 'configs/dior_r/server_trainval/rio.py'
SOURCES = ('mmrotate/models/losses/rotated_rio_loss.py',
           'mmrotate/models/dense_heads/rotated_fcos_head.py', CONFIG)


def source_hashes(config=CONFIG):
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (*SOURCES[:2], 'tools/check_dior_r_rio.py', config)}


def warmup_learning_rates(optimizer, initial_lrs, step, schedule):
    """Apply the configured initial linear warm-up to every parameter group."""
    kind = schedule.get('warmup')
    if kind not in (None, 'linear'):
        raise ValueError('The short check supports linear warm-up or none')
    iterations = schedule.get('warmup_iters', 0)
    ratio = schedule.get('warmup_ratio', 1.)
    factor = (ratio + (1 - ratio) * step / iterations
              if kind and iterations > 0 and step < iterations else 1.)
    for group, initial in zip(optimizer.param_groups, initial_lrs):
        group['lr'] = initial * factor


def stability_errors(report):
    """Reject finite-but-unlearned/collapsed models as well as numerical failure."""
    errors = []
    if report.get('steps', 0) < 1000:
        errors.append('At least 1000 steps are required')
    for key, limit in (('last200_grad_median', 350), ('last200_grad_p95', 3500),
                       ('angle_p95', 1000)):
        value = report.get(key, float('inf'))
        if not math.isfinite(value) or value >= limit:
            errors.append(f'{key} exceeds {limit}: {value}')
    for key, minimum, description in (
            ('positive_wh_fraction', .95, 'Degenerate or missing detections'),
            ('train_image_mAP', .01, 'No meaningful training-image localization'),
            ('regression_image_response', 1e-5, 'No regression response to image content')):
        value = report.get(key, float('nan'))
        if not math.isfinite(value) or value < minimum or (
                key != 'positive_wh_fraction' and value == minimum):
            errors.append(description)
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--config', default=CONFIG)
    parser.add_argument('--report', default='work_dirs/dior_r/rio_preflight_20261016.json')
    args = parser.parse_args()
    os.chdir(ROOT)
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        parser.error(f'Existing report will not be overwritten: {path}')
    lock_path = ROOT / 'work_dirs/dior_r_trainval/server_queue.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open('a')
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error('Another trainval experiment is running')
    report = dict(status='running', protocol_version=2,
                  config=args.config, steps=args.steps,
                  source_sha256=source_hashes(args.config), history=[],
                  note='Stability check on training images; no test mAP used.')

    def save():
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(path)

    save()
    try:
        torch.set_num_threads(2)
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        cfg = mmcv.Config.fromfile(args.config)
        report['resolved_config_sha256'] = hashlib.sha256(
            cfg.pretty_text.encode()).hexdigest()
        snapshots = path.with_suffix('') / 'snapshot'
        snapshots.mkdir(parents=True, exist_ok=False)
        cfg.dump(str(snapshots / 'experiment.py'))
        for name in report['source_sha256']:
            relative = (ROOT / name).resolve().relative_to(ROOT)
            saved = snapshots / 'source' / relative
            saved.parent.mkdir(parents=True, exist_ok=True)
            saved.write_bytes((ROOT / name).read_bytes())
        report['optimizer'] = copy.deepcopy(cfg.optimizer)
        clip = cfg.optimizer_config.grad_clip
        report['clip_max_norm'] = clip.max_norm
        train = build_dataset(cfg.data.train)
        report['training_images'] = len(train)
        model = build_detector(cfg.model).cuda()
        model.init_weights()
        model.train()
        captured = {}

        def capture_head(module, inputs, outputs):
            captured['angles'] = [value.detach().reshape(-1) for value in outputs[2]]
            if not module.training:
                captured['regression'] = torch.cat(
                    [value.detach().reshape(-1) for level in outputs[1:3]
                     for value in level])

        handle = model.bbox_head.register_forward_hook(capture_head)
        from mmcv.runner import build_optimizer
        optimizer = build_optimizer(model, cfg.optimizer)
        initial_lrs = [group['lr'] for group in optimizer.param_groups]
        indices = np.random.RandomState(42).permutation(len(train))
        gradients = []
        for step in range(args.steps):
            sample_ids = indices[(2 * step + np.arange(2)) % len(indices)]
            batch = scatter(collate([train[int(i)] for i in sample_ids],
                                    samples_per_gpu=2), [0])[0]
            warmup_learning_rates(optimizer, initial_lrs, step, cfg.lr_config)
            optimizer.zero_grad()
            output = model.train_step(batch, optimizer)
            if not torch.isfinite(output['loss']):
                raise RuntimeError(f'Nonfinite loss at step {step + 1}')
            output['loss'].backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), clip.max_norm, norm_type=clip.norm_type)
            if not torch.isfinite(norm):
                raise RuntimeError(f'Nonfinite gradient at step {step + 1}')
            gradients.append(float(norm))
            optimizer.step()
            if step == 0 or (step + 1) % 50 == 0:
                record = dict(step=step + 1, **output['log_vars'],
                              grad_norm=float(norm),
                              learning_rate=optimizer.param_groups[0]['lr'],
                              angle_p95=float(torch.quantile(torch.cat(
                                  captured['angles']).abs(), .95)))
                if model.bbox_head.is_scale_angle:
                    record['angle_scale'] = float(model.bbox_head.scale_angle.scale)
                report['history'].append(record)
                save()
                print(json.dumps(record), flush=True)
            del batch, output

        # Inspect predictions only on training images, with inference transforms.
        report.update(last200_grad_median=float(np.median(gradients[-200:])),
                      last200_grad_p95=float(np.percentile(gradients[-200:], 95)),
                      angle_p95=report['history'][-1]['angle_p95'])
        checkpoint = path.with_suffix('.pth')
        torch.save(dict(state_dict={name: value.detach().cpu()
                                   for name, value in model.state_dict().items()},
                        meta=dict(iter=args.steps, diagnostic_only=True,
                                  source_sha256=report['source_sha256'])), checkpoint)
        report['diagnostic_checkpoint'] = str(checkpoint)
        infer_cfg = copy.deepcopy(cfg.data.test)
        infer_cfg.ann_file = 'data/DIOR-R/ImageSets/Main/train.txt'
        infer_cfg.img_prefix = 'data/DIOR-R/JPEGImages-trainval/'
        infer_cfg.test_mode = True
        inference = build_dataset(infer_cfg)
        inference.data_infos = inference.data_infos[:64]
        model.eval()
        detections = []
        results, regression_samples = [], []
        with torch.no_grad():
            for i in range(len(inference)):
                batch = scatter(collate([inference[i]], samples_per_gpu=1), [0])[0]
                result = model(return_loss=False, rescale=True, **batch)[0]
                results.append(result)
                detections.extend(boxes for boxes in result if len(boxes))
                if i < 2:
                    regression_samples.append(captured['regression'].cpu())
        handle.remove()
        boxes = np.concatenate(detections) if detections else np.empty((0, 6))
        if not len(boxes) or not np.isfinite(boxes).all():
            raise RuntimeError('No finite detections after the short training')
        fraction = float(((boxes[:, 2] > 0) & (boxes[:, 3] > 0)).mean())
        report.update(detections=len(boxes), positive_wh_fraction=fraction,
                      checked_train_images=len(inference),
                      train_image_mAP=float(inference.evaluate(
                          results, nproc=2, use_07_metric=True)['mAP']),
                      regression_image_response=float(
                          (regression_samples[1] - regression_samples[0]).abs().mean()
                          / (regression_samples[0].abs().mean() + 1e-6)),
                      peak_gpu_memory_mib=torch.cuda.max_memory_allocated() / 1024**2,
                      gpu=torch.cuda.get_device_name(0))
        errors = stability_errors(report)
        if errors:
            raise RuntimeError('; '.join(errors))
        report['status'] = 'passed'
        save()
        print(json.dumps({key: value for key, value in report.items()
                          if key != 'history'}), flush=True)
    except Exception as error:
        report.update(status='failed', error=str(error))
        save()
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    main()
