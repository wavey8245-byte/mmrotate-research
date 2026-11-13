"""Matched real-image short controls for paper loss ports on DIOR-R FCOS.

Training split only. Same initialized weights, ordered/augmented batches,
optimizer and warmup; all arms retain 800x800, batch 2 and SGD. Reports are
diagnostic, not held-out AP or a completed epoch-12 experiment.
"""
import argparse
import copy
from datetime import datetime
import fcntl
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import random
import time
import traceback

import mmcv
import numpy as np
import torch
from mmcv.ops import box_iou_rotated
from mmcv.parallel import collate, scatter
from mmcv.runner import build_optimizer, save_checkpoint
from mmcv.utils import get_logger
from mmdet.datasets import build_dataloader

from mmrotate.datasets import build_dataset
from mmrotate.models import build_detector
from mmrotate.models.builder import build_loss


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = 'configs/dior_r/server_trainval/'
CONFIGS = dict(riou='riou.py', dcagpl='paper_dcagpl.py',
               bd='paper_bd.py', bd_agbb='paper_bd_agbb.py')


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def digest(paths):
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in paths}


def now():
    return datetime.now().astimezone().isoformat()


def warmup(optimizer, initial_lrs, step, schedule):
    if schedule.get('warmup') != 'linear':
        raise ValueError('Only the current linear warmup is supported')
    count, initial_ratio = schedule.warmup_iters, schedule.warmup_ratio
    factor = initial_ratio + (1 - initial_ratio) * min(step / count, 1.)
    for group, lr in zip(optimizer.param_groups, initial_lrs):
        group['lr'] = lr * factor


def geometry(captured):
    if 'pred' not in captured or captured['pred'].numel() == 0:
        return dict(positives=0, mean_iou=0., weighted_iou=0.,
                    weight_sum=0., iou_ge_half_fraction=0.,
                    zero_wh_fraction=0., mean_angle_error_degrees=0.)
    p, t, w = captured['pred'], captured['target'], captured['weight']
    valid = (p[:, 2:4] > 0).all(-1)
    iou = p.new_zeros(len(p))
    if valid.any():
        iou[valid] = box_iou_rotated(p[valid], t[valid], aligned=True).reshape(-1)
    if not torch.isfinite(iou).all():
        raise RuntimeError('Nonfinite decoded IoU')
    ap = p[:, 4] + (p[:, 2] < p[:, 3]).to(p.dtype) * math.pi / 2
    at = t[:, 4] + (t[:, 2] < t[:, 3]).to(t.dtype) * math.pi / 2
    angle = ((ap - at + math.pi / 2) % math.pi - math.pi / 2).abs()
    return dict(positives=len(p), mean_iou=float(iou.mean()),
                weighted_iou=float((iou * w).sum() / w.sum().clamp_min(1e-6)),
                weight_sum=float(w.sum()),
                iou_ge_half_fraction=float((iou >= .5).float().mean()),
                zero_wh_fraction=float((~valid).float().mean()),
                mean_angle_error_degrees=float(angle.mean() * 180 / math.pi))


def aggregate(values):
    count = sum(x['positives'] for x in values)
    weight = sum(x['weight_sum'] for x in values)
    result = dict(positives=count)
    for key in ('mean_iou', 'iou_ge_half_fraction', 'zero_wh_fraction',
                'mean_angle_error_degrees'):
        result[key] = sum(x[key] * x['positives'] for x in values) / max(count, 1)
    result['weighted_iou'] = sum(x['weighted_iou'] * x['weight_sum']
                                 for x in values) / max(weight, 1e-6)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--arms', nargs='+', choices=list(CONFIGS),
                        default=list(CONFIGS))
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--anchor-images', type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.steps <= 2000 or args.anchor_images < 2:
        parser.error('Require 1..2000 steps and at least 2 anchor images')
    os.chdir(ROOT)
    output = Path(args.work_dir).resolve()
    if not output.is_relative_to(ROOT / 'work_dirs'):
        parser.error('Output must be inside this repository work_dirs')
    lock_path = ROOT / 'work_dirs/dior_r_trainval/server_queue.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('Another trainval job holds the shared experiment lock')
        output.mkdir(parents=True, exist_ok=False)
        sources = [str(Path(__file__).relative_to(ROOT)),
                   'mmrotate/models/losses/paper_rotated_losses.py',
                   'mmrotate/models/losses/rotated_iou_loss.py',
                   'mmrotate/models/dense_heads/rotated_fcos_head.py',
                   'mmrotate/datasets/dior.py',
                   'mmrotate/core/bbox/coder/distance_angle_point_coder.py']
        sources += [CONFIG_ROOT + CONFIGS[name] for name in CONFIGS]
        report = dict(status='starting', created_at=now(), steps_per_arm=args.steps,
                      pid=os.getpid(), seed=42, arms={}, source_sha256=digest(sources),
                      scope='Training-image diagnostics only; no test data or final mAP.',
                      config_names=CONFIGS, active_arm=None)

        def save():
            path = output / 'report.json'
            temporary = path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(report, indent=2) + '\n')
            temporary.replace(path)

        save()
        try:
            torch.set_num_threads(2)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA is required for this 800px batch-2 probe')
            report['gpu'] = torch.cuda.get_device_name(0)
            report['versions'] = dict(torch=torch.__version__, mmcv=mmcv.__version__)
            configs = {name: mmcv.Config.fromfile(CONFIG_ROOT + CONFIGS[name])
                       for name in CONFIGS}
            base = configs['riou']
            for name, cfg in configs.items():
                cfg.dump(str(output / (name + '_resolved.py')))
                stripped = copy.deepcopy(cfg.model)
                stripped.bbox_head.loss_bbox = base.model.bbox_head.loss_bbox
                if stripped != base.model:
                    raise RuntimeError('Model differs beyond loss: ' + name)
                for key in ('data', 'optimizer', 'optimizer_config', 'lr_config',
                            'runner', 'evaluation', 'seed', 'load_from', 'resume_from'):
                    if cfg[key] != base[key]:
                        raise RuntimeError('Training setting differs: ' + name + '/' + key)
            snapshot = output / 'snapshot'
            for name in sources:
                path = snapshot / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((ROOT / name).read_bytes())
            report['optimizer'] = copy.deepcopy(base.optimizer)
            report['warmup'] = copy.deepcopy(base.lr_config)
            report['clip'] = copy.deepcopy(base.optimizer_config.grad_clip)
            seed(42)
            train = build_dataset(base.data.train)
            report['training_images'] = len(train)
            # Fixed augmented training anchors for objective-independent
            # decoded localization measurements, identical in every arm.
            seed(314159)
            anchor_ids = np.random.RandomState(314159).choice(
                len(train), args.anchor_images, replace=False)
            anchors = [train[int(index)] for index in anchor_ids]
            report['anchor_train_indices'] = anchor_ids.tolist()
            report['anchor_image_names'] = [x['img_metas'].data['ori_filename']
                                            for x in anchors]
            # A second, unaugmented training-only view for diagnostic AP.
            infer_cfg = copy.deepcopy(base.data.train)
            if not isinstance(infer_cfg.ann_file, list):
                raise RuntimeError('Expected current train+val training protocol')
            infer_cfg.ann_file = infer_cfg.ann_file[0]  # train.txt, never test.txt
            infer_cfg.pipeline = copy.deepcopy(base.data.val.pipeline)
            infer_cfg.test_mode = True
            inference = build_dataset(infer_cfg)
            chosen = np.random.RandomState(271828).choice(
                len(inference), args.anchor_images, replace=False)
            inference.data_infos = [inference.data_infos[int(i)] for i in chosen]
            report['diagnostic_ap_train_ids'] = [x['id'] for x in inference.data_infos]
            seed(42)
            model = build_detector(base.model).cuda()
            get_logger('mmcv').setLevel(logging.ERROR)
            model.init_weights()
            initial = {name: tensor.detach().cpu().clone()
                       for name, tensor in model.state_dict().items()}
            state_hash = hashlib.sha256()
            for name, tensor in initial.items():
                state_hash.update(name.encode())
                state_hash.update(tensor.numpy().tobytes())
            report['initial_state_sha256'] = state_hash.hexdigest()
            reference_hashes = None
            save()

            for name in args.arms:
                start = time.monotonic()
                seed(42)
                model.bbox_head.loss_bbox = build_loss(configs[name].model.bbox_head.loss_bbox)
                model.load_state_dict(initial, strict=True)
                model.train()
                captured = {}
                original_forward = model.bbox_head.loss_bbox.forward

                def capture(pred, target, *a, **kw):
                    weight = kw.get('weight')
                    if weight is None:
                        weight = pred.new_ones(len(pred))
                    captured.update(pred=pred.detach(), target=target.detach(),
                                    weight=weight.detach())
                    return original_forward(pred, target, *a, **kw)

                def angle_hook(module, inputs, outputs):
                    captured['angles'] = torch.cat([x.detach().reshape(-1)
                                                    for x in outputs[2]])

                model.bbox_head.loss_bbox.forward = capture
                handle = model.bbox_head.register_forward_hook(angle_hook)
                optimizer = build_optimizer(model, configs[name].optimizer)
                initial_lrs = [x['lr'] for x in optimizer.param_groups]
                loader = build_dataloader(
                    train, samples_per_gpu=2, workers_per_gpu=0, num_gpus=1,
                    dist=False, shuffle=True, seed=42)
                report['active_arm'] = name
                arm = dict(status='running', started_at=now(), history=[],
                           loss=dict(configs[name].model.bbox_head.loss_bbox),
                           augmented_batch_sha256=[])
                report['arms'][name] = arm
                report['status'] = 'running'
                save()

                def anchor_measure():
                    model.train()
                    stats = []
                    with torch.no_grad():
                        for offset in range(0, len(anchors), 2):
                            batch = scatter(collate(anchors[offset:offset+2],
                                                    samples_per_gpu=2), [0])[0]
                            captured.pop('pred', None)
                            model(return_loss=True, **batch)
                            stats.append(geometry(captured))
                    return aggregate(stats)

                def diagnostic_ap():
                    model.eval()
                    results = []
                    with torch.no_grad():
                        for index in range(len(inference)):
                            batch = scatter(collate([inference[index]],
                                                    samples_per_gpu=1), [0])[0]
                            prediction = model(return_loss=False, rescale=True, **batch)
                            if not all(np.isfinite(x).all() for x in prediction[0]):
                                raise RuntimeError('Nonfinite inference output')
                            results.extend(prediction)
                    return float(inference.evaluate(results, nproc=1,
                                 use_07_metric=True, logger='silent')['mAP'])

                arm['anchors_before'] = anchor_measure()
                arm['train_ap_before'] = diagnostic_ap()
                model.train()
                # Reset after all measurements: augmentation RNG is exactly
                # the same at the first training batch for every arm.
                seed(42)
                torch.cuda.reset_peak_memory_stats()
                norms = []
                for step, cpu_batch in enumerate(loader):
                    if step >= args.steps:
                        break
                    image_hash = hashlib.sha256(
                        cpu_batch['img'].data[0].numpy().tobytes()).hexdigest()
                    arm['augmented_batch_sha256'].append(image_hash)
                    if reference_hashes is not None and image_hash != reference_hashes[step]:
                        raise RuntimeError('Different augmented batch at step ' + str(step+1))
                    batch = scatter(cpu_batch, [0])[0]
                    warmup(optimizer, initial_lrs, step, configs[name].lr_config)
                    optimizer.zero_grad(set_to_none=True)
                    captured.pop('pred', None)
                    result = model.train_step(batch, optimizer)
                    if not torch.isfinite(result['loss']):
                        raise RuntimeError('Nonfinite training loss at step ' + str(step+1))
                    result['loss'].backward()
                    clipping = configs[name].optimizer_config.grad_clip
                    norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), clipping.max_norm, clipping.norm_type)
                    if not torch.isfinite(norm):
                        raise RuntimeError('Nonfinite gradient at step ' + str(step+1))
                    optimizer.step()
                    norms.append(float(norm))
                    if step == 0 or (step+1) % 25 == 0 or step+1 == args.steps:
                        angles = captured['angles'].abs()
                        record = dict(step=step+1, **result['log_vars'],
                                      grad_norm=float(norm),
                                      learning_rate=optimizer.param_groups[0]['lr'],
                                      angle_p95=float(torch.quantile(angles, .95)),
                                      angle_scale=float(model.bbox_head.scale_angle.scale),
                                      elapsed_seconds=time.monotonic()-start,
                                      **geometry(captured))
                        arm['history'].append(record)
                        print(name, json.dumps(record), flush=True)
                        save()
                    del batch, cpu_batch, result
                if len(norms) != args.steps:
                    raise RuntimeError('Training loader ended before requested steps')
                if reference_hashes is None:
                    reference_hashes = arm['augmented_batch_sha256']
                arm['identical_augmented_batches'] = (
                    arm['augmented_batch_sha256'] == reference_hashes)
                arm['anchors_after'] = anchor_measure()
                arm['train_ap_after'] = diagnostic_ap()
                arm['last100_gradient_median'] = float(np.median(norms[-100:]))
                arm['last100_gradient_p95'] = float(np.percentile(norms[-100:], 95))
                arm['peak_memory_allocated_mib'] = torch.cuda.max_memory_allocated()/1024**2
                arm['peak_memory_reserved_mib'] = torch.cuda.max_memory_reserved()/1024**2
                arm['elapsed_seconds'] = time.monotonic()-start
                arm['steps_completed'] = len(norms)
                model.bbox_head.loss_bbox.forward = original_forward
                handle.remove()
                save_checkpoint(model, str(output / (name + '_step_' + str(args.steps) + '.pth')),
                                optimizer=optimizer,
                                meta=dict(probe_steps=args.steps, seed=42,
                                          config=configs[name].pretty_text,
                                          scope='Diagnostic short run, not a final epoch checkpoint'))
                arm.update(status='completed_short_probe', finished_at=now())
                save()
                print('RESULT', name, json.dumps({k: v for k, v in arm.items()
                      if k not in ('history', 'augmented_batch_sha256')}), flush=True)
                del optimizer, loader, captured
                torch.cuda.empty_cache()
            report['sources_unchanged'] = digest(sources) == report['source_sha256']
            if not report['sources_unchanged']:
                raise RuntimeError('A source file changed during the experiment')
            report.update(status='completed_short_controls', active_arm=None,
                          finished_at=now())
            save()
            print('COMPLETE', output / 'report.json', flush=True)
        except Exception as error:
            report.update(status='failed', error=str(error),
                          traceback=traceback.format_exc(), finished_at=now())
            save()
            raise


if __name__ == '__main__':
    main()
