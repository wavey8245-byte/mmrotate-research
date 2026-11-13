"""Validate DIOR-R splits and run a training/inference smoke check.

Run from the repository root with the mmrotate034 Python environment.
This performs one optimizer step, not a full experiment.
CUDA and the experiment's input size are used by default. Optional CPU and
size overrides allow a lightweight check while the GPU is training.
"""
import argparse
import json
from pathlib import Path

import mmcv
import numpy as np
import torch
from mmcv.parallel import collate, scatter
from mmrotate.datasets import build_dataset
from mmrotate.models import build_detector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=
                        'configs/dior_r/rotated_fcos_r50_fpn_1x_dior_r_le90.py')
    parser.add_argument('--report', default='work_dirs/dior_r/smoke_report.json')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--image-size', type=int, default=None,
                        help='Override size only for this smoke check')
    args = parser.parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    if args.image_size is not None:
        if args.image_size <= 0:
            parser.error('--image-size must be positive')
        for name in ('train', 'val', 'test'):
            for step in cfg.data[name].pipeline:
                if 'img_scale' in step:
                    step['img_scale'] = (args.image_size, args.image_size)
    torch.manual_seed(42)
    np.random.seed(42)
    if args.device == 'cpu':
        torch.set_num_threads(2)
    report = dict(config=args.config, device=args.device,
                  smoke_image_size_override=args.image_size, splits={})
    splits = {}
    train = None
    val = None
    for name in ('train', 'val', 'test'):
        dataset_cfg = cfg.data[name].copy()
        dataset_cfg.update(test_mode=True, filter_empty_gt=False)
        dataset = build_dataset(dataset_cfg)
        splits[name] = {info['id'] for info in dataset.data_infos}
        assert len(splits[name]) == len(dataset), 'Duplicate image ids'
        report['splits'][name] = dict(
            images=len(dataset),
            objects=sum(len(info['ann']['labels']) for info in dataset.data_infos),
            difficult=sum(len(info['ann']['labels_ignore']) for info in dataset.data_infos),
            excluded=len(dataset.invalid_annotations),
            excluded_annotations=dataset.invalid_annotations)
        print(name, report['splits'][name], flush=True)
        if name == 'train':
            train = dataset
            train.test_mode = False
        elif name == 'val':
            val = dataset
    assert not (splits['train'] & splits['val'])
    assert not (splits['train'] & splits['test'])
    assert not (splits['val'] & splits['test'])
    if args.device == 'cuda':
        assert torch.cuda.is_available()
        device = torch.cuda.current_device()
    else:
        device = -1  # MMCV scatter to CPU.
    model = build_detector(cfg.model)
    if args.device == 'cuda':
        model = model.cuda()
    model.init_weights()
    model.train()
    optimizer_cfg = cfg.optimizer.copy()
    optimizer_cfg.pop('type')
    optimizer = torch.optim.SGD(model.parameters(), **optimizer_cfg)
    batch_size = cfg.data.samples_per_gpu
    samples = [train[i] for i in range(batch_size)]
    batch = scatter(collate(samples, samples_per_gpu=batch_size), [device])[0]
    report['batch_size'] = batch_size
    report['input_shape'] = list(batch['img'].shape)
    optimizer.zero_grad()
    output = model.train_step(batch, optimizer)
    assert torch.isfinite(output['loss']), output['log_vars']
    output['loss'].backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 35)
    assert torch.isfinite(grad_norm)
    optimizer.step()
    report['losses'] = output['log_vars']
    report['grad_norm'] = float(grad_norm)
    print('Training step:', report['losses'], flush=True)
    del batch, output
    model.eval()
    with torch.no_grad():
        batch = scatter(collate([val[0]], samples_per_gpu=1), [device])[0]
        results = model(return_loss=False, rescale=True, **batch)
    assert len(results) == 1 and len(results[0]) == 20
    assert all(r.shape[1] == 6 and np.isfinite(r).all() for r in results[0])
    report['inference_detections'] = sum(len(r) for r in results[0])
    # Perfect detections should score 1.0; exercise rotated IoU + VOC07 AP.
    val.data_infos = val.data_infos[:2]
    perfect = []
    for info in val.data_infos:
        ann = info['ann']
        perfect.append([
            np.concatenate((ann['bboxes'][ann['labels'] == c],
                            np.ones((sum(ann['labels'] == c), 1), dtype=np.float32)),
                           axis=1) for c in range(20)])
    report['perfect_prediction_mAP'] = float(val.evaluate(perfect, nproc=1)['mAP'])
    assert abs(report['perfect_prediction_mAP'] - 1.0) < 1e-5
    if args.device == 'cuda':
        report['gpu'] = torch.cuda.get_device_name(device)
        report['peak_gpu_memory_mib'] = torch.cuda.max_memory_allocated() / 1024**2
        report['peak_gpu_reserved_mib'] = torch.cuda.max_memory_reserved() / 1024**2
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + '\n')
    print('PASS:', args.report, flush=True)


if __name__ == '__main__':
    main()
