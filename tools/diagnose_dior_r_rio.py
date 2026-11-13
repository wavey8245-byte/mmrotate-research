"""Profile the failed RiO checkpoint on training images, including per-term gradients."""
import argparse
import copy
import json
import math
from pathlib import Path
import random

import mmcv
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmcv.ops import box_iou_rotated
import numpy as np
import torch

from mmrotate.datasets import build_dataset
from mmrotate.models import build_detector
from mmrotate.models.losses.rotated_iou_loss import RotatedIoULoss


def statistics(tensor):
    array = tensor.detach().float().cpu().numpy().reshape(-1)
    if not array.size:
        return dict(count=0)
    return dict(min=float(array.min()), median=float(np.median(array)),
                p95=float(np.percentile(array, 95)), max=float(array.max()),
                nonpositive_fraction=float((array <= 0).mean()))


def geometry_diagnosis(pred, target, weight):
    """Aligned positive-box diagnostics, not detector AP or a retrained model."""
    # Use the CPU overlap operator, as used by the dataset's AP evaluator.
    pred, target, weight = pred.detach().cpu(), target.detach().cpu(), weight.detach().cpu()
    swap = pred[:, 2] < pred[:, 3]
    canonical = pred.clone()
    canonical[:, 2:4] = torch.where(swap[:, None], pred[:, [3, 2]], pred[:, 2:4])
    canonical[:, 4] += swap.float() * math.pi / 2
    variants = dict(actual=pred, canonical_equivalent=canonical)
    for name, columns in (('oracle_center', slice(0, 2)),
                          ('oracle_size', slice(2, 4)),
                          ('oracle_angle', slice(4, 5))):
        variant = canonical.clone()
        variant[:, columns] = target[:, columns]
        variants[name] = variant
    variants['oracle_all'] = target
    result = dict(positives=len(pred), swapped_fraction=float(swap.float().mean()),
                  zero_side_fraction=float((pred[:, 2:4] <= 0).any(-1).float().mean()),
                  angular_error_degrees=statistics((
                      (canonical[:, 4] - target[:, 4] + math.pi / 2)
                      % math.pi - math.pi / 2).abs() * 180 / math.pi),
                  width_ratio=statistics(canonical[:, 2] / target[:, 2]),
                  height_ratio=statistics(canonical[:, 3] / target[:, 3]),
                  variants={})
    for name, variant in variants.items():
        # The head decodes tiny stride-unit boxes at image-coordinate points.
        # Recenter each pair to avoid polygon-area cancellation at large origins.
        local_pred, local_target = variant.clone(), target.clone()
        local_pred[:, :2] -= target[:, :2]
        local_target[:, :2] = 0
        iou = box_iou_rotated(local_pred.contiguous(), local_target.contiguous(),
                              aligned=True).reshape(-1)
        result['variants'][name] = dict(mean=float(iou.mean()), median=float(iou.median()),
            weighted_mean=float((iou * weight).sum() / weight.sum().clamp_min(1e-6)),
            iou_ge_half_fraction=float((iou >= .5).float().mean()),
            no_overlap_fraction=float((iou == 0).float().mean()))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', default='work_dirs/dior_r_trainval/fcos_rio_bs2_800_20261015')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--batches', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--train-eval-images', type=int, default=0)
    parser.add_argument('--eval-seed', type=int)
    parser.add_argument('--report')
    parser.add_argument('--checkpoint', default='epoch_1.pth')
    parser.add_argument('--sample-seed', type=int)
    parser.add_argument('--geometry', action='store_true')
    parser.add_argument('--geometry-only', action='store_true')
    parser.add_argument('--centerness-conflict', action='store_true')
    args = parser.parse_args()
    folder = Path(args.work_dir)
    torch.set_num_threads(2)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    cfg = mmcv.Config.fromfile(str(folder / 'snapshot/experiment.py'))
    if args.sample_seed is not None:
        for transform in cfg.data.train.pipeline:
            if transform.type == 'RRandomFlip':
                transform.flip_ratio = [0., 0., 0.]
    dataset = build_dataset(cfg.data.train)
    sample_ids = (np.random.RandomState(args.sample_seed).choice(
        len(dataset), args.batches * args.batch_size, replace=False)
        if args.sample_seed is not None else np.arange(args.batches * args.batch_size))
    model = build_detector(cfg.model).to(args.device)
    checkpoint = load_checkpoint(model, str(folder / args.checkpoint), map_location='cpu')
    model.train()
    original = model.bbox_head.loss_bbox.forward
    captured = {}

    def capture(pred, target, *args, **kwargs):
        value = original(pred, target, *args, **kwargs)
        captured.update(pred=pred, target=target, terms=value,
                        weight=kwargs.get('weight'), avg_factor=kwargs.get('avg_factor'))
        return value

    model.bbox_head.loss_bbox.forward = capture
    parameters = [(name, parameter) for name, parameter in model.named_parameters()
                  if parameter.requires_grad]
    if args.device == 'cpu':
        parameters = [(name, parameter) for name, parameter in parameters
                      if name.startswith('bbox_head.')]
    report = dict(checkpoint=str(folder / args.checkpoint),
                  checkpoint_epoch=checkpoint.get('meta', {}).get('epoch'),
                  checkpoint_iteration=checkpoint.get('meta', {}).get('iter'), batches=[],
                  sample_ids=sample_ids.tolist(),
                  geometry_only=args.geometry_only,
                  gradient_scope='head only' if args.device == 'cpu' else 'all trainable parameters')
    raw_logits = []
    handle = model.bbox_head.conv_angle.register_forward_hook(
        lambda module, inputs, output: raw_logits.append(output.detach().reshape(-1)))
    raw_distances = []
    distance_handle = model.bbox_head.conv_reg.register_forward_hook(
        lambda module, inputs, output: raw_distances.append(output.detach().reshape(-1)))
    predictions = []
    for step in range(args.batches):
        raw_logits.clear()
        raw_distances.clear()
        batch = scatter(collate([dataset[int(sample_ids[step * args.batch_size + i])]
                                for i in range(args.batch_size)],
                                samples_per_gpu=args.batch_size),
                        [0 if args.device == 'cuda' else -1])[0]
        with torch.set_grad_enabled(not args.geometry_only):
            if args.device == 'cpu':
                with torch.no_grad():
                    features = model.extract_feat(batch['img'])
            else:
                features = model.extract_feat(batch['img'])
            output = model.bbox_head(features)
            losses = model.bbox_head.loss(*output, batch['gt_bboxes'],
                                          batch['gt_labels'], batch['img_metas'])
        angles = torch.cat([value.reshape(-1) for value in output[2]])
        distances = torch.cat([value.reshape(-1) for value in output[1]])
        record = dict(batch=step, raw_scaled_angles=statistics(angles),
                      absolute_scaled_angles=statistics(angles.abs()),
                      raw_angle_logits=statistics(torch.cat(raw_logits)),
                      raw_distance_logits=statistics(torch.cat(raw_distances)),
                      distances_after_clamp=statistics(distances),
                      positive_box_widths=statistics(captured['pred'][:, 2]),
                      positive_box_heights=statistics(captured['pred'][:, 3]),
                      component_gradients={})
        predictions.append(torch.cat([value.detach().reshape(-1)
                                      for values in output[1:3] for value in values]).cpu())
        if model.bbox_head.bounded_angle:
            p = captured['pred'][:, 4] / math.pi + .5
            record['positive_angle_boundary_fraction'] = float(
                (captured['pred'][:, 4].abs() > math.pi / 2 - 1e-3).float().mean())
            record['positive_sigmoid_jacobian'] = statistics(math.pi * p * (1 - p))
        gradient_terms = dict(captured['terms'])
        if args.geometry or args.geometry_only:
            record['geometry'] = geometry_diagnosis(
                captured['pred'], captured['target'], captured['weight'])
            gradient_terms['reference_riou_same_boxes'] = RotatedIoULoss()(
                captured['pred'], captured['target'], weight=captured['weight'],
                avg_factor=captured['avg_factor'])
        gradient_terms.update({name: losses[name] for name in
                               ('loss_cls', 'loss_bbox', 'loss_centerness')})
        gradient_terms['loss_total'] = sum(losses[name] for name in
                                         ('loss_cls', 'loss_bbox', 'loss_centerness'))
        if args.geometry_only:
            record['loss_values'] = {name: float(value.detach())
                                    for name, value in gradient_terms.items()}
            gradient_terms = {}
        angle_gradients = {}
        shared_reg_gradients = {}
        for name, value in gradient_terms.items():
            gradients = torch.autograd.grad(value, [p for _, p in parameters],
                                            retain_graph=True, allow_unused=True)
            norms = {param_name: float(gradient.double().norm())
                     for (param_name, _), gradient in zip(parameters, gradients)
                     if gradient is not None}
            angle_gradients[name] = torch.cat([
                gradient.detach().reshape(-1)
                for (param_name, _), gradient in zip(parameters, gradients)
                if gradient is not None and ('conv_angle.' in param_name
                                            or 'scale_angle.' in param_name)]) if any(
                    ('conv_angle.' in key or 'scale_angle.' in key) for key in norms
                ) else None
            record['component_gradients'][name] = dict(
                value=float(value.detach()),
                global_norm=sum(x * x for x in norms.values()) ** .5,
                angle_output_norm=sum(x * x for key, x in norms.items()
                    if 'conv_angle.' in key or 'scale_angle.' in key) ** .5,
                distance_output_norm=sum(x * x for key, x in norms.items()
                    if 'conv_reg.' in key or 'bbox_head.scales.' in key) ** .5,
                top_parameters=sorted(norms.items(), key=lambda x: -x[1])[:6])
            if args.centerness_conflict:
                shared_reg_gradients[name] = torch.cat([
                    (gradient.detach() if gradient is not None else torch.zeros_like(parameter)).reshape(-1)
                    for (param_name, parameter), gradient in zip(parameters, gradients)
                    if param_name.startswith('bbox_head.reg_convs.')])
                record['component_gradients'][name]['shared_reg_tower_norm'] = float(
                    shared_reg_gradients[name].double().norm())
        record['angle_gradient_cosines'] = {}
        for left, right in (('bbox_kld', 'bbox_sp_l1'),
                            ('loss_bbox', 'reference_riou_same_boxes')):
            a, b = angle_gradients.get(left), angle_gradients.get(right)
            if a is not None and b is not None:
                record['angle_gradient_cosines'][left + '__' + right] = float(
                    torch.nn.functional.cosine_similarity(a, b, dim=0))
        if args.centerness_conflict and not args.geometry_only:
            reference = shared_reg_gradients['loss_centerness']
            record['shared_reg_tower_cosines_with_centerness'] = {
                name: float(torch.nn.functional.cosine_similarity(value, reference, dim=0))
                for name, value in shared_reg_gradients.items()
                if name in ('bbox_kld', 'bbox_l1', 'bbox_sp_l1',
                            'reference_riou_same_boxes', 'loss_bbox')}
            reg_to_ctr = torch.autograd.grad(losses['loss_bbox'], output[3],
                                             retain_graph=True, allow_unused=True)
            ctr_to_reg = torch.autograd.grad(losses['loss_centerness'],
                list(output[1])+list(output[2]), retain_graph=True, allow_unused=True)
            record['direct_output_dependencies'] = dict(
                regression_depends_on_predicted_centerness=any(g is not None for g in reg_to_ctr),
                centerness_loss_depends_on_predicted_distances_or_angles=any(
                    g is not None for g in ctr_to_reg))
        report['batches'].append(record)
        print(json.dumps(record), flush=True)
        del batch, features, output, losses
        captured.clear()
    handle.remove()
    distance_handle.remove()
    if model.bbox_head.is_scale_angle:
        report['angle_scale'] = float(model.bbox_head.scale_angle.scale.detach())
    if len(predictions) > 1:
        report['regression_image_response'] = float(
            (predictions[1] - predictions[0]).abs().mean()
            / (predictions[0].abs().mean() + 1e-6))
    if args.train_eval_images:
        infer_cfg = copy.deepcopy(cfg.data.test)
        infer_cfg.ann_file = 'data/DIOR-R/ImageSets/Main/train.txt'
        infer_cfg.img_prefix = 'data/DIOR-R/JPEGImages-trainval/'
        infer_cfg.test_mode = True
        inference = build_dataset(infer_cfg)
        indices = (np.random.RandomState(args.eval_seed).permutation(len(inference))[
            :args.train_eval_images] if args.eval_seed is not None
            else np.arange(min(args.train_eval_images, len(inference))))
        inference.data_infos = [inference.data_infos[int(i)] for i in indices]
        report['train_evaluation_sample_ids'] = indices.tolist()
        report['train_evaluation_seed'] = args.eval_seed
        model.eval()
        results = []
        with torch.no_grad():
            for i in range(len(inference)):
                batch = scatter(collate([inference[i]], samples_per_gpu=1),
                                [0 if args.device == 'cuda' else -1])[0]
                results.extend(model(return_loss=False, rescale=True, **batch))
        report.update(checked_train_images=len(inference), train_image_mAP=float(
            inference.evaluate(results, nproc=2, use_07_metric=True)['mAP']))
    path = Path(args.report) if args.report else folder / 'failure_diagnosis.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(str(path), flush=True)


if __name__ == '__main__':
    main()
