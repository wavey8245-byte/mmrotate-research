"""FCOS output/assignment with RiO-DETRv2's available loss branches.

No FDR, query matching, decoder auxiliaries, or teacher distillation is implied.
The base head and its other experiments are unchanged.
"""
import math

import torch
from mmcv.runner import force_fp32
from mmdet.core import reduce_mean

from ..builder import ROTATED_HEADS
from ..losses.rio_v2_varifocal_loss import RiOV2VarifocalLoss
from ..losses.rotated_rio_loss import RotatedRiOLoss, stable_box_kld
from .rotated_fcos_head import RotatedFCOSHead


@ROTATED_HEADS.register_module()
class RiOV2FCOSHead(RotatedFCOSHead):
    """Retain FCOS predictions, replace classification and regression weighting."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.separate_angle or not isinstance(self.loss_bbox, RotatedRiOLoss):
            raise ValueError('RiO v2 FCOS requires joint decoded RotatedRiOLoss')
        if not isinstance(self.loss_cls, RiOV2VarifocalLoss):
            raise ValueError('RiO v2 FCOS requires RiOV2VarifocalLoss')
        if (not self.loss_bbox.kld_normalized
                or not math.isclose(self.loss_bbox.angle_normalizer, math.pi)):
            raise ValueError('Use image-normalized KLD and unit-period angular L1')
        # Preserve the four-output FCOS interface, but do not train an unused
        # centerness branch or use its random scores in inference.
        self.conv_centerness.requires_grad_(False)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'angle_preds', 'centernesses'))
    def loss(self, cls_scores, bbox_preds, angle_preds, centernesses,
             gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None):
        if gt_bboxes_ignore is not None and any(len(x) for x in gt_bboxes_ignore):
            raise ValueError('Ignored GT boxes are not supported by FCOS assignment')
        points = self.prior_generator.grid_priors(
            [x.shape[-2:] for x in cls_scores], dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)
        labels, distances, angles = self.get_targets(points, gt_bboxes, gt_labels)
        batch = cls_scores[0].shape[0]

        def flatten(tensors, channels):
            return torch.cat([x.permute(0, 2, 3, 1).reshape(-1, channels)
                              for x in tensors])

        logits = flatten(cls_scores, self.cls_out_channels)
        encoded = torch.cat((flatten(bbox_preds, 4), flatten(angle_preds, 1)), -1)
        target_encoded = torch.cat((torch.cat(distances), torch.cat(angles)), -1)
        labels = torch.cat(labels)
        positive = ((labels >= 0) & (labels < self.num_classes)).nonzero().flatten()
        count = reduce_mean(logits.new_tensor(float(positive.numel()))).clamp_min(1)
        origins = torch.cat([x.repeat(batch, 1) for x in points])
        quality = logits.new_zeros(labels.shape)
        if positive.numel():
            prediction = self.bbox_coder.decode(origins[positive], encoded[positive])
            target = self.bbox_coder.decode(origins[positive], target_encoded[positive])
            scales = []
            for level_points, stride in zip(points, self.strides):
                factor = stride if self.norm_on_bbox else 1
                per_image = prediction.new_tensor([
                    [factor / meta['img_shape'][1], factor / meta['img_shape'][0]] * 2
                    for meta in img_metas])
                scales.append(per_image.repeat_interleave(len(level_points), 0))
            scale = torch.cat(scales)[positive]
            components = self.loss_bbox(
                prediction, target, spatial_scale=scale, avg_factor=count,
                return_components=True)
            # Same translation-invariant, image-normalized representation as
            # regression, including covariance regularization in these units.
            with torch.no_grad():
                p = torch.cat((torch.zeros_like(prediction[:, :2]),
                               prediction[:, 2:4] * scale[:, 2:4],
                               prediction[:, 4:]), -1)
                t = torch.cat(((target[:, :2] - prediction[:, :2]) * scale[:, :2],
                               target[:, 2:4] * scale[:, 2:4], target[:, 4:]), -1)
                quality[positive] = (1 - stable_box_kld(
                    p, t, self.loss_bbox.min_size,
                    self.loss_bbox.covariance_epsilon)).clamp(0, 1)
        else:
            zero = encoded.sum() * 0
            components = {key: zero for key in ('bbox_kld', 'bbox_l1', 'bbox_sp_l1')}
        return dict(
            loss_cls=self.loss_cls(logits, labels, quality, avg_factor=count),
            loss_bbox=sum(components.values()),
            quality_mean=quality[positive].mean() if positive.numel() else quality.sum(),
            **{name: value.detach() for name, value in components.items()})

    def _get_bboxes_single(self, cls_scores, bbox_preds, angle_preds, centernesses,
                           mlvl_points, img_shape, scale_factor, cfg, rescale=False):
        # The inherited path applies sigmoid to these score factors. +inf
        # gives exactly one for both pre-NMS ranking and final NMS scores.
        neutral = [torch.full_like(x, float('inf')) for x in centernesses]
        return super()._get_bboxes_single(
            cls_scores, bbox_preds, angle_preds, neutral, mlvl_points,
            img_shape, scale_factor, cfg, rescale)
