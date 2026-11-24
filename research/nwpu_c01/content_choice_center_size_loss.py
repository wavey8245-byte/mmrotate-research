"""Retain CONTENT_CHOICE and add IoU-weighted content gradients to distances."""
import torch
from mmcv.ops import box_iou_rotated
from mmdet.models.losses.utils import weight_reduce_loss
from mmrotate.models.builder import ROTATED_HEADS, ROTATED_LOSSES
from mmrotate.models.losses.bd_near_square_full_content_loss import full_content_values

from .c01_refinement import C01RefinementFCOSHead, C01RefinementLoss


@ROTATED_LOSSES.register_module()
class ContentChoiceCenterSizeLoss(C01RefinementLoss):
    def __init__(self, center_size_weight, **kwargs):
        super().__init__(**kwargs)
        self.center_size_weight = center_size_weight
        self.center_size_boxes = None
        self.last_center_size = None

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None,
                feature_maps=None, image_indices=None, feature_strides=None,
                pixel_pred=None, pixel_target=None, **kwargs):
        original = super().forward(pred, target, weight, avg_factor, reduction_override,
                                   feature_maps, image_indices, feature_strides,
                                   pixel_pred, pixel_target, **kwargs)
        if not self.center_size_weight:
            return original
        if self.center_size_boxes is None:
            raise ValueError('Missing pixel boxes decoded with detached angles')
        p = self.center_size_boxes
        g = pixel_target.detach()
        if weight is not None and weight.ndim > 1:
            weight = weight.mean(-1)
        ratio = g[:, 2:4].amax(1) / g[:, 2:4].amin(1).clamp_min(self.min_size)
        eligible = ratio <= self.square_ratio
        if weight is not None:
            eligible = eligible & (weight > 0)
        raw = p[:, 0] * 0
        for image in range(len(feature_maps[0])):
            indices = (eligible & (image_indices == image)).nonzero().flatten()
            if not len(indices):
                continue
            field = feature_maps[0][image:image+1].detach().to(pred.dtype)
            field = field / torch.sqrt(field.square().sum(1, keepdim=True) + self.eps**2)
            for selected in indices.split(self.chunk_size):
                region = p[selected]
                targets = g[selected]
                with torch.no_grad():
                    boxes = region.detach().clone()
                    boxes[:, 2:4].clamp_min_(self.min_size)
                    quality = box_iou_rotated(boxes.contiguous(), targets.contiguous(),
                                              aligned=True).clamp(0, 1)
                values = full_content_values(region, targets, field,
                                             float(feature_strides[0]), self.grid_size, self.min_size)
                raw = raw.index_copy(0, selected, quality * values)
        reduction = reduction_override or self.reduction
        self.last_center_size = self.loss_weight * weight_reduce_loss(
            self.center_size_weight * raw, weight, reduction, avg_factor)
        if reduction != 'none':
            self.last_diagnostics['center_size_term'] = float(self.last_center_size.detach())
        return original + self.last_center_size


@ROTATED_HEADS.register_module()
class ContentChoiceCenterSizeFCOSHead(C01RefinementFCOSHead):
    # Reuse the original assignment and detection losses. During assignment, prepare
    # a separate pixel decode whose center and size depend on distances alone.
    def loss(self, cls_scores, bbox_preds, angle_preds, *args, **kwargs):
        self._center_size_outputs = (bbox_preds, angle_preds) if self.loss_bbox.center_size_weight else None
        self.loss_bbox.last_center_size = None
        try:
            return super().loss(cls_scores, bbox_preds, angle_preds, *args, **kwargs)
        finally:
            self._center_size_outputs = None
            self.loss_bbox.center_size_boxes = None

    def get_targets(self, points, gt_bboxes_list, gt_labels_list):
        targets = super().get_targets(points, gt_bboxes_list, gt_labels_list)
        outputs = getattr(self, '_center_size_outputs', None)
        if outputs is None:
            return targets
        distances, angles = outputs
        boxes = []
        for labels, prior, d, a, stride in zip(targets[0], points, distances, angles, self.strides):
            positive = (labels >= 0) & (labels < self.num_classes)
            d = d.permute(0, 2, 3, 1).reshape(-1, 4)[positive].to(prior.dtype) * stride
            a = a.detach().permute(0, 2, 3, 1).reshape(-1, 1)[positive].to(prior.dtype)
            centers = prior.repeat(len(gt_bboxes_list), 1)[positive]
            boxes.append(self.bbox_coder.decode(centers, torch.cat((d, a), dim=1)))
        self.loss_bbox.center_size_boxes = torch.cat(boxes)
        return targets
