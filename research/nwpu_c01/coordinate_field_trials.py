"""Continuous C01 auxiliary with image-field and native FCOS rotation checks."""
import torch
from mmcv.ops import box_iou_rotated
from mmdet.models.losses.utils import weight_reduce_loss
from mmrotate.models.builder import ROTATED_DETECTORS, ROTATED_HEADS, ROTATED_LOSSES
from mmrotate.models.dense_heads.bd_content_rotated_fcos_head import BDContentRotatedFCOSHead
from mmrotate.models.detectors.base import RotatedBaseDetector
from mmrotate.models.detectors.rotated_fcos import RotatedFCOS
from mmrotate.models.losses.bd_near_square_angle_content_loss import (
    BDNearSquareAngleContentLoss, angle_content_values,
)

from .c01_edge_control import edge_texture_field


def content_angle_derivative(box, target, field, stride, grid_size, min_size):
    """First derivative in decoded angle coordinates, with original C01 C4 alignment."""
    with torch.enable_grad():
        angle = box[:, 4:5].detach().requires_grad_(True)
        proxy = torch.cat((box[:, :4].detach(), angle), dim=1)
        value = angle_content_values(proxy, target.detach(), field.detach(), stride, grid_size, min_size)
        derivative, = torch.autograd.grad(value.sum(), angle)
    return value.detach(), derivative[:, 0].detach()


@torch.no_grad()
def native_iou_gain(box, target, prior, delta):
    """Changing FCOS angle at fixed distances rotates its center about its feature point."""
    offset = box[:, :2] - prior
    c, s = delta.cos(), delta.sin()
    dx = c * offset[:, 0] - s * offset[:, 1]
    dy = s * offset[:, 0] + c * offset[:, 1]
    moved = torch.cat((prior + torch.stack((dx, dy), dim=1), box[:, 2:4],
                       (box[:, 4] + delta)[:, None]), dim=1)
    target = target.contiguous()
    return (box_iou_rotated(moved.contiguous(), target, aligned=True)
            - box_iou_rotated(box.contiguous(), target, aligned=True))


@ROTATED_LOSSES.register_module()
class CoordinateFieldLoss(BDNearSquareAngleContentLoss):
    def __init__(self, field_mode='p3', probe_angle=0.05, **kwargs):
        super().__init__(**kwargs)
        if field_mode not in ('p3', 'edge', 'agreement') or not 0 < probe_angle < 0.5:
            raise ValueError((field_mode, probe_angle))
        self.field_mode = field_mode
        self.probe_angle = probe_angle
        self.positive_priors = None
        self.last_auxiliary = None
        self.last_positive_count = 0

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None,
                feature_maps=None, image_indices=None, feature_strides=None,
                pixel_pred=None, pixel_target=None, **kwargs):
        reduction = reduction_override or self.reduction
        if weight is not None and weight.ndim > 1:
            weight = weight.mean(-1)
        active = torch.ones_like(pred[:, 4], dtype=torch.bool) if weight is None else weight > 0
        geometry = pred.sum(-1) * 0
        if active.any():
            ids = active.nonzero().flatten()
            geom = self.geometry_values(pred[ids], target[ids].detach()).to(pred.dtype)
            geometry = geometry.index_copy(0, ids, geom)
        p = pred if pixel_pred is None else pixel_pred
        g = (target if pixel_target is None else pixel_target).detach()
        ratio = g[:, 2:4].amax(1) / g[:, 2:4].amin(1).clamp_min(self.min_size)
        eligible = active & (ratio <= self.square_ratio)
        content = p[:, 4] * 0
        kept = pred.new_zeros(())
        iou_safe = pred.new_zeros(())
        if self.semantic_weight and eligible.any():
            if self.positive_priors is None or self.positive_priors.shape != (len(p), 2):
                raise ValueError('Positive pixel priors must correspond to all regression rows')
            if not feature_maps or feature_strides is None or image_indices is None:
                raise ValueError('Image field, stride and positive row image indices required')
            stride = float(feature_strides[0])
            priors = self.positive_priors.detach().to(p)
            for image in range(len(feature_maps[0])):
                ids = (eligible & (image_indices == image)).nonzero().flatten()
                if not len(ids):
                    continue
                fields = []
                for field in feature_maps:
                    field = field[image:image+1].detach().to(pred.dtype)
                    fields.append(field / (field.square().sum(1, keepdim=True) + self.eps**2).sqrt())
                for select in ids.split(self.chunk_size):
                    box = p[select]
                    value, grad = content_angle_derivative(box, g[select], fields[0], stride,
                                                           self.grid_size, self.min_size)
                    gain = native_iou_gain(box.detach(), g[select], priors[select],
                                           -self.probe_angle * grad.sign())
                    accept = gain > 1e-6
                    iou_safe += accept.sum()
                    if self.field_mode == 'agreement':
                        _, edge_grad = content_angle_derivative(box, g[select], fields[1], stride,
                                                                self.grid_size, self.min_size)
                        accept = accept & (grad * edge_grad > 0)
                    kept += accept.sum()
                    # Preserve the exact C01 value and first derivative for accepted rows.
                    linearized = value + grad * (box[:, 4] - box[:, 4].detach())
                    content = content.index_copy(0, select, linearized * accept)
        auxiliary = self.semantic_weight * content
        self.last_positive_count = len(p)
        self.last_auxiliary = self.loss_weight * weight_reduce_loss(auxiliary, weight, reduction, avg_factor)
        if reduction != 'none':
            self.last_diagnostics = dict(content_term=float(self.last_auxiliary.detach()),
                                         eligible=int(eligible.sum()), iou_safe=int(iou_safe),
                                         accepted=int(kept))
        return self.loss_weight * weight_reduce_loss(geometry + auxiliary, weight, reduction, avg_factor)


@ROTATED_HEADS.register_module()
class CoordinateFieldHead(BDContentRotatedFCOSHead):
    def get_targets(self, points, gt_bboxes_list, gt_labels_list):
        targets = super().get_targets(points, gt_bboxes_list, gt_labels_list)
        batch_size = len(gt_bboxes_list)
        self.loss_bbox.positive_priors = torch.cat([
            grid.repeat(batch_size, 1)[(label >= 0) & (label < self.num_classes)]
            for grid, label in zip(points, targets[0])])
        return targets

    def loss(self, *args, **kwargs):
        self.loss_bbox.last_auxiliary = None
        self.loss_bbox.last_positive_count = 0
        try:
            return super().loss(*args, **kwargs)
        finally:
            self.loss_bbox.positive_priors = None


@ROTATED_DETECTORS.register_module()
class CoordinateFieldFCOS(RotatedFCOS):
    def forward_train(self, img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None):
        RotatedBaseDetector.forward_train(self, img, img_metas)
        features = self.extract_feat(img)
        mode = self.bbox_head.loss_bbox.field_mode
        fields = () if mode == 'edge' else (features[0].detach(),)
        if mode in ('edge', 'agreement'):
            edge = edge_texture_field(img, self.bbox_head.strides[0])
            if edge.shape[-2:] != features[0].shape[-2:]:
                raise ValueError('Fixed edge field must have the same resolution as P3')
            fields += (edge,)
        return self.bbox_head.loss(*self.bbox_head(features), gt_bboxes, gt_labels, img_metas,
                                   gt_bboxes_ignore=gt_bboxes_ignore, semantic_features=fields)
