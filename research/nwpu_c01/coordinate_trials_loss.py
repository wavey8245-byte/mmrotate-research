"""Choose finite angle and log-size corrections on training positives."""
import math

import torch
from mmcv.ops import box_iou_rotated
from mmdet.models.losses.utils import weight_reduce_loss
from mmrotate.models.builder import ROTATED_HEADS, ROTATED_LOSSES
from mmrotate.models.dense_heads.bd_content_rotated_fcos_head import BDContentRotatedFCOSHead
from mmrotate.models.losses.bd_near_square_angle_content_loss import BDNearSquareAngleContentLoss

from .c01_refinement import choose_candidate, local_candidates


def aligned_overlaps(boxes, target):
    reference = target[:, None, :].expand_as(boxes)
    return box_iou_rotated(boxes.reshape(-1, 5).contiguous(),
                           reference.reshape(-1, 5).contiguous(), aligned=True).reshape(boxes.shape[:2])


def rotated_centers(center, prior, steps):
    # An FCOS distance+angle box is centered at prior + R(angle)*local_offset.
    # Keeping the four raw distances fixed rotates the offset about that prior.
    dx, dy = (center - prior).unbind(1)
    cos, sin = steps.cos(), steps.sin()
    return torch.stack((prior[:, 0:1] + dx[:, None] * cos - dy[:, None] * sin,
                        prior[:, 1:2] + dx[:, None] * sin + dy[:, None] * cos), -1)


@torch.no_grad()
def angle_trial(pred, target, prior, field, stride, choice, guard):
    boxes, overlaps, steps = local_candidates(pred, target)
    if guard:
        native = boxes.clone()
        native[:, :, :2] = rotated_centers(pred[:, :2], prior, steps)
        native[:, 0, :2] = pred[:, :2]
        native_iou = aligned_overlaps(native, target)
        valid = (overlaps >= overlaps[:, :1] - 1e-6) & (native_iou >= native_iou[:, :1] - 1e-6)
        # Keep the archived feature costs on the fixed-center trials. For the
        # geometry control rank the same feasible trials by their native IoU.
        scores = native_iou if choice == 'iou' else overlaps
        overlaps = scores.masked_fill(~valid, -torch.inf)
    mode = 'IOU_CHOICE' if choice == 'iou' else 'CONTENT_CHOICE'
    return steps[choose_candidate(boxes, overlaps, target, field, stride, mode)]


@torch.no_grad()
def size_trial(pred, target, field, stride, choice, step):
    log_steps = pred.new_tensor(((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1),
                                 (-1, -1), (1, 1))) * step
    boxes = pred[:, None, :].expand(-1, len(log_steps), -1).clone()
    boxes[:, :, 2:4] *= log_steps.exp()
    overlaps = aligned_overlaps(boxes, target)
    mode = 'IOU_CHOICE' if choice == 'iou' else 'CONTENT_CHOICE'
    return log_steps[choose_candidate(boxes, overlaps, target, field, stride, mode)]


@ROTATED_LOSSES.register_module()
class CoordinateTrialsLoss(BDNearSquareAngleContentLoss):
    def __init__(self, angle_guard=True, angle_choice='content', size_choice='content',
                 size_weight=0., size_step=.05, **kwargs):
        super().__init__(**kwargs)
        if angle_choice not in ('content', 'iou') or size_choice not in ('content', 'iou'):
            raise ValueError('Choices must be content or iou')
        if not math.isfinite(size_weight) or size_weight < 0 or not 0 < size_step <= 1:
            raise ValueError('Invalid size trial weight or step')
        if self.grid_size != 5:
            raise ValueError('Candidate costs use the archived 5x5 sampler')
        self.angle_guard = angle_guard
        self.angle_choice = angle_choice
        self.size_choice = size_choice
        self.size_step = size_step
        self.size_weight = size_weight
        self.positive_priors = None
        self.last_angle_auxiliary = None
        self.last_size_auxiliary = None
        self.last_positive_count = 0

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None,
                feature_maps=None, image_indices=None, feature_strides=None,
                pixel_pred=None, pixel_target=None, **kwargs):
        reduction = reduction_override or self.reduction
        if weight is not None and weight.ndim > 1:
            weight = weight.mean(-1)
        active = torch.ones(len(pred), device=pred.device, dtype=torch.bool) if weight is None else weight > 0
        geometry = pred.sum(-1) * 0
        ids = active.nonzero().flatten()
        if len(ids):
            geometry = geometry.index_copy(0, ids, self.geometry_values(
                pred[ids], target[ids].detach()).to(pred.dtype))
        p = pred if pixel_pred is None else pixel_pred
        g = (target if pixel_target is None else pixel_target).detach()
        wh = p[:, 2:4].clamp_min(self.min_size)
        angle_values = p[:, 4] * 0
        size_values = wh.sum(1) * 0
        aspect_ratio = g[:, 2:4].amax(1) / g[:, 2:4].amin(1).clamp_min(self.min_size)
        eligible = active & (aspect_ratio <= self.square_ratio)
        if self.angle_guard and self.semantic_weight and self.positive_priors is None:
            raise ValueError('Native angle checks require assigned FCOS priors')
        self.last_positive_count = len(pred)
        if eligible.any() and (self.semantic_weight or self.size_weight):
            stride = float(feature_strides[0])
            for image in range(len(feature_maps[0])):
                selected = (eligible & (image_indices == image)).nonzero().flatten()
                if not len(selected):
                    continue
                field = feature_maps[0][image:image+1].detach().to(pred.dtype)
                field = field / torch.sqrt(field.square().sum(1, keepdim=True) + self.eps**2)
                for chunk in selected.split(self.chunk_size):
                    base = p[chunk].detach().clone()
                    base[:, 2:4] = wh[chunk].detach()
                    gt = g[chunk]
                    if self.semantic_weight:
                        prior = self.positive_priors[chunk] if self.angle_guard else None
                        step = angle_trial(base, gt, prior, field, stride, self.angle_choice, self.angle_guard)
                        target_angle = base[:, 4] + step
                        values = (p[chunk, 4] - target_angle).sin().square()
                        angle_values = angle_values.index_copy(0, chunk, values)
                    if self.size_weight:
                        step = size_trial(base, gt, field, stride, self.size_choice, self.size_step)
                        log_size = wh[chunk].log()
                        target_log_size = log_size.detach() + step
                        values = .5 * (log_size - target_log_size).square().sum(1)
                        size_values = size_values.index_copy(0, chunk, values)

        def reduce(values):
            return self.loss_weight * weight_reduce_loss(values, weight, reduction, avg_factor)

        self.last_angle_auxiliary = reduce(self.semantic_weight * angle_values)
        self.last_size_auxiliary = reduce(self.size_weight * size_values)
        if reduction != 'none':
            self.last_diagnostics = dict(content_term=float(self.last_angle_auxiliary.detach()),
                                         size_term=float(self.last_size_auxiliary.detach()), eligible=int(eligible.sum()))
        return reduce(geometry + self.semantic_weight * angle_values + self.size_weight * size_values)


@ROTATED_HEADS.register_module()
class CoordinateTrialsFCOSHead(BDContentRotatedFCOSHead):
    def loss(self, *args, **kwargs):
        self.loss_bbox.last_positive_count = 0
        self.loss_bbox.last_angle_auxiliary = None
        self.loss_bbox.last_size_auxiliary = None
        try:
            return super().loss(*args, **kwargs)
        finally:
            self.loss_bbox.positive_priors = None

    def get_targets(self, points, gt_bboxes_list, gt_labels_list):
        targets = super().get_targets(points, gt_bboxes_list, gt_labels_list)
        if self.loss_bbox.angle_guard and self.loss_bbox.semantic_weight:
            priors = []
            for prior, labels in zip(points, targets[0]):
                positive = (labels >= 0) & (labels < self.num_classes)
                priors.append(prior.repeat(len(gt_bboxes_list), 1)[positive])
            self.loss_bbox.positive_priors = torch.cat(priors)
        return targets
