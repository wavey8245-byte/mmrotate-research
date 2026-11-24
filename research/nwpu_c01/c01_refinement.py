"""C01 variants on decoded boxes. The FCOS adapter only supplies image extents."""
import torch
import torch.nn.functional as F
from mmcv.ops import box_iou_rotated
from mmdet.models.losses.utils import weight_reduce_loss
from mmrotate.models.builder import ROTATED_HEADS, ROTATED_LOSSES
from mmrotate.models.dense_heads.bd_content_rotated_fcos_head import BDContentRotatedFCOSHead
from mmrotate.models.losses.bd_near_square_angle_content_loss import (
    BDNearSquareAngleContentLoss, _sample, angle_content_values, geometry_branch,
)

MODES = ('C01', 'POINTS', 'DYNAMIC', 'CONTENT_CHOICE', 'IOU_CHOICE')
ANGLE_STEPS = (0., -.05, .05, -.10, .10)


@torch.no_grad()
def point_weights(field, target, sampled_gt, stride, valid_hw):
    """Fixed mass per normalized radius, and a uniform fallback near borders."""
    axis = torch.arange(-2, 3, device=target.device, dtype=target.dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    ring = torch.maximum(xx.abs(), yy.abs()) == 2
    # The GT's expanded outer grid ring sits 0.6 side lengths from the center.
    expanded = target.clone()
    expanded[:, 2:4] *= 1.5
    outside = _sample(field, expanded, stride, 5)[:, :, ring]
    x = .3 * target[:, 2:3] * xx[ring]
    y = .3 * target[:, 3:4] * yy[ring]
    c, s = target[:, 4:5].cos(), target[:, 4:5].sin()
    gx, gy = target[:, :1] + c*x - s*y, target[:, 1:2] + s*x + c*y
    h, w = valid_hw
    valid = ((gx >= stride/2) & (gx <= w-stride/2) &
             (gy >= stride/2) & (gy <= h-stride/2))
    interior = F.normalize(sampled_gt[:, :, 1:4, 1:4].mean((2, 3)), dim=1, eps=1e-6)
    exterior = F.normalize((outside * valid[:, None]).sum(-1), dim=1, eps=1e-6)
    unit_samples = F.normalize(sampled_gt, dim=1, eps=1e-6)
    logits = (unit_samples * (interior - exterior)[:, :, None, None]).sum(1) / .25
    radii = xx.square() + yy.square()
    weights = target.new_ones((len(target), 5, 5))
    for radius, count in ((0, 1), (1, 4), (2, 4), (4, 4), (5, 8), (8, 4)):
        stratum = radii == radius
        probability = logits[:, stratum].softmax(-1)
        weights[:, stratum] = .5 + .5 * count * probability
    return torch.where((valid.sum(1) >= 4)[:, None, None], weights, 1.)


def content_distances(pred, gt, field, stride, sampled_gt, weights=None):
    branch = geometry_branch(pred.detach(), gt)
    samples = _sample(field, pred, stride, 5)
    variants = torch.stack([torch.rot90(samples, k, (-2, -1)) for k in range(4)], 1)
    aligned = variants[torch.arange(len(pred), device=pred.device), branch]
    residual = (aligned - sampled_gt).square().sum(1)
    return .25 * (residual if weights is None else residual * weights).mean((-2, -1))


@torch.no_grad()
def local_candidates(pred, gt):
    steps = pred.new_tensor(ANGLE_STEPS)
    boxes = pred[:, None, :].expand(-1, len(steps), -1).clone()
    boxes[:, :, 4] += steps
    references = gt[:, None, :].expand_as(boxes).reshape(-1, 5).contiguous()
    overlaps = box_iou_rotated(boxes.reshape(-1, 5).contiguous(), references, aligned=True)
    return boxes, overlaps.reshape(-1, len(steps)), steps


@torch.no_grad()
def choose_candidate(boxes, overlaps, gt, field, stride, mode):
    if mode == 'IOU_CHOICE':
        # Preserve the zero candidate for improvements within numerical tolerance.
        best = overlaps.argmax(1)
        best_iou = overlaps.gather(1, best[:, None]).squeeze(1)
        return torch.where(best_iou > overlaps[:, 0] + 1e-6, best, 0)
    admissible = overlaps >= overlaps[:, :1] - 1e-6
    sampled_gt = _sample(field, gt, stride, 5)
    cost = torch.stack([content_distances(boxes[:, k], gt, field, stride, sampled_gt)
                        for k in range(boxes.shape[1])], 1)
    cost = cost.masked_fill(~admissible, torch.inf)
    best = cost.argmin(1)
    best_cost = cost.gather(1, best[:, None]).squeeze(1)
    return torch.where(best_cost < cost[:, 0] - 1e-7, best, 0)


def refined_values(pred, gt, field, stride, valid_hw, mode, min_size=1e-4):
    # Match the direct gradient scope of original C01.
    p = torch.cat((pred[:, :2].detach(), pred[:, 2:4].detach().clamp_min(min_size), pred[:, 4:5]), 1)
    gt, field = gt.detach(), field.detach()
    if mode == 'C01':
        return angle_content_values(pred, gt, field, stride, 5, min_size), None
    if mode == 'POINTS':
        sampled_gt = _sample(field, gt, stride, 5)
        weights = point_weights(field, gt, sampled_gt, stride, valid_hw)
        return content_distances(p, gt, field, stride, sampled_gt, weights), None
    boxes, overlaps, steps = local_candidates(p.detach(), gt)
    if mode == 'DYNAMIC':
        ratio = gt[:, 2:4].amax(1) / gt[:, 2:4].amin(1).clamp_min(min_size)
        kappa = ((ratio.square()-1) / (ratio.square()+1)).square()
        kappa_ref = ((1.2**2-1) / (1.2**2+1))**2
        potential = (overlaps.amax(1) - overlaps[:, 0]).clamp_min(0)
        gate = kappa_ref/(kappa_ref+kappa) * potential/(potential+.01)
        return gate * angle_content_values(pred, gt, field, stride, 5, min_size), gate
    index = choose_candidate(boxes, overlaps, gt, field, stride, mode)
    target_angle = p[:, 4].detach() + steps[index]
    return (p[:, 4] - target_angle).sin().square(), steps[index]


@ROTATED_LOSSES.register_module()
class C01RefinementLoss(BDNearSquareAngleContentLoss):
    def __init__(self, refinement='C01', **kwargs):
        super().__init__(**kwargs)
        if refinement not in MODES:
            raise ValueError(refinement)
        self.refinement = refinement
        self.last_auxiliary = None
        self.last_positive_count = 0

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None,
                feature_maps=None, image_indices=None, feature_strides=None,
                pixel_pred=None, pixel_target=None, **kwargs):
        reduction = reduction_override or self.reduction
        if weight is not None and weight.ndim > 1:
            weight = weight.mean(-1)
        raw = pred.sum(-1)*0
        active = torch.ones(len(pred), dtype=torch.bool, device=pred.device) if weight is None else weight > 0
        self.last_positive_count = len(pred)
        geometry = raw.index_copy(0, active.nonzero().flatten(), self.geometry_values(pred[active], target[active].detach()).to(pred.dtype)) if active.any() else raw
        p = pred if pixel_pred is None else pixel_pred
        g = target if pixel_target is None else pixel_target
        ratio = g[:, 2:4].amax(1)/g[:, 2:4].amin(1).clamp_min(self.min_size)
        include = active if self.refinement == 'DYNAMIC' else active & (ratio <= self.square_ratio)
        content, selection = pred[:, 4]*0, []
        if self.semantic_weight and include.any():
            if feature_maps is None or image_indices is None or feature_strides is None:
                raise ValueError('Missing C01 pixel features')
            stride = float(feature_strides[0])
            for image in range(len(feature_maps[0])):
                indices = (include & (image_indices == image)).nonzero().flatten()
                if not len(indices):
                    continue
                field = feature_maps[0][image:image+1].detach().to(pred.dtype)
                field = field / torch.sqrt(field.square().sum(1, keepdim=True)+self.eps**2)
                for selected in indices.split(self.chunk_size):
                    values, marks = refined_values(p[selected], g[selected], field, stride,
                        self.image_shapes[image], self.refinement, self.min_size)
                    content = content.index_copy(0, selected, values)
                    if marks is not None:
                        selection.append(marks)
        auxiliary = self.semantic_weight * content
        self.last_auxiliary = self.loss_weight * weight_reduce_loss(auxiliary, weight, reduction, avg_factor)
        if reduction != 'none':
            self.last_diagnostics = dict(content_term=float(self.last_auxiliary.detach()), eligible=int(include.sum()))
            if selection:
                self.last_diagnostics['nonzero_selection_fraction'] = float((torch.cat(selection)!=0).float().mean())
        return self.loss_weight * weight_reduce_loss(geometry + auxiliary, weight, reduction, avg_factor)


@ROTATED_HEADS.register_module()
class C01RefinementFCOSHead(BDContentRotatedFCOSHead):
    def loss(self, cls_scores, bbox_preds, angle_preds, centernesses, gt_bboxes, gt_labels,
             img_metas, gt_bboxes_ignore=None, semantic_features=None):
        self.loss_bbox.image_shapes = [meta['img_shape'][:2] for meta in img_metas]
        self.loss_bbox.last_auxiliary = None
        self.loss_bbox.last_positive_count = 0
        return super().loss(cls_scores, bbox_preds, angle_preds, centernesses, gt_bboxes, gt_labels,
                            img_metas, gt_bboxes_ignore, semantic_features)
