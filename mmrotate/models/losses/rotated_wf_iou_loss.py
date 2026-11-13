"""WF reshaped IoU with an explicit extension to rotated boxes.

The paper defines d on the overlap, not on the distance between centers.
For rotated boxes the overlap is a polygon. In the ground-truth box's
coordinate frame, use the diagonal of its axis-aligned enclosing rectangle
for d and the diagonal enclosing both boxes for c. This recovers the paper's
geometry for horizontal boxes and is invariant to a joint rigid rotation.
It is our rotated extension, not an implementation supplied by the authors.
"""
import torch
import torch.nn as nn
from mmdet.models.losses.utils import weighted_loss

from ..builder import ROTATED_LOSSES


def _corners(boxes):
    """Convert (cx, cy, w, h, angle) to four corners, in MMCV convention."""
    signs = boxes.new_tensor([[.5, .5], [-.5, .5], [-.5, -.5], [.5, -.5]])
    local = boxes[:, None, 2:4] * signs
    cos, sin = boxes[:, 4].cos()[:, None], boxes[:, 4].sin()[:, None]
    x = local[..., 0] * cos - local[..., 1] * sin + boxes[:, None, 0]
    y = local[..., 0] * sin + local[..., 1] * cos + boxes[:, None, 1]
    return torch.stack([x, y], dim=-1)


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _inside(points, boxes, eps):
    relative = points - boxes[:, None, :2]
    cos, sin = boxes[:, 4].cos()[:, None], boxes[:, 4].sin()[:, None]
    local = torch.stack([
        relative[..., 0] * cos + relative[..., 1] * sin,
        -relative[..., 0] * sin + relative[..., 1] * cos], dim=-1)
    half_size = boxes[:, None, 2:4] / 2
    return (local.abs() <= half_size + eps * (half_size + 1)).all(-1)


def rotated_wf_geometry(pred, target, eps=1e-7):
    """Return aligned rotated IoU and d^2/c^2, retaining geometric gradients.

    All polygon operations use PyTorch and support CPU and CUDA. Boolean
    membership and vertex order are piecewise constant, as in differentiable
    polygon IoU. Coordinates are centered and rotated into the target frame
    before intersection, to reduce cancellation for large image coordinates.
    """
    if pred.shape != target.shape or pred.ndim != 2 or pred.shape[1] != 5:
        raise ValueError('Expected matching (N, 5) rotated boxes')
    if pred.numel() == 0:
        return pred.sum(-1), pred.sum(-1)
    if pred.dtype in (torch.float16, torch.bfloat16):
        pred, target = pred.float(), target.float()

    delta = pred[:, :2] - target[:, :2]
    cos, sin = target[:, 4].cos(), target[:, 4].sin()
    local_pred = torch.stack([
        delta[:, 0] * cos + delta[:, 1] * sin,
        -delta[:, 0] * sin + delta[:, 1] * cos,
        pred[:, 2], pred[:, 3], pred[:, 4] - target[:, 4]], dim=-1)
    zeros = target[:, 0] * 0
    local_target = torch.stack([
        zeros, zeros, target[:, 2], target[:, 3], zeros], dim=-1)
    corners_p, corners_t = _corners(local_pred), _corners(local_target)

    # Intersection vertices: corners inside the other box and edge crossings.
    start_p = corners_p[:, :, None, :]
    start_t = corners_t[:, None, :, :]
    edge_p = (corners_p.roll(-1, 1) - corners_p)[:, :, None, :]
    edge_t = (corners_t.roll(-1, 1) - corners_t)[:, None, :, :]
    delta_edge = start_t - start_p
    denominator = _cross(edge_p, edge_t)
    nonparallel = denominator != 0
    safe_denominator = torch.where(
        nonparallel, denominator, torch.ones_like(denominator))
    t = _cross(delta_edge, edge_t) / safe_denominator
    u = _cross(delta_edge, edge_p) / safe_denominator
    valid_edges = nonparallel & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    crossings = (start_p + t[..., None] * edge_p).reshape(-1, 16, 2)
    vertices = torch.cat([corners_p, corners_t, crossings], dim=1)
    mask = torch.cat([
        _inside(corners_p, local_target, eps),
        _inside(corners_t, local_pred, eps),
        valid_edges.reshape(-1, 16)], dim=1)
    # Invalid crossings can be very large for almost parallel edges; discard
    # them before computing a center, order, or area.
    vertices = torch.where(mask[..., None], vertices, torch.zeros_like(vertices))
    counts = mask.sum(-1)
    center = vertices.sum(1) / counts.clamp(min=1)[:, None]
    shifted = vertices - center[:, None, :]
    angles = torch.atan2(shifted[..., 1], shifted[..., 0])
    angles = angles.masked_fill(~mask, float('inf'))
    order = angles.argsort(dim=1)
    ordered = shifted.gather(1, order[..., None].expand(-1, -1, 2))
    positions = torch.arange(24, device=pred.device)[None, :]
    valid_positions = positions < counts[:, None]
    next_positions = (positions + 1) % counts.clamp(min=1)[:, None]
    following = ordered.gather(
        1, next_positions[..., None].expand(-1, -1, 2))
    terms = torch.where(valid_positions, _cross(ordered, following),
                        torch.zeros_like(angles))
    intersection = terms.sum(1).abs() / 2
    area_p = pred[:, 2].clamp(min=0) * pred[:, 3].clamp(min=0)
    area_t = target[:, 2].clamp(min=0) * target[:, 3].clamp(min=0)
    intersection = torch.minimum(intersection, torch.minimum(area_p, area_t))
    union = (area_p + area_t - intersection).clamp(min=eps**2)
    iou = (intersection / union).clamp(min=0, max=1)

    all_corners = torch.cat([corners_p, corners_t], dim=1)
    enclosing_span = all_corners.amax(1) - all_corners.amin(1)
    c_squared = enclosing_span.square().sum(-1).clamp(min=eps**2)
    overlap_min = vertices.masked_fill(~mask[..., None], float('inf')).amin(1)
    overlap_max = vertices.masked_fill(~mask[..., None], -float('inf')).amax(1)
    overlap_span = torch.where(
        (counts > 0)[:, None], overlap_max - overlap_min,
        torch.zeros_like(overlap_min))
    d_squared = torch.where(intersection > 0, overlap_span.square().sum(-1),
                            torch.full_like(intersection, eps**2))
    ratio = (d_squared / c_squared).clamp(min=0, max=1)
    return iou, ratio


def wf_weighting_factor(iou, ratio, tau=0.7):
    """Paper's three branches, including relu6, without detaching weights."""
    # Equivalent to relu6(1 / R), but avoids overflowing the reciprocal and
    # its derivative when the overlap vanishes. R is in [0, 1].
    low_quality = ratio.clamp(min=1.0 / 6).reciprocal()
    return torch.where(iou > tau, ratio,
                       torch.where(iou < 1 - tau, low_quality,
                                   torch.ones_like(iou)))


@weighted_loss
def rotated_wf_iou_loss(pred, target, tau=0.7, eps=1e-7):
    iou, ratio = rotated_wf_geometry(pred, target, eps=eps)
    factor = wf_weighting_factor(iou, ratio, tau=tau)
    return (1 - factor * iou).clamp(min=0)


@ROTATED_LOSSES.register_module()
class RotatedWFIoULoss(nn.Module):
    """Reshaped linear rotated IoU; accepts FCOS centerness weights.

    This is not w * (-log IoU). With zero overlap the pure IoU variant has
    loss 1 and no localization gradient, like ordinary linear IoU.
    """

    def __init__(self, tau=0.7, eps=1e-7, reduction='mean', loss_weight=1.0):
        super().__init__()
        if not 0.5 <= tau < 1:
            raise ValueError('tau must be in [0.5, 1)')
        if eps <= 0:
            raise ValueError('eps must be positive')
        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError('Invalid reduction')
        self.tau = tau
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None,
                reduction_override=None, **kwargs):
        if reduction_override not in (None, 'none', 'mean', 'sum'):
            raise ValueError('Invalid reduction_override')
        reduction = reduction_override or self.reduction
        if pred.numel() == 0:
            return pred.sum(-1) * 0 if reduction == 'none' else pred.sum() * 0
        if weight is not None and weight.dim() > 1:
            if weight.shape != pred.shape:
                raise ValueError('Coordinate weights must match (N, 5) boxes')
            weight = weight.mean(-1)
        if weight is not None and not torch.any(weight > 0):
            zeros = pred.sum(-1) * weight * 0
            return zeros if reduction == 'none' else zeros.sum()
        return self.loss_weight * rotated_wf_iou_loss(
            pred, target, weight, tau=self.tau, eps=self.eps,
            reduction=reduction, avg_factor=avg_factor)
