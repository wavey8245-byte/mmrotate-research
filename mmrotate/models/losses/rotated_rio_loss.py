"""RiO-DETR's three regression terms adapted to decoded FCOS boxes.

This is a loss-only adaptation, not the complete RiO-DETR architecture.
Spatial residuals are normalized to image size by the FCOS head. KLD uses
the target-to-prediction direction and log1p/bounded transform of MMRotate's
stable GDLoss configuration, evaluated analytically without matrix inversion.
"""
import math

import torch
from mmdet.models.losses.utils import weight_reduce_loss
from torch import nn

from ..builder import ROTATED_LOSSES


def periodic_l1(pred, target):
    """Absolute shortest angular difference, in radians, with period pi."""
    return torch.remainder(pred - target + math.pi / 2, math.pi).sub(
        math.pi / 2).abs()


def stable_box_kld(pred, target, min_size=1e-4, covariance_epsilon=0.):
    """Bounded KL(N_target || N_pred) for xywha boxes.

    Compute in the prediction's principal axes. This avoids cancellation in a
    rotated covariance determinant when one predicted side is zero or tiny.
    min_size is in the same units as the input box. An optional covariance
    regularizer also uses those units squared.
    """
    wp, hp = pred[..., 2:4].clamp_min(min_size).unbind(-1)
    wt, ht = target[..., 2:4].clamp_min(min_size).unbind(-1)
    if covariance_epsilon > 0:
        # R diag(w^2/4, h^2/4) R^T + eps I is equivalent to these
        # principal side lengths. The public RiO baseline uses eps=1e-8
        # after normalizing boxes to image size.
        wp = (wp.square() + 4 * covariance_epsilon).sqrt()
        hp = (hp.square() + 4 * covariance_epsilon).sqrt()
        wt = (wt.square() + 4 * covariance_epsilon).sqrt()
        ht = (ht.square() + 4 * covariance_epsilon).sqrt()
    theta = pred[..., 4]
    dx, dy = (target[..., :2] - pred[..., :2]).unbind(-1)
    cosine, sine = theta.cos(), theta.sin()
    u = dx * cosine + dy * sine
    v = -dx * sine + dy * cosine
    delta = target[..., 4] - theta
    c2, s2 = delta.cos().square(), delta.sin().square()
    trace = ((wt / wp).square() * c2 + (ht / wp).square() * s2
             + (wt / hp).square() * s2 + (ht / hp).square() * c2)
    center = 4 * ((u / wp).square() + (v / hp).square())
    logdet = 2 * (wp.log() + hp.log() - wt.log() - ht.log())
    divergence = (0.5 * (center + trace + logdet - 2)).clamp_min(0)
    transformed = torch.log1p(divergence)
    return transformed / (1 + transformed)


@ROTATED_LOSSES.register_module()
class RotatedRiOLoss(nn.Module):
    """KLD + image-normalized spatial L1 + shortest-path angular L1.

All three terms use FCOS's existing ground-truth centerness weights and
normalizer. The paper lists weight 5 for KLD and the combined L1 term;
    the legacy default uses 5 for spatial L1 and 5 for angular error in radians.
    angle_normalizer=pi instead measures the angular error in unit-period
    coordinates. kld_normalized and covariance_epsilon align the public
    baseline's normalized-box covariance regularization when requested.
"""
    requires_spatial_normalizer = True

    def __init__(self, kld_weight=5.0, spatial_weight=5.0, angle_weight=5.0,
                 min_size=1e-4, reduction='mean', loss_weight=1.0,
                 angle_normalizer=1., kld_normalized=False,
                 covariance_epsilon=0.):
        super().__init__()
        if min_size <= 0 or reduction not in ('none', 'mean', 'sum'):
            raise ValueError('Invalid min_size or reduction')
        if min(kld_weight, spatial_weight, angle_weight, loss_weight) < 0:
            raise ValueError('Loss weights must be nonnegative')
        if angle_normalizer <= 0 or covariance_epsilon < 0:
            raise ValueError('Invalid angle normalizer or covariance epsilon')
        self.kld_weight = kld_weight
        self.spatial_weight = spatial_weight
        self.angle_weight = angle_weight
        self.min_size = min_size
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.angle_normalizer = angle_normalizer
        self.kld_normalized = kld_normalized
        self.covariance_epsilon = covariance_epsilon

    def forward(self, pred, target, weight=None, avg_factor=None,
                reduction_override=None, spatial_scale=None,
                return_components=False):
        if pred.shape != target.shape or pred.shape[-1] != 5:
            raise ValueError('Expected matching decoded Nx5 xywha boxes')
        if spatial_scale is None or spatial_scale.shape != pred[:, :4].shape:
            raise ValueError('Expected Nx4 spatial_scale from the FCOS head')
        reduction = reduction_override or self.reduction
        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError('Invalid reduction')
        if weight is not None and weight.ndim > 1:
            if weight.shape != pred.shape:
                raise ValueError('Coordinate weights must match the boxes')
            weight = weight.mean(-1)
        if pred.numel() == 0:
            zero = pred[:, 0] if reduction == 'none' else pred.sum() * 0
            components = {name: zero for name in
                          ('bbox_kld', 'bbox_l1', 'bbox_sp_l1')}
        else:
            kld_pred, kld_target = pred, target
            if self.kld_normalized:
                # Translation invariance permits a zero prediction origin.
                # Scale residuals before any covariance regularization so eps
                # has the same units as the public normalized-box baseline.
                kld_pred = torch.cat((torch.zeros_like(pred[:, :2]),
                    pred[:, 2:4] * spatial_scale[:, 2:4], pred[:, 4:]), -1)
                kld_target = torch.cat(((target[:, :2] - pred[:, :2])
                    * spatial_scale[:, :2],
                    target[:, 2:4] * spatial_scale[:, 2:4], target[:, 4:]), -1)
            terms = {
                'bbox_kld': self.kld_weight * stable_box_kld(
                    kld_pred, kld_target, self.min_size, self.covariance_epsilon),
                'bbox_l1': self.spatial_weight * (
                    (pred[:, :4] - target[:, :4]) * spatial_scale).abs().sum(-1),
                'bbox_sp_l1': self.angle_weight * periodic_l1(
                    pred[:, 4], target[:, 4]) / self.angle_normalizer,
            }
            components = {name: self.loss_weight * weight_reduce_loss(
                value, weight, reduction, avg_factor)
                for name, value in terms.items()}
        return components if return_components else sum(components.values())
