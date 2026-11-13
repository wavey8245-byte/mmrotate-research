"""Li et al., TGRS 2023, doi:10.1109/TGRS.2023.3305578.

Independent implementation of Eqs. (3), (11), (12), (17), not author code.
The signed rotation in Eq. (10)/(11) is retained literally. See the audit
document for the inconsistent coordinate notation in printed Eq. (7).
"""
import math

import torch
from torch import nn
from mmdet.models.losses.utils import weight_reduce_loss

from ..builder import ROTATED_LOSSES


def sgkld_divergence(pred, target, min_size=1e-7):
    """Directed D_4(target || pred), in decoded xywha / radians.

    Evaluate the fourth moments in Eq. (11) as sums of nonnegative terms.
    This is algebraically the same expression, without subtracting the large
    mixed odd moments B/D. No Monte Carlo, Gaussian surrogate or sqrt(D).
    """
    if pred.shape != target.shape or pred.ndim != 2 or pred.shape[-1] != 5:
        raise ValueError('Expected matching Nx5 decoded xywha boxes')
    if not torch.isfinite(pred).all() or not torch.isfinite(target).all():
        raise ValueError('Nonfinite boxes')
    if (pred[:, 2:4] < 0).any() or (target[:, 2:4] <= 0).any():
        raise ValueError('Predicted sizes must be nonnegative; GT positive')
    p, t = pred.double(), target.double()
    wp, hp = p[:, 2:4].clamp_min(min_size).unbind(-1)
    wt, ht = t[:, 2:4].clamp_min(min_size).unbind(-1)
    # Common scale cancels exactly; keeps fourth powers numerically small.
    scale = torch.stack([wp, hp, wt, ht], -1).amax(-1).detach()
    wp, hp, wt, ht = [v / scale for v in (wp, hp, wt, ht)]
    dx, dy = ((p[:, :2] - t[:, :2]) / scale[:, None]).unbind(-1)
    ct, st = t[:, 4].cos(), t[:, 4].sin()
    mux, muy = ct * dx + st * dy, -st * dx + ct * dy  # Eq. (3)
    angle = p[:, 4] - t[:, 4]
    c, s = angle.cos(), angle.sin()
    # E[U^2] for density 2/Gamma(1/4) exp(-U^4).
    second = math.gamma(.75) / math.gamma(.25)
    vx, vy = (wt / 2).square() * second, (ht / 2).square() * second
    qx, qy = (wt / 2).pow(4) / 4, (ht / 2).pow(4) / 4

    def fourth(a, b, shift):
        return (a.pow(4) * qx + b.pow(4) * qy
                + 6 * a.square() * b.square() * vx * vy
                + 6 * shift.square() * (a.square() * vx + b.square() * vy)
                + shift.pow(4))

    x = fourth(c, -s, c * mux - s * muy) / (wp / 2).pow(4)
    y = fourth(s, c, s * mux + c * muy) / (hp / 2).pow(4)
    divergence = wp.log() - wt.log() + hp.log() - ht.log() - .5 + x + y
    # KL is nonnegative analytically. Only removes roundoff at equality.
    return divergence.clamp_min(0)


@ROTATED_LOSSES.register_module()
class PaperSGKLDLoss(nn.Module):
    """Fixed n=4 and Eq. (17): 1 - 1/(1 + ln(1 + D))."""

    def __init__(self, min_size=1e-7, reduction='mean', loss_weight=1.):
        super().__init__()
        if min_size <= 0 or reduction not in ('none', 'mean', 'sum'):
            raise ValueError('Invalid min_size or reduction')
        if loss_weight < 0:
            raise ValueError('loss_weight must be nonnegative')
        self.min_size, self.reduction = min_size, reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None,
                reduction_override=None, **kwargs):
        reduction = reduction_override or self.reduction
        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError('Invalid reduction')
        if weight is not None and weight.ndim > 1:
            if weight.shape != pred.shape:
                raise ValueError('Coordinate weights must match boxes')
            weight = weight.mean(-1)
        if pred.numel() == 0:
            return pred.sum(-1) * 0 if reduction == 'none' else pred.sum() * 0
        active = (weight > 0 if weight is not None else
                  torch.ones(len(pred), dtype=torch.bool, device=pred.device))
        values = torch.where(active[:, None], pred, torch.zeros_like(pred)).double().sum(-1) * 0
        if active.any():
            d = sgkld_divergence(pred[active], target[active], self.min_size)
            log = torch.log1p(d)
            values = values.index_copy(0, active.nonzero().flatten(), log / (1 + log))
        return self.loss_weight * weight_reduce_loss(
            values.to(pred.dtype), weight, reduction, avg_factor)
