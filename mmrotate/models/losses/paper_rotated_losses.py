"""Independent, paper-based ports; these are NOT authors' released code.

Thai et al. (2025), arXiv:2510.16445: BD and optional AGBB.
Yang and Mohamed (2025), doi:10.1016/j.neucom.2024.129248: DCAGPL.
Input: decoded aligned Nx5 (cx, cy, w, h, theta) boxes, angle in radians.
Use custom_imports in configs; existing registries/files need no edits.
"""
import torch
from torch import nn
from mmdet.models.losses.utils import weight_reduce_loss

from ..builder import ROTATED_LOSSES


def _check_boxes(pred, target):
    if pred.shape != target.shape or pred.ndim != 2 or pred.shape[-1] != 5:
        raise ValueError('Expected matching decoded Nx5 xywha boxes')
    if not torch.isfinite(pred).all() or not torch.isfinite(target).all():
        raise ValueError('Nonfinite input boxes')
    if (pred[:, 2:4] < 0).any() or (target[:, 2:4] <= 0).any():
        raise ValueError('Predicted sizes must be nonnegative; GT positive')


def _pair_geometry(pred, target, min_size, agbb=False,
                   square_ratio_threshold=1.1, delta=5.):
    """Stable scalar evaluation of the paper's covariance operations.

    A common detached size scale cancels from both dimensionless losses.
    Positive determinant identities avoid subtractive cancellation in thin
    rotated boxes. Float64 is used internally, including under AMP.
    AGBB follows the paper's diag(h'/2, w'/2), rotation 4*theta, exactly.
    Applying it to both boxes is decided by GT aspect ratio only. The paper
    does not specify a numeric square threshold; 1.1 is an explicit port
    setting, not a reported paper hyperparameter.
    """
    _check_boxes(pred, target)
    p, t = pred.double(), target.double()
    wp, hp = p[:, 2:4].clamp_min(min_size).unbind(-1)
    wt, ht = t[:, 2:4].clamp_min(min_size).unbind(-1)
    rp, rt = p[:, 4], t[:, 4]
    if agbb:
        square = torch.maximum(wt / ht, ht / wt) <= square_ratio_threshold
        cp, ct = torch.cos(4 * rp) / delta, torch.cos(4 * rt) / delta
        wp, hp = (torch.where(square, hp * (1 + cp), wp),
                  torch.where(square, wp * (1 - cp), hp))
        wt, ht = (torch.where(square, ht * (1 + ct), wt),
                  torch.where(square, wt * (1 - ct), ht))
        rp, rt = torch.where(square, 4 * rp, rp), torch.where(square, 4 * rt, rt)
    scale = torch.stack([wp, hp, wt, ht], -1).amax(-1).detach()
    ap, bp = (wp / (2 * scale)).square(), (hp / (2 * scale)).square()
    at, bt = (wt / (2 * scale)).square(), (ht / (2 * scale)).square()
    c2, s2 = (rp - rt).cos().square(), (rp - rt).sin().square()
    dp, dt = ap * bp, at * bt
    # det(Sigma_p + Sigma_t), expressed as a sum of positive terms.
    ds = dp + dt + c2 * (ap * bt + bp * at) + s2 * (ap * at + bp * bt)
    dx, dy = ((p[:, :2] - t[:, :2]) / scale[:, None]).unbind(-1)
    up, vp = dx * rp.cos() + dy * rp.sin(), -dx * rp.sin() + dy * rp.cos()
    ut, vt = dx * rt.cos() + dy * rt.sin(), -dx * rt.sin() + dy * rt.cos()
    return ap, bp, at, bt, dp, dt, ds, up, vp, ut, vt


def bd_values(pred, target, alpha=3., min_size=1e-4, agbb=False,
              square_ratio_threshold=1.1, delta=5.):
    """Thai Eq.(2) and unnumbered weighted BD/AGBB definitions, per box."""
    ap, bp, at, bt, dp, dt, ds, up, vp, ut, vt = _pair_geometry(
        pred, target, min_size, agbb, square_ratio_threshold, delta)
    # Sigma_bar = (Sigma_p + Sigma_t)/2. In 2D adj(A+B)=adj(A)+adj(B).
    mahalanobis = 2 * (bp * up.square() + ap * vp.square()
                       + bt * ut.square() + at * vt.square()) / ds
    covariance = .5 * torch.log(ds / (4 * torch.sqrt(dp * dt)))
    distance = (alpha * mahalanobis / 8 + covariance).clamp_min(0)
    # Preserve exact sqrt(D) values; choose zero subgradient at D=0.
    # The inactive branch must remain finite to avoid 0 * inf in backward.
    root = torch.where(distance > 0, distance.clamp_min(1e-30).sqrt(),
                       torch.zeros_like(distance))
    return 1 - 1 / (1 + root)


def dcagpl_values(pred, target, min_size=1e-4, eps=1e-6):
    """Yang & Mohamed Eqs.(6),(8)-(11),(13); per-box Lc and Lkf.

    The product covariance determinant is det_p*det_t/det(Sigma_p+Sigma_t).
    This is an algebraic evaluation of Eq.(8), not an extra loss term.
    No rescaling by 3 or subtracting ln(3): the printed KFIoU is retained.
    Autograd differentiates Eq.(11), not the inconsistent printed Eq.(12).
    """
    _, _, at, bt, dp, dt, ds, _, _, ut, vt = _pair_geometry(
        pred, target, min_size)
    center = torch.log1p(ut.square() / at + vt.square() / bt)
    area_p, area_t = 4 * dp.sqrt(), 4 * dt.sqrt()
    area_product = 4 * (dp * dt / ds).sqrt()
    kfiou = area_product / (area_p + area_t - area_product)
    shape = -torch.log(kfiou + eps)
    return center, shape


class _DecodedPaperLoss(nn.Module):
    def __init__(self, min_size=1e-4, reduction='mean', loss_weight=1.):
        super().__init__()
        if min_size <= 0 or reduction not in ('none', 'mean', 'sum'):
            raise ValueError('Invalid size floor or reduction')
        if loss_weight < 0:
            raise ValueError('loss_weight must be nonnegative')
        self.min_size = min_size
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None,
                reduction_override=None, **kwargs):
        _check_boxes(pred, target)
        reduction = reduction_override or self.reduction
        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError('Invalid reduction_override')
        if weight is not None and weight.ndim > 1:
            if weight.shape != pred.shape:
                raise ValueError('Coordinate weights must match boxes')
            weight = weight.mean(-1)
        if pred.numel() == 0:
            return pred.sum(-1) * 0 if reduction == 'none' else pred.sum() * 0
        # Mask before evaluation: ignored boxes must not trigger unstable
        # matrix/exponential operations or change centerness normalization.
        active = torch.ones(pred.shape[0], dtype=torch.bool, device=pred.device)
        if weight is not None:
            active = weight > 0
        raw = pred.double().sum(-1) * 0
        if active.any():
            raw = raw.index_copy(0, active.nonzero().flatten(),
                                 self.values(pred[active], target[active]))
        raw = raw.to(dtype=pred.dtype)
        return self.loss_weight * weight_reduce_loss(
            raw, weight=weight, reduction=reduction, avg_factor=avg_factor)


@ROTATED_LOSSES.register_module()
class PaperBDLoss(_DecodedPaperLoss):
    def __init__(self, alpha=3., agbb=False, square_ratio_threshold=1.1,
                 delta=5., **kwargs):
        super().__init__(**kwargs)
        if alpha <= 0 or delta <= 1 or square_ratio_threshold < 1:
            raise ValueError('Require alpha>0, delta>1, square threshold>=1')
        self.alpha = alpha
        self.agbb = agbb
        self.square_ratio_threshold = square_ratio_threshold
        self.delta = delta

    def values(self, pred, target):
        return bd_values(pred, target, self.alpha, self.min_size, self.agbb,
                         self.square_ratio_threshold, self.delta)


@ROTATED_LOSSES.register_module()
class PaperDCAGPLLoss(_DecodedPaperLoss):
    def __init__(self, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        if eps <= 0:
            raise ValueError('eps must be positive')
        self.eps = eps

    def values(self, pred, target):
        center, shape = dcagpl_values(pred, target, self.min_size, self.eps)
        return center + shape


def evcs_literal(pred, target):
    """CRL Eq.(4) verbatim in meaning, for audit ONLY, not registered.

    No absolute value has been inserted into the exponent. This published
    expression is unbounded below and can overflow. Do not train with it.
    No guessed correction is silently offered as the paper's method.
    """
    _check_boxes(pred, target)

    def edges(box):
        box = box.double()
        signs = box.new_tensor([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]])
        local = box[:, None, 2:4] * signs
        c, s = box[:, None, 4].cos(), box[:, None, 4].sin()
        # Translation cancels in edges; omit it to avoid precision loss.
        points = torch.stack([local[..., 0] * c - local[..., 1] * s,
                              local[..., 0] * s + local[..., 1] * c], -1)
        left = points[..., 0] == points[..., 0].amin(-1, keepdim=True)
        index = points[..., 1].masked_fill(~left, float('inf')).argmin(-1)
        rows = torch.arange(len(box), device=box.device)
        origin = points[rows, index]
        return torch.stack([points[rows, (index + 1) % 4] - origin,
                            points[rows, (index - 1) % 4] - origin], 1)

    ep, et = edges(pred), edges(target)
    lp, lt = ep.norm(dim=-1), et.norm(dim=-1)
    cosine = (ep * et).sum(-1) / (lp * lt)
    return 4 - (torch.exp(-(lt - lp)) + cosine).sum(-1)
