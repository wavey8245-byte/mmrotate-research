"""Diagnostic FCOS ports of KFIoU and the published CRL Eq.(4)-(5).

Independent implementation, not author code. EVCS is intentionally the
printed signed exponential, with no absolute value or exponential clamp.
It is unbounded below; finite-value checks abort rather than repair it.
Inputs retain FCOS native decoded training units (FPN-stride units).
This is a loss transfer experiment, not a full two-stage CRL reproduction.
"""
import torch

from ..builder import ROTATED_LOSSES
from .kf_iou_loss import kfiou_loss
from .paper_rotated_losses import _DecodedPaperLoss, evcs_literal


@ROTATED_LOSSES.register_module()
class PaperKFIoUDecodedLoss(_DecodedPaperLoss):
    """MMRotate KFIoU with decoded FCOS center residuals, log transform."""
    def __init__(self, fun='ln', beta=1. / 9., eps=1e-6,
                 min_size=1e-7, **kwargs):
        super().__init__(min_size=min_size, **kwargs)
        if fun not in ('none', 'ln', 'exp') or beta <= 0 or eps <= 0:
            raise ValueError('Invalid KFIoU specification')
        self.fun, self.beta, self.eps = fun, beta, eps
        self.last_diagnostics = {}

    def safe_boxes(self, boxes):
        # Native FCOS distances may collapse to zero after its normal clamp.
        # This floor matches MMRotate KFIoU's size floor and also defines
        # EVCS cosines at collapsed predictions; it does not cap the exponent.
        boxes = boxes.double()
        return torch.cat((boxes[:, :2], boxes[:, 2:4].clamp_min(self.min_size),
                          boxes[:, 4:]), dim=-1)

    def base_values(self, pred, target):
        return kfiou_loss(pred, target, pred_decode=pred,
                          targets_decode=target, fun=self.fun, beta=self.beta,
                          eps=self.eps, reduction='none')

    def values(self, pred, target):
        p, t = self.safe_boxes(pred), self.safe_boxes(target)
        value = self.base_values(p, t)
        if not torch.isfinite(value).all():
            raise FloatingPointError('Nonfinite native KFIoU value')
        self.last_diagnostics = dict(kfiou_mean=float(value.detach().mean()))
        return value


@ROTATED_LOSSES.register_module()
class PaperEVCSPublishedLoss(PaperKFIoUDecodedLoss):
    """Native KFIoU + lambda * literal published EVCS; no total clamp."""
    def __init__(self, evcs_weight=.006, **kwargs):
        super().__init__(**kwargs)
        if evcs_weight <= 0:
            raise ValueError('EVCS weight must be positive')
        self.evcs_weight = evcs_weight

    def values(self, pred, target):
        p, t = self.safe_boxes(pred), self.safe_boxes(target)
        base = self.base_values(p, t)
        evcs = evcs_literal(p, t)
        value = base + self.evcs_weight * evcs
        with torch.no_grad():
            self.last_diagnostics = dict(
                kfiou_mean=float(base.mean()), evcs_mean=float(evcs.mean()),
                evcs_min=float(evcs.min()), evcs_max=float(evcs.max()),
                weighted_evcs_mean=float((self.evcs_weight * evcs).mean()),
                evcs_negative_fraction=float((evcs < 0).double().mean()),
                maximum_predicted_side=float(p[:, 2:4].max()),
                maximum_gt_side=float(t[:, 2:4].max()),
                maximum_area_ratio=float((p[:, 2] * p[:, 3] /
                                          (t[:, 2] * t[:, 3])).max()))
        if not torch.isfinite(value).all():
            raise FloatingPointError('Nonfinite published EVCS objective: ' +
                                     str(self.last_diagnostics))
        if (value.abs() > torch.finfo(pred.dtype).max).any():
            raise FloatingPointError('Published EVCS exceeds model dtype range: ' +
                                     str(self.last_diagnostics))
        return value
