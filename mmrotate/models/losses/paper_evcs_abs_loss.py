"""Explicit engineering correction of printed EVCS: abs length difference.

Not the published Eq.(4), and not author code. Only the exponent's length
residual changes to absolute distance; no exponential or total-loss clamp.
The original published implementation and experiment remain unchanged.
"""
import torch

from ..builder import ROTATED_LOSSES
from .paper_evcs_loss import PaperKFIoUDecodedLoss


def evcs_abs_values(pred, target):
    """Return the corrected per-box EVCS and length/angular diagnostics."""
    def edges(box):
        box = box.double()
        signs = box.new_tensor([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]])
        local = box[:, None, 2:4] * signs
        c, s = box[:, None, 4].cos(), box[:, None, 4].sin()
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
    length_difference = (lt - lp).abs()
    value = 4 - (torch.exp(-length_difference) + cosine).sum(-1)
    return value, length_difference, cosine


@ROTATED_LOSSES.register_module()
class EVCSAbsCorrectionLoss(PaperKFIoUDecodedLoss):
    """KFIoU + lambda * abs-corrected EVCS; explicitly a modified method."""
    def __init__(self, evcs_weight=.006, **kwargs):
        super().__init__(**kwargs)
        if evcs_weight <= 0:
            raise ValueError('EVCS weight must be positive')
        self.evcs_weight = evcs_weight

    def values(self, pred, target):
        p, t = self.safe_boxes(pred), self.safe_boxes(target)
        base = self.base_values(p, t)
        evcs, length_difference, cosine = evcs_abs_values(p, t)
        value = base + self.evcs_weight * evcs
        with torch.no_grad():
            self.last_diagnostics = dict(
                kfiou_mean=float(base.mean()), evcs_mean=float(evcs.mean()),
                evcs_min=float(evcs.min()), evcs_max=float(evcs.max()),
                weighted_evcs_mean=float((self.evcs_weight * evcs).mean()),
                evcs_negative_fraction=float((evcs < -1e-12).double().mean()),
                mean_absolute_length_difference=float(length_difference.mean()),
                mean_edge_cosine=float(cosine.mean()),
                maximum_predicted_side=float(p[:, 2:4].max()),
                maximum_gt_side=float(t[:, 2:4].max()),
                maximum_area_ratio=float((p[:, 2] * p[:, 3] /
                                          (t[:, 2] * t[:, 3])).max()))
        if not torch.isfinite(value).all():
            raise FloatingPointError('Nonfinite corrected EVCS objective: ' +
                                     str(self.last_diagnostics))
        return value
