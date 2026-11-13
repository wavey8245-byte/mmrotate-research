"""Experimental log rotated IoU with WF as a small auxiliary loss.

This is a project variant, not the paper's reshaped-IoU objective. Keeping
the official log-IoU term prevents inaccurate zero-WF-loss boxes from having
zero total regression loss. Local gradient conflicts remain possible and
must be checked in controlled short runs before full experiments.
"""
from ..builder import ROTATED_LOSSES
from .rotated_iou_loss import RotatedIoULoss
from .rotated_wf_iou_loss import RotatedWFIoULoss


@ROTATED_LOSSES.register_module()
class RotatedIoUWithWFAuxLoss(RotatedIoULoss):
    """L_bbox = loss_weight * (L_log_IoU + wf_weight * L_WF)."""

    def __init__(self, wf_weight=0.1, tau=0.7, eps=1e-6,
                 reduction='mean', loss_weight=1.0):
        if not 0 <= wf_weight <= 1:
            raise ValueError('wf_weight must be in [0, 1]')
        super().__init__(mode='log', eps=eps, reduction=reduction,
                         loss_weight=loss_weight)
        self.wf_weight = wf_weight
        self.wf_aux = RotatedWFIoULoss(tau=tau, eps=eps,
                                     reduction=reduction, loss_weight=loss_weight)

    def forward(self, pred, target, weight=None, avg_factor=None,
                reduction_override=None, **kwargs):
        base = super().forward(pred, target, weight=weight, avg_factor=avg_factor,
                               reduction_override=reduction_override, **kwargs)
        if self.wf_weight == 0:
            return base
        auxiliary = self.wf_aux(pred, target, weight=weight, avg_factor=avg_factor,
                                reduction_override=reduction_override, **kwargs)
        return base + self.wf_weight * auxiliary
