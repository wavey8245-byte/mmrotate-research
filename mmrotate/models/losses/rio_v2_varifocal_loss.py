"""Quality classification from the audited RiO-DETRv2 criterion."""
import torch
import torch.nn.functional as F
from torch import nn

from ..builder import ROTATED_LOSSES


@ROTATED_LOSSES.register_module()
class RiOV2VarifocalLoss(nn.Module):
    """VFL with detached bounded-KLD quality and matched-positive normalization.

    Unlike generic IoU-weighted VFL, positive weights are exactly the quality,
    without an additional focal factor. Negative weights are alpha*p**gamma.
    """

    def __init__(self, use_sigmoid=True, alpha=.75, gamma=2., loss_weight=1.):
        super().__init__()
        if not use_sigmoid or not 0 <= alpha <= 1 or gamma < 0 or loss_weight < 0:
            raise ValueError('Invalid RiO v2 VFL configuration')
        self.use_sigmoid = True
        self.alpha = alpha
        self.gamma = gamma
        self.loss_weight = loss_weight

    def forward(self, pred, labels, quality, avg_factor):
        if labels.shape != pred.shape[:1] or quality.shape != labels.shape:
            raise ValueError('Expected one label and quality per prediction')
        classes = pred.shape[-1]
        foreground = (labels >= 0) & (labels < classes)
        onehot = F.one_hot(labels.clamp(0, classes), classes + 1)[..., :-1]
        onehot = onehot.to(pred.dtype) * foreground[:, None]
        target = onehot * quality.detach().clamp(0, 1)[:, None]
        weight = (self.alpha * pred.detach().sigmoid().pow(self.gamma)
                  * (1 - onehot) + target)
        weight = weight * (labels >= 0)[:, None]
        value = F.binary_cross_entropy_with_logits(
            pred, target, weight=weight, reduction='sum')
        return self.loss_weight * value / avg_factor
