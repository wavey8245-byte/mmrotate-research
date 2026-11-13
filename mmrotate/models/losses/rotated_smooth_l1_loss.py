"""Legacy decoded and corrected encoded Smooth L1 for rotated FCOS."""
from mmdet.models.losses import SmoothL1Loss

from ..builder import ROTATED_LOSSES


@ROTATED_LOSSES.register_module()
class RotatedSmoothL1Loss(SmoothL1Loss):
    """Apply standard Smooth L1 to (cx, cy, w, h, angle).

    With FCOS norm_on_bbox=True, decoded widths/heights and center offsets
    are in stride-normalized units; point origins remain in image pixels.
    Kept unchanged for loading historical experiment configurations.
    FCOS provides one centerness weight per box; broadcast over coordinates.
    """

    def forward(self, pred, target, weight=None, **kwargs):
        if weight is not None and weight.dim() == 1:
            weight = weight[:, None]
        return super().forward(pred, target, weight=weight, **kwargs)


@ROTATED_LOSSES.register_module()
class EncodedRotatedSmoothL1Loss(RotatedSmoothL1Loss):
    """Standard Smooth L1 on normalized l/t/r/b and canonical GT angle.

    The head supplies raw normalized distances before clamping and decoding.
    This permits negative predictions to recover and avoids coupling
    center-coordinate errors to predicted angles. All five parameters regress
    the same canonical GT representation. Do not wrap the angle residual alone:
    shifting an encoded angle by pi also requires swapping l/r and t/b.
    Coordinate weights and centerness normalization match the legacy loss.
    """
    requires_encoded_boxes = True

    def forward(self, pred, target, weight=None, **kwargs):
        if pred.shape != target.shape or pred.shape[-1] != 5:
            raise ValueError('Expected matching Nx5 encoded distance/angle tensors')
        return super().forward(pred, target, weight=weight, **kwargs)
