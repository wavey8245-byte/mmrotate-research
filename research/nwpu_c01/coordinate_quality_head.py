"""Change the centerness training target without changing regression weights."""
import torch
from mmcv.ops import box_iou_rotated
from mmrotate.models.builder import ROTATED_HEADS
from mmrotate.models.dense_heads.bd_content_rotated_fcos_head import BDContentRotatedFCOSHead


@ROTATED_HEADS.register_module()
class CoordinateQualityHead(BDContentRotatedFCOSHead):
    def __init__(self, quality_fraction=0.25, permute_quality=False, **kwargs):
        super().__init__(**kwargs)
        self.quality_fraction = quality_fraction
        self.permute_quality = permute_quality

    def loss(self, *args, **kwargs):
        quality = None

        @torch.no_grad()
        def measure(module, inputs):
            nonlocal quality
            # The two boxes share a prior and stride, so the common rescaling
            # from native to pixel coordinates preserves their pairwise IoU.
            p, g = [b.detach().float().clone() for b in inputs[:2]]
            p[:, 2:4].clamp_min_(1e-4)
            g[:, 2:4].clamp_min_(1e-4)
            quality = torch.nan_to_num(box_iou_rotated(p.contiguous(), g.contiguous(),
                                                     aligned=True)).clamp_(0, 1)

        def change_target(module, inputs):
            logits, target = inputs[:2]
            values = quality
            if values is None or values.shape != target.shape:
                raise RuntimeError('Missing matched localization quality for positive positions')
            if self.permute_quality:
                values = values.clone()
                bins = (target.detach() * 4).long().clamp_max(3)
                for bucket in range(4):
                    indices = (bins == bucket).nonzero().flatten()
                    if len(indices) > 1:
                        values[indices] = quality[indices].roll(len(indices) // 2)
            mixed = target.detach().lerp(values.to(target.dtype), self.quality_fraction)
            return (logits, mixed, *inputs[2:])

        capture = self.loss_bbox.register_forward_pre_hook(measure)
        replace = self.loss_centerness.register_forward_pre_hook(change_target)
        try:
            return super().loss(*args, **kwargs)
        finally:
            replace.remove()
            capture.remove()
