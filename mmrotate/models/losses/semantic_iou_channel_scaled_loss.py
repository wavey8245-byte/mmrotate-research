"""Engineering channel-scale adaptation of semantic SIoU, not an author setting.

For each C-channel FPN level use f/sqrt(C), so the intra-region squared
L2 distance becomes the channel-mean squared difference. The cosine
context is invariant to this common positive scale except at eps-level
feature norms. Alpha/beta and all SIoU coupling equations are retained.
No adaptive scale, learnable temperature, detach or history is introduced.
"""
import math
from ..builder import ROTATED_LOSSES
from .semantic_iou_loss import SemanticIoULoss


@ROTATED_LOSSES.register_module()
class ChannelScaledSemanticIoULoss(SemanticIoULoss):
    def forward(self,pred,target,*args,feature_maps=None,**kwargs):
        if feature_maps is not None:
            if any(f.shape[1]<1 for f in feature_maps):
                raise ValueError('Semantic features must have positive channel count')
            feature_maps=[f/math.sqrt(f.shape[1]) for f in feature_maps]
        return super().forward(pred,target,*args,feature_maps=feature_maps,**kwargs)
