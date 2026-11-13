"""Independent FCOS adapter providing stable per-image GT history context."""
from ..builder import ROTATED_HEADS
from .semantic_rotated_fcos_head import SemanticRotatedFCOSHead


# FrSIoU 检测头：在语义头外再提供当前批次的图片名和 GT 标注身份。
@ROTATED_HEADS.register_module()
class FractionalRotatedFCOSHead(SemanticRotatedFCOSHead):
    def loss(self,cls_scores,bbox_preds,angle_preds,centernesses,gt_bboxes,gt_labels,
             img_metas,gt_bboxes_ignore=None,semantic_features=None):
        # 损失要用稳定图片名及 GT 行号查历史，因此先暂存本批次的元信息与 GT。
        contextual=hasattr(self.loss_bbox,'set_batch_context')
        if contextual:
            self.loss_bbox.set_batch_context(img_metas,gt_bboxes)
        # 实际预测展开、解码和损失计算复用 SemanticRotatedFCOSHead。
        try:
            return super().loss(cls_scores,bbox_preds,angle_preds,centernesses,gt_bboxes,
                gt_labels,img_metas,gt_bboxes_ignore=gt_bboxes_ignore,semantic_features=semantic_features)
        # 即使损失报错也清理上下文，避免下一批次误用上一批次的图片/GT。
        finally:
            if contextual:
                self.loss_bbox.clear_batch_context()
