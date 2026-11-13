"""FCOS loss adapter for per-positive pixel boxes and differentiable FPN semantics.

Convolutional architecture, assignment, classification, centerness and test
behavior are inherited unchanged. Existing RotatedFCOSHead is not edited.
"""
import torch
from mmcv.runner import force_fp32
from mmdet.core import reduce_mean
from ..builder import ROTATED_HEADS
from .rotated_fcos_head import RotatedFCOSHead


# 语义损失适配头：继承 FCOS 的卷积、正样本分配和推理，只补充训练损失输入。
@ROTATED_HEADS.register_module()
class SemanticRotatedFCOSHead(RotatedFCOSHead):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        # 本适配器联合优化旋转框，且要求 FCOS 训练距离已按 stride 归一化。
        if self.separate_angle or not self.norm_on_bbox:
            raise ValueError('This semantic adapter requires norm_on_bbox=True, separate_angle=False')

    def forward_train(self,x,img_metas,gt_bboxes,gt_labels=None,gt_bboxes_ignore=None,proposal_cfg=None,**kwargs):
        if gt_labels is None:
            raise ValueError('Semantic FCOS requires class labels')
        # x 为各层原始 FPN 特征；先做标准预测，再把同一组特征传给语义损失。
        outs=self(x)
        losses=self.loss(*outs,gt_bboxes,gt_labels,img_metas,
                         gt_bboxes_ignore=gt_bboxes_ignore,semantic_features=x)
        if proposal_cfg is None:
            return losses
        return losses,self.get_bboxes(*outs,img_metas,cfg=proposal_cfg)

    # 框解码、几何运算与语义计算使用 float32，避免混合精度带来的数值问题。
    @force_fp32(apply_to=('cls_scores','bbox_preds','angle_preds','centernesses','semantic_features'))
    def loss(self,cls_scores,bbox_preds,angle_preds,centernesses,gt_bboxes,gt_labels,
             img_metas,gt_bboxes_ignore=None,semantic_features=None):
        if semantic_features is None:
            raise ValueError('Pass the original FPN features for semantic supervision')
        sizes=[score.shape[-2:] for score in cls_scores]
        points=self.prior_generator.grid_priors(sizes,dtype=bbox_preds[0].dtype,device=bbox_preds[0].device)
        # 沿用父类 FCOS 的中心采样、回归范围与冲突分配规则，得到各层监督目标。
        labels,bbox_targets,angle_targets=self.get_targets(points,gt_bboxes,gt_labels)
        batch=cls_scores[0].shape[0]
        # 统一按「层 → 批内图片 → 空间位置」展平，保持预测、标签和身份索引一一对应。
        def flatten(values,channels):
            return torch.cat([x.permute(0,2,3,1).reshape(-1,channels) for x in values])
        cls=flatten(cls_scores,self.cls_out_channels)
        distances=flatten(bbox_preds,4)
        angles=flatten(angle_preds,1)
        ctr=flatten(centernesses,1).reshape(-1)
        labels=torch.cat(labels)
        target_distances=torch.cat(bbox_targets)
        target_angles=torch.cat(angle_targets)
        # 为每个展平位置生成图片坐标、批内图片号、FPN 层号和 stride。
        # 语义采样和历史身份都依赖这些索引，不能与展平顺序错位。
        point=torch.cat([p.repeat(batch,1) for p in points])
        image=torch.cat([torch.arange(batch,device=point.device).repeat_interleave(len(p)) for p in points])
        level=torch.cat([torch.full((batch*len(p),),i,device=point.device,dtype=torch.long) for i,p in enumerate(points)])
        stride=torch.cat([distances.new_full((batch*len(p),1),self.strides[i]) for i,p in enumerate(points)])
        # 分类使用全部位置；旋转框和 centerness 仅对前景正样本计算。
        positive=((labels>=0)&(labels<self.num_classes)).nonzero().reshape(-1)
        count=max(reduce_mean(torch.tensor(len(positive),dtype=torch.float,device=point.device)),1.)
        loss_cls=self.loss_cls(cls,labels,avg_factor=count)
        d,td=distances[positive],target_distances[positive]
        # 权重由 GT 的四边距离计算，不使用网络预测的 centerness 来给回归损失加权。
        centerness_targets=self.centerness_target(td)
        # 回归分母是 GT centerness 总和；与分类/centerness 使用的正样本数量分母不同。
        denominator=max(reduce_mean(centerness_targets.sum().detach()),1e-6)
        if len(positive):
            # Undo FPN distance normalization BEFORE decoding into image pixels.
            # 训练距离乘回 stride，角度不乘 stride；预测框与 GT 均解码为图片像素单位。
            pencoded=torch.cat((d*stride[positive],angles[positive]),-1)
            tencoded=torch.cat((td*stride[positive],target_angles[positive]),-1)
            pred=self.bbox_coder.decode(point[positive],pencoded)
            target=self.bbox_coder.decode(point[positive],tencoded)
            # 向损失额外传入原始 FPN、图片号和层号，使每个框取到正确的特征。
            loss_bbox=self.loss_bbox(pred,target,weight=centerness_targets,avg_factor=denominator,
                                     feature_maps=semantic_features,image_indices=image[positive],
                                     level_indices=level[positive],feature_strides=self.strides)
            loss_ctr=self.loss_centerness(ctr[positive],centerness_targets,avg_factor=count)
        # 无正样本时返回可反向的零回归/centerness 损失；分类损失仍正常计算。
        else:
            loss_bbox=d.sum()
            loss_ctr=ctr[positive].sum()
            self.loss_bbox.last_diagnostics={}
        # 训练总损失由框架相加：Focal 分类 + FrSIoU 回归 + centerness BCE。
        return dict(loss_cls=loss_cls,loss_bbox=loss_bbox,loss_centerness=loss_ctr)
