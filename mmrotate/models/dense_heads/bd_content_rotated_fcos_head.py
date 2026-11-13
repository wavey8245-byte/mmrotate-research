"""Keep native FCOS BD decoding; separately decode pixels for P3 sampling."""
import torch
from mmcv.runner import force_fp32
from mmdet.core import reduce_mean
from ..builder import ROTATED_HEADS
from .semantic_rotated_fcos_head import SemanticRotatedFCOSHead


@ROTATED_HEADS.register_module()
class BDContentRotatedFCOSHead(SemanticRotatedFCOSHead):
    @force_fp32(apply_to=('cls_scores','bbox_preds','angle_preds','centernesses','semantic_features'))
    def loss(self, cls_scores, bbox_preds, angle_preds, centernesses,
             gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None,
             semantic_features=None):
        if semantic_features is None:
            raise ValueError('Original FPN features required')
        sizes = [score.shape[-2:] for score in cls_scores]
        points = self.prior_generator.grid_priors(sizes, dtype=bbox_preds[0].dtype, device=bbox_preds[0].device)
        labels, bbox_targets, angle_targets = self.get_targets(points, gt_bboxes, gt_labels)
        batch = cls_scores[0].shape[0]
        def flatten(values, channels):
            return torch.cat([value.permute(0,2,3,1).reshape(-1,channels) for value in values])
        cls = flatten(cls_scores, self.cls_out_channels)
        distances, angles = flatten(bbox_preds,4), flatten(angle_preds,1)
        centerness = flatten(centernesses,1).reshape(-1)
        labels = torch.cat(labels)
        td, ta = torch.cat(bbox_targets), torch.cat(angle_targets)
        point = torch.cat([p.repeat(batch,1) for p in points])
        image = torch.cat([torch.arange(batch,device=point.device).repeat_interleave(len(p)) for p in points])
        strides = torch.cat([distances.new_full((batch*len(p),1),self.strides[i]) for i,p in enumerate(points)])
        positive = ((labels>=0)&(labels<self.num_classes)).nonzero().reshape(-1)
        count = max(reduce_mean(torch.tensor(len(positive),dtype=torch.float,device=point.device)),1.)
        loss_cls = self.loss_cls(cls,labels,avg_factor=count)
        d, target_d = distances[positive], td[positive]
        target_ctr = self.centerness_target(target_d)
        denominator = max(reduce_mean(target_ctr.sum().detach()),1e-6)
        if len(positive):
            pencoded = torch.cat((d,angles[positive]),-1)
            tencoded = torch.cat((target_d,ta[positive]),-1)
            # EXACT ordinary RotatedFCOSHead inputs to PaperBDLoss.
            pred = self.bbox_coder.decode(point[positive],pencoded)
            target = self.bbox_coder.decode(point[positive],tencoded)
            # Extra pixel decoding affects ONLY the new content branch.
            pixel_pred = self.bbox_coder.decode(point[positive],torch.cat((d*strides[positive],angles[positive]),-1))
            pixel_target = self.bbox_coder.decode(point[positive],torch.cat((target_d*strides[positive],ta[positive]),-1))
            loss_bbox = self.loss_bbox(pred,target,weight=target_ctr,avg_factor=denominator,
                feature_maps=semantic_features,image_indices=image[positive],feature_strides=self.strides,
                pixel_pred=pixel_pred,pixel_target=pixel_target)
            loss_ctr = self.loss_centerness(centerness[positive],target_ctr,avg_factor=count)
        else:
            loss_bbox = d.sum()
            loss_ctr = centerness[positive].sum()
            self.loss_bbox.last_diagnostics = {}
        return dict(loss_cls=loss_cls,loss_bbox=loss_bbox,loss_centerness=loss_ctr)
