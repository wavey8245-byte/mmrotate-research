"""Experimental additive BD + spatial-feature regression; no history or adaptive gates.

FPN values are detached, but predicted sampling coordinates keep gradients.
This is a proposal, not a published method or established improvement.
"""
import torch
from torch import nn
import torch.nn.functional as F
from mmdet.models.losses.utils import weight_reduce_loss
from ..builder import ROTATED_LOSSES
from .paper_rotated_losses import bd_values
from .semantic_iou_loss import sample_box_features


def spatial_feature_distance(fp, ft, mode='spatial', eps=1e-6):
    """Square-grid C4 orbit makes equivalent box parameterizations agree.

    Four rotations match ROI grids, not physical image rotations. The minimum
    is piecewise differentiable; branch switches need auditing, not assumed smooth.
    GT-zero locations are excluded; a zero predicted vector has cosine zero.
    """
    if mode == 'pooled':
        return (1-F.cosine_similarity(fp.mean((2,3)), ft.mean((2,3)), dim=1, eps=eps)).clamp(0,2)
    if mode != 'spatial':
        raise ValueError('Expected spatial or pooled semantics')
    target=ft.detach()
    valid=(target.square().sum(1).sqrt()>eps).to(fp.dtype)
    candidates=[]
    for k in range(4):
        rotated=torch.rot90(fp,k,dims=(-2,-1))
        cosine=F.cosine_similarity(rotated,target,dim=1,eps=eps)
        distance=((1-cosine).clamp(0,2)*valid).sum((1,2))/valid.sum((1,2)).clamp_min(1)
        candidates.append(distance)
    return torch.stack(candidates,-1).min(-1).values


def semantic_bd_components(pred,target,feature_maps,image_indices,level_indices,strides,
                           alpha=3.,grid_size=5,mode='spatial',min_size=1e-4,eps=1e-6):
    geometry=bd_values(pred,target,alpha=alpha,min_size=min_size,agbb=False).to(pred.dtype)
    semantic=pred.sum(-1)*0
    for level,features in enumerate(feature_maps):
        for im in range(features.shape[0]):
            ids=((image_indices==im)&(level_indices==level)).nonzero().flatten()
            if ids.numel()==0:
                continue
            # Detach map VALUES before grid_sample; do not detach predicted grids.
            f=features[im:im+1].detach()
            p=pred[ids]
            p=torch.cat([p[:,:2],p[:,2:4].clamp_min(min_size),p[:,4:]],-1)
            fp=sample_box_features(f,p,strides[level],grid_size)
            ft=sample_box_features(f,target[ids].detach(),strides[level],grid_size).detach()
            sem=spatial_feature_distance(fp,ft,mode,eps)
            semantic=semantic.index_copy(0,ids,sem)
    return geometry,semantic


@ROTATED_LOSSES.register_module()
class SpatialSemanticBDLoss(nn.Module):
    def __init__(self,alpha=3.,semantic_weight=.1,semantic_mode='spatial',grid_size=5,
                 min_size=1e-4,cosine_eps=1e-6,reduction='mean',loss_weight=1.):
        super().__init__()
        if alpha<=0 or semantic_weight<0 or min_size<=0 or cosine_eps<=0 or grid_size<2:
            raise ValueError('Invalid loss settings')
        if semantic_mode not in ('spatial','pooled') or reduction not in ('none','mean','sum') or loss_weight<0:
            raise ValueError('Invalid semantic mode/reduction/weight')
        self.alpha,self.semantic_weight,self.semantic_mode=alpha,semantic_weight,semantic_mode
        self.grid_size,self.min_size,self.cosine_eps=grid_size,min_size,cosine_eps
        self.reduction,self.loss_weight=reduction,loss_weight
        self.last_diagnostics={}

    def forward(self,pred,target,weight=None,avg_factor=None,reduction_override=None,
                feature_maps=None,image_indices=None,level_indices=None,feature_strides=None,**kwargs):
        if pred.shape!=target.shape or pred.ndim!=2 or pred.shape[-1]!=5:
            raise ValueError('Expected paired Nx5 pixel boxes')
        reduction=reduction_override or self.reduction
        if weight is not None and weight.ndim>1:
            weight=weight.mean(-1)
        raw=pred.sum(-1)*0
        if not len(pred):
            self.last_diagnostics={}
            return raw if reduction=='none' else raw.sum()
        active=torch.ones(len(pred),device=pred.device,dtype=torch.bool) if weight is None else weight>0
        if active.any():
            p,t=pred[active],target[active].detach()
            if self.semantic_weight==0:
                g=bd_values(p,t,alpha=self.alpha,min_size=self.min_size,agbb=False).to(p.dtype)
                sem=g*0
            else:
                if any(x is None for x in [feature_maps,image_indices,level_indices,feature_strides]):
                    raise ValueError('Spatial semantics requires FPN maps and image/level indices')
                g,sem=semantic_bd_components(p,t,feature_maps,image_indices[active],level_indices[active],
                                            feature_strides,self.alpha,self.grid_size,self.semantic_mode,
                                            self.min_size,self.cosine_eps)
            values=g+self.semantic_weight*sem
            if not torch.isfinite(values).all():
                raise FloatingPointError('Nonfinite BD or semantic loss')
            raw=raw.index_copy(0,active.nonzero().flatten(),values)
            self.last_diagnostics=dict(bd_mean=float(g.detach().mean()),semantic_mean=float(sem.detach().mean()),
                                       weighted_semantic_mean=float(self.semantic_weight*sem.detach().mean()))
        return self.loss_weight*weight_reduce_loss(raw,weight,reduction,avg_factor)
