"""Semantic IoU Eq.(3)-(9), independently adapted to FCOS FPN features.

No fractional modulation or history is implemented. Alpha=.3, beta=.5
follow the paper. 5x5 bin-center samples, horizontal AABB context expanded
by 1.5 and use of assigned FPN features are explicit adaptation choices.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
# 本模块实现 SIoU 的语义与几何部分；FrSIoU 在此基础上另加历史调制。
# 采样网格 5×5、分配层 FPN 特征、1.5 倍上下文是本项目的适配约定。
from mmcv.ops import diff_iou_rotated_2d
from mmdet.models.losses.utils import weight_reduce_loss
from ..builder import ROTATED_LOSSES


# 可微框特征采样：feature 为 [1,C,H,W]，boxes 为 [N,5] 的像素坐标旋转框。
# 框的五个量按 (cx,cy,w,h,angle) 排列，角度单位为弧度；返回 [N,C,g,g]。
def sample_box_features(feature, boxes, stride, grid_size=5, context_scale=None):
    """Pack all ROI grids into one grid_sample; never replicate feature maps.

    Boxes are in image pixels. FPN cell (0,0) corresponds to image point
    (stride/2,stride/2), matching the FCOS point generator offset=.5.
    """
    if feature.shape[0] != 1 or boxes.ndim != 2 or boxes.shape[1] != 5:
        raise ValueError('Expected one feature image and Nx5 pixel boxes')
    count=len(boxes)
    if not count:
        return feature.new_empty((0,feature.shape[1],grid_size,grid_size))
    b=boxes.to(feature.dtype)
    cx,cy,w,h,angle=b.unbind(-1)
    # 上下文分支：先求旋转框的水平外接框，再按 context_scale 放大。
    # 水平框角度置零，但外接框宽高仍依赖原角度，相关梯度继续保留。
    if context_scale is not None:
        c,s=angle.cos().abs(),angle.sin().abs()
        w,h=(w*c+h*s)*context_scale,(w*s+h*c)*context_scale
        angle=angle*0
    # 在每个采样格中心取点；g=5 时归一化位置为 -0.4、-0.2、0、0.2、0.4。
    axis=(torch.arange(grid_size,device=b.device,dtype=b.dtype)+.5)/grid_size-.5
    yy,xx=torch.meshgrid(axis,axis,indexing='ij')
    x,y=xx[None]*w[:,None,None],yy[None]*h[:,None,None]
    c,s=angle[:,None,None].cos(),angle[:,None,None].sin()
    # 把框内局部点旋转并平移到图片坐标；预测框位置、大小和角度均可参与求导。
    gx=cx[:,None,None]+c*x-s*y
    gy=cy[:,None,None]+s*x+c*y
    # 图片坐标转换为 grid_sample 的 [-1,1] 坐标；与 align_corners=False 配套。
    # 此处假定 FPN 首个采样中心在 stride/2，与本项目 FCOS 网格偏移一致。
    grid=torch.stack([2*gx/(stride*feature.shape[-1])-1,
                      2*gy/(stride*feature.shape[-2])-1],-1)
    # 把 N 个框的网格合并成一次采样，避免把整张特征图复制 N 份。
    # 双线性插值保留对采样网格和特征的梯度；越界点按零填充。
    packed=grid.reshape(1,count*grid_size,grid_size,2)
    sampled=F.grid_sample(feature,packed,mode='bilinear',padding_mode='zeros',align_corners=False)
    return sampled.reshape(feature.shape[1],count,grid_size,grid_size).permute(1,0,2,3)


# 语义得分：pred/target 为配对框；image_indices、level_indices 指明各框的图片和层级。
def semantic_scores(pred,target,feature_maps,image_indices,level_indices,strides,
                    alpha=.3,beta=.5,grid_size=5,context_scale=1.5,cosine_eps=1e-8):
    """Keep gradients through predicted sampling grids and both feature views."""
    s1=pred.sum(-1)*0
    s2=pred.sum(-1)*0
    # 按 FPN 层和批内图片分组采样，保证每个正样本使用对应的特征图。
    for level,feature in enumerate(feature_maps):
        for image in range(feature.shape[0]):
            select=(image_indices==image)&(level_indices==level)
            ids=select.nonzero().flatten()
            if not len(ids):
                continue
            p,t=pred[ids],target[ids]
            f=feature[image:image+1]
            # 预测框和 GT 框在各自的同名网格位置配对；两路特征都保留计算图。
            fp=sample_box_features(f,p,strides[level],grid_size)
            fg=sample_box_features(f,t,strides[level],grid_size)
            # 对应式（3）：先对 C 个通道求平方和，再对 g×g 个采样位置平均。
            # 这里不是对通道取平均；原始 FPN 特征也未额外做通道归一化。
            distance=(fp-fg).square().sum(1).mean((1,2))
            # 对应式（4）：s1=exp(-alpha*d)，当前 alpha=0.3；距离越小，相似度越高。
            inside=torch.exp(-alpha*distance)
            # 对应式（5）：在水平扩展上下文中计算逐位置通道余弦，再做空间平均。
            cp=sample_box_features(f,p,strides[level],grid_size,context_scale)
            cg=sample_box_features(f,t,strides[level],grid_size,context_scale)
            context=(F.cosine_similarity(cp,cg,dim=1,eps=cosine_eps).mean((1,2))+1)/2
            s1=s1.index_copy(0,ids,inside)
            s2=s2.index_copy(0,ids,context)
    # 对应式（6）：s=beta*s1+(1-beta)*s2；本项目 beta=0.5。
    return beta*s1+(1-beta)*s2,s1,s2


# 使用可微旋转多边形 IoU；不是高斯距离或高斯 IoU 近似。
def geometric_iou(pred,target,min_size=1e-4):
    """Common translation/scale improves polygon precision without changing IoU."""
    pwh=pred[:,2:4].clamp_min(min_size)
    twh=target[:,2:4].clamp_min(min_size)
    # 对两个框使用相同平移和正比例缩放以改善浮点精度，数学上不改变 IoU。
    # 缩放参考量 detach；框自身的坐标与角度仍可求导。
    scale=torch.maximum(pwh.amax(-1),twh.amax(-1)).detach()[:,None]
    p=torch.cat(((pred[:,:2]-target[:,:2])/scale,pwh/scale,pred[:,4:]),-1)
    t=torch.cat((target[:,:2]*0,twh/scale,target[:,4:]),-1)
    return diff_iou_rotated_2d(p[None],t[None])[0].clamp(0,1)


# 纯 SIoU 对照入口：无分数阶项、无历史缓存；仍沿用 FCOS 的回归权重归约。
@ROTATED_LOSSES.register_module()
class SemanticIoULoss(nn.Module):
    """(1-s)(1-u), native FCOS centerness weighting; not FrSIoU."""
    def __init__(self,alpha=.3,beta=.5,grid_size=5,context_scale=1.5,
                 min_size=1e-4,cosine_eps=1e-8,semantic_enabled=True,
                 reduction='mean',loss_weight=1.):
        super().__init__()
        if alpha<=0 or not 0<=beta<=1 or grid_size<1 or context_scale<1 or min_size<=0 or cosine_eps<=0:
            raise ValueError('Invalid semantic sampling configuration')
        if reduction not in ('none','mean','sum') or loss_weight<0:
            raise ValueError('Invalid loss reduction or weight')
        # alpha 控制框内特征距离敏感度；beta 控制框内/上下文语义的混合比例。
        self.alpha,self.beta=alpha,beta
        self.grid_size,self.context_scale=grid_size,context_scale
        self.min_size,self.cosine_eps=min_size,cosine_eps
        self.semantic_enabled=semantic_enabled
        self.reduction,self.loss_weight=reduction,loss_weight
        self.last_diagnostics={}

    def forward(self,pred,target,weight=None,avg_factor=None,reduction_override=None,
                feature_maps=None,image_indices=None,level_indices=None,feature_strides=None,
                **kwargs):
        # forward 接收一一配对的 [N,5] 像素框；weight 来自 GT centerness。
        if pred.shape!=target.shape or pred.ndim!=2 or pred.shape[-1]!=5:
            raise ValueError('Expected aligned Nx5 pixel boxes')
        reduction=reduction_override or self.reduction
        if weight is not None and weight.ndim>1:
            weight=weight.mean(-1)
        if not pred.numel():
            self.last_diagnostics={}
            return pred.sum(-1)*0 if reduction=='none' else pred.sum()*0
        # 仅对有效正权重样本做较昂贵的 IoU 和语义采样，其他位置保留零损失。
        active=torch.ones(len(pred),device=pred.device,dtype=torch.bool) if weight is None else weight>0
        raw=pred.sum(-1)*0
        if active.any():
            p,t=pred[active],target[active]
            if not torch.isfinite(p).all() or not torch.isfinite(t).all() or (t[:,2:4]<=0).any():
                raise ValueError('Nonfinite boxes or nonpositive GT dimensions')
            p=torch.cat((p[:,:2],p[:,2:4].clamp_min(self.min_size),p[:,4:]),-1)
            iou=geometric_iou(p,t,self.min_size)
            if self.semantic_enabled:
                if feature_maps is None or image_indices is None or level_indices is None or feature_strides is None:
                    raise ValueError('SemanticIoULoss requires FPN features and per-positive image/level indices')
                s,s1,s2=semantic_scores(p,t,feature_maps,image_indices[active],level_indices[active],
                                       feature_strides,self.alpha,self.beta,self.grid_size,self.context_scale,self.cosine_eps)
            else:
                s=s1=s2=iou*0
            # 对应式（8）：语义与几何残差相乘；s 和 iou 均保留当前计算图。
            # 无重叠时 IoU 路径可能无梯度，但可微语义采样仍可提供框的更新信号。
            values=(1-s)*(1-iou)
            if not torch.isfinite(values).all():
                raise FloatingPointError('Nonfinite Semantic IoU objective')
            raw=raw.index_copy(0,active.nonzero().flatten(),values)
            # 诊断量只用于记录，不额外参与优化。
            with torch.no_grad():
                self.last_diagnostics=dict(semantic_mean=float(s.mean()),intra_similarity_mean=float(s1.mean()),
                                           context_similarity_mean=float(s2.mean()),iou_mean=float(iou.mean()),
                                           semantic_min=float(s.min()),semantic_max=float(s.max()),
                                           nonoverlap_fraction=float((iou<=1e-7).float().mean()))
        # 按 GT centerness 加权，实际检测头传入 avg_factor=sum(centerness)。
        return self.loss_weight*weight_reduce_loss(raw,weight=weight,reduction=reduction,avg_factor=avg_factor)
