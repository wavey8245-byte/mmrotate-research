"""Explicit FCOS FrSIoU engineering interpretations; not author code.

Legacy mode: group-averaged adaptation of Eq.(10),(12)-(14).
FCOSFrSIoUForwardLoss: per-box current score and frozen group history.
GL coefficients follow Eq.(12), rather than the incompatible printed Eq.(11).
values mode detaches historical values. box_tangent additionally transports
stored box-coordinate gradients by a first-order local surrogate; it does NOT
replay old network parameter Jacobians and is not an exact Eq.(20) reproduction.
Current semantic feature/box gradients remain enabled (Eq.(9)).
"""
# 阅读入口：当前 12 轮新版使用本文件末尾的 FCOSFrSIoUForwardLoss。
# 新版为逐框当前 q + 冻结的组历史；旧 box_tangent 仅供历史实验回溯。
# 本模块含 FCOS 适配约定，不能据此称为作者代码或式（20）的完整复现。
from collections import OrderedDict
import math
import torch
from mmdet.models.losses.utils import weight_reduce_loss
from ..builder import ROTATED_LOSSES
from .semantic_iou_loss import SemanticIoULoss, semantic_scores, geometric_iou


# 生成截断 Grünwald–Letnikov 系数：c0=1，ck=c(k-1)*(k-1-nu)/k。
# 采用论文式（12）对应的展开；nu=0.5、M=2 时得到 [1,-0.5,-0.125]。
def gl_coefficients(order,steps):
    if not 0<order<1 or not isinstance(steps,int) or steps<0:
        raise ValueError('Require 0<order<1 and nonnegative integer memory steps')
    values=[1.]
    for k in range(1,steps+1):
        values.append(values[-1]*(k-1-order)/k)
    return values


def fractional_terms(q,z,group,weights,lag_q,lag_gradient,coefficients,intensity,
                     gradient_mode='values',reference=None,modulation_scope='group'):
    """Local surrogate has frozen reference; moving-reference training is explicit.

    Keeping reference fixed permits ordinary numerical gradcheck of the local
    tangent objective. Refreshing it to zbar.detach() each training call retains
    historical forward values but defines a prescribed surrogate gradient.
    """
    if modulation_scope not in ('group','per_box'):
        raise ValueError('Unknown fractional modulation scope')
    if modulation_scope=='per_box' and gradient_mode!='values':
        raise ValueError('Per-box forward formulation requires frozen history values')
    # q:[N] 为当前框分数；group:[N] 把框映射到 G 个图片/GT/FPN 组。
    # lag_q:[G,M] 按从近到远存历史组均值；weights:[N] 为 GT centerness。
    n=lag_q.shape[0]
    norm=q.new_zeros(n).index_add(0,group,weights).clamp_min(1e-12)
    # 组内加权平均 q 用于写入下一次历史；新版当前调制仍使用各框自己的 q。
    meanq=q.new_zeros(n).index_add(0,group,weights*q)/norm
    # meanz 为旧版代理梯度提供组坐标；新版保留返回接口，但不使用它转移梯度。
    meanz=z.new_zeros((n,5)).index_add(0,group,weights[:,None]*z)/norm[:,None]
    # 当前新版执行分支：历史冻结，当前 q 与指数权重 W 都保留梯度。
    if modulation_scope=='per_box':
        # Eq.(10): c0 multiplies THIS prediction, not its group's current mean.
        # The past group means are an explicit FCOS history convention, and are
        # constants for this optimizer step. Never detach q or the resulting W.
        # 历史只作为数值条件；不是跨训练步保留计算图，也不反向重放过去的网络。
        past=(lag_q.detach()*q.new_tensor(coefficients[1:])[None]).sum(-1)
        # 每个框 D_i=q_i-0.5*h1[group_i]-0.125*h2[group_i]（当前默认设置）。
        # 缺失历史补零：首次访问时 D_i=q_i，W 仍通常不等于 1。
        derivative=q+past[group]
        # W=exp(-lambda*D)，L_i=W*(1-q_i)；1-q_i 等于 (1-s_i)*(1-u_i)。
        # 固定历史时 dL_i/dq_i=-W*[1+lambda*(1-q_i)]，不能把 W 当作常量。
        modulation=torch.exp(-intensity*derivative)
        return modulation*(1-q),meanq,meanz,derivative,modulation
    # 以下是旧版组级调制；当前 FCOSFrSIoUForwardLoss 在上面已返回，不执行此处。
    history=lag_q
    # 旧版局部切线代理：默认参考点处增量为零，但引入存储的框坐标梯度。
    # 该代理不等于历史网络参数雅可比的重放，不能直接等同论文式（20）。
    if gradient_mode=='box_tangent':
        reference=meanz.detach() if reference is None else reference
        history=history+(lag_gradient*(meanz-reference)[:,None,:]).sum(-1)
    elif gradient_mode!='values':
        raise ValueError('Unknown history gradient interpretation')
    derivative=meanq+(history*q.new_tensor(coefficients[1:])[None]).sum(-1)
    modulation=torch.exp(-intensity*derivative)
    return modulation[group]*(1-q),meanq,meanz,derivative,modulation


# 仅旧版使用：按 GT 面积尺度与翻转方向统一框坐标，以转移局部框梯度。
# 当前新版不调用该函数；这里的归一化也不用于当前语义框采样。
def canonical_box_coordinates(pred,target,images,metas):
    """GT-area units, original image axes; only uniform resize and le90 flips.

    No long-edge swapping is done. le90 RRandomFlip retains w/h order.
    z=(sx*dx/L,sy*dy/L,w/L,h/L,sx*sy*(angle-angle_gt)), L=sqrt(gt area).
    d pred / d z is diagonal and returned to transform stored box gradients.
    """
    scale=(target[:,2]*target[:,3]).sqrt().detach()
    signs=pred.new_ones((len(pred),2))
    for image,meta in enumerate(metas):
        factor=meta.get('scale_factor',[1.,1.,1.,1.])
        if hasattr(factor,'__len__') and not math.isclose(float(factor[0]),float(factor[1]),rel_tol=1e-5):
            raise ValueError('History tangent requires uniform image resize')
        if meta.get('flip',False):
            direction=meta.get('flip_direction')
            if direction not in ('horizontal','vertical','diagonal'):
                raise ValueError('Unsupported history flip')
            selected=images==image
            if direction in ('horizontal','diagonal'):
                signs[selected,0]=-1
            if direction in ('vertical','diagonal'):
                signs[selected,1]=-1
    # 水平/垂直翻转会改变角度方向；用 pi 周期的最短角差避免等价角跳变。
    parity=signs.prod(-1)
    delta=pred[:,4]-target[:,4]
    angle=.5*torch.atan2(torch.sin(2*delta),torch.cos(2*delta))
    z=torch.cat(((pred[:,:2]-target[:,:2])*signs/scale[:,None],
                 pred[:,2:4]/scale[:,None],(parity*angle)[:,None]),-1)
    jac=torch.cat((scale[:,None]*signs,scale[:,None].expand(-1,2),parity[:,None]),-1)
    return z,jac


# 为正样本恢复 GT 标注行号，生成可跨批次查找的历史身份。
# 当前 resize/flip 流程保留 GT 行顺序；这里不通过预测框做目标跟踪。
def object_history_keys(target,images,levels,metas,gt_boxes):
    """Recover GT annotation row from decoded pixel target, never from prediction.

    Current resize/flip pipeline preserves annotation order. Matching is
    checked with pixel+angle tolerance and restricted to the same batch image.
    """
    assignments=torch.empty(len(target),device=target.device,dtype=torch.long)
    for image,gt in enumerate(gt_boxes):
        ids=(images==image).nonzero().flatten()
        if not len(ids):
            continue
        if not len(gt):
            raise ValueError('Positive target in an empty GT image')
        for start in range(0,len(ids),256):
            selected=ids[start:start+256]
            t=target[selected].detach()
            # 仅在同一图片内，将解码后的目标框与输入 GT 匹配；角差按 pi 周期比较。
            error=(t[:,None,:4]-gt[None,:,:4]).abs().amax(-1)
            angle=(t[:,None,4]-gt[None,:,4]+math.pi/2)%math.pi-math.pi/2
            error=error+angle.abs()*100
            best,index=error.min(-1)
            if (best>2e-2).any():
                raise RuntimeError('Decoded target does not match a GT annotation row')
            assignments[selected]=index
    image_cpu=images.detach().cpu().tolist()
    level_cpu=levels.detach().cpu().tolist()
    annotation_cpu=assignments.detach().cpu().tolist()
    unique=[]
    lookup={}
    group=[]
    for image,level,annotation in zip(image_cpu,level_cpu,annotation_cpu):
        name=metas[image].get('ori_filename',metas[image].get('filename'))
        if not name:
            raise ValueError('Stable image filename required for FrSIoU history')
        # 同图、同 GT、同 FPN 层共享一份历史；不是某个固定预测点的时间轨迹。
        # 历史时间指该组前几次被访问，未必是连续 SGD 步或连续 epoch。
        key=(str(name),int(annotation),int(level))
        if key not in lookup:
            lookup[key]=len(unique)
            unique.append(key)
        group.append(lookup[key])
    return unique,torch.tensor(group,device=target.device,dtype=torch.long)


# 兼容旧实验的基类：默认参数仍是旧组级 box_tangent 模式。
# 当前训练通过末尾新版子类强制改为 per_box + values。
@ROTATED_LOSSES.register_module()
class FCOSFrSIoUEngineeringLoss(SemanticIoULoss):
    """Full forward recurrence plus optional explicit box-tangent gradient memory."""
    def __init__(self,fractional_order=.5,memory_steps=2,modulation_lambda=.3,
                 history_gradient='box_tangent',max_history_keys=200000,
                 modulation_scope='group',**kwargs):
        super().__init__(**kwargs)
        if modulation_lambda<0 or history_gradient not in ('values','box_tangent') or max_history_keys<1:
            raise ValueError('Invalid fractional history configuration')
        if modulation_scope not in ('group','per_box'):
            raise ValueError('Invalid fractional modulation scope')
        if modulation_scope=='per_box' and history_gradient!='values':
            raise ValueError('Per-box forward formulation requires frozen history values')
        self.coefficients=gl_coefficients(fractional_order,memory_steps)
        self.fractional_order=fractional_order
        self.memory_steps=memory_steps
        self.modulation_lambda=modulation_lambda
        self.history_gradient=history_gradient
        self.modulation_scope=modulation_scope
        self.max_history_keys=max_history_keys
        # 历史是独立于模型 state_dict 的数值缓存；最近提交的组移动到字典末尾。
        # _pending 暂存本次前向的组均值，等待优化器更新后由 Hook 提交。
        self.history=OrderedDict()
        self._context=None
        self._pending=None
        self.committed_updates=0
        self.total_history_hits=0

    # 检测头临时提供图片元信息和 GT 列表，供历史键识别；计算后会清理。
    def set_batch_context(self,metas,gt_boxes):
        if self._context is not None:
            raise RuntimeError('Nested FrSIoU batch contexts are unsupported')
        self._context=(metas,gt_boxes)

    def clear_batch_context(self):
        self._context=None

    # 优化器更新后提交；写入的是本批次更新前前向得到的分数，不会重新前向。
    def commit_history(self):
        if self._pending is None:
            return
        for key,value,gradient in self._pending:
            # 每组最新记录放首位，最多保留 M 条；超出组容量时淘汰最久未提交的组。
            records=self.history.pop(key,[])
            self.history[key]=[(float(value),tuple(float(g) for g in gradient))]+records[:max(0,self.memory_steps-1)] if self.memory_steps else []
        while len(self.history)>self.max_history_keys:
            self.history.popitem(last=False)
        self.committed_updates+=1
        self._pending=None

    # 缓存签名记录数学参数和身份约定，避免恢复不兼容的历史。
    def signature(self):
        signature=dict(schema=1,order=self.fractional_order,steps=self.memory_steps,
                    intensity=self.modulation_lambda,gradient=self.history_gradient,
                    max_history_keys=self.max_history_keys,alpha=self.alpha,beta=self.beta,
                    grid_size=self.grid_size,context_scale=self.context_scale,
                    min_size=self.min_size,cosine_eps=self.cosine_eps,
                    semantic_enabled=self.semantic_enabled,loss_weight=self.loss_weight,
                    key='image_filename,GT_annotation_row,FPN_level',
                    missing_history='zero',coefficients=self.coefficients)
        # Keep legacy checkpoint signatures exactly compatible. New semantics
        # must reject old histories even when all hyperparameters happen to match.
        # 新版使用 schema=2，与旧版组级缓存显式区分；不是仅改类名后混用旧缓存。
        if self.modulation_scope=='per_box':
            signature.update(schema=2,modulation_scope='per_box',
                             history_clock='previous_group_visits',
                             history_gradient='frozen_values',
                             current_semantic_gradient='enabled')
        return signature

    # 保存时不能仍有未提交状态，否则模型进度与历史进度可能不一致。
    def export_history(self):
        if self._pending is not None:
            raise RuntimeError('Cannot checkpoint uncommitted optimizer-step history')
        return dict(signature=self.signature(),records=list(self.history.items()),
                    committed_updates=self.committed_updates,total_history_hits=self.total_history_hits)

    # 恢复前核对签名、条数、分数范围与梯度维度；新版梯度槽位固定为零。
    def import_history(self,state):
        if state.get('signature')!=self.signature():
            raise ValueError('History sidecar specification does not match loss')
        records=OrderedDict()
        for key,values in state['records']:
            key=tuple(key)
            if len(key)!=3 or len(values)>self.memory_steps:
                raise ValueError('Malformed history record')
            for score,gradient in values:
                if not math.isfinite(score) or not 0<=score<=1 or len(gradient)!=5 or not all(math.isfinite(x) for x in gradient):
                    raise ValueError('Nonfinite or invalid history state')
            records[key]=values
        if len(records)>self.max_history_keys:
            raise ValueError('History exceeds configured capacity')
        self.history=records
        self.committed_updates=int(state['committed_updates'])
        self.total_history_hits=int(state['total_history_hits'])
        self._pending=None

    # 完整调用链：筛选样本 → 历史身份 → IoU/语义 → 当前 q → 读取历史 → 调制与归约。
    def forward(self,pred,target,weight=None,avg_factor=None,reduction_override=None,
                feature_maps=None,image_indices=None,level_indices=None,feature_strides=None,**kwargs):
        if pred.shape!=target.shape or pred.ndim!=2 or pred.shape[-1]!=5:
            raise ValueError('Expected aligned Nx5 pixel boxes')
        reduction=reduction_override or self.reduction
        if weight is not None and weight.ndim>1:
            weight=weight.mean(-1)
        # 仅训练且启用梯度时生成待提交历史；验证/no_grad 调用不会改写训练历史。
        learning=self.training and torch.is_grad_enabled() and pred.requires_grad
        if learning and self._pending is not None:
            raise RuntimeError('Commit FrSIoU history after optimizer.step before next training batch')
        if not pred.numel():
            self.last_diagnostics={}
            return pred.sum(-1)*0 if reduction=='none' else pred.sum()*0
        # N 为传入的配对框数；active 过滤零权重样本，最后再回填到原 N 个位置。
        active=torch.ones(len(pred),device=pred.device,dtype=torch.bool) if weight is None else weight>0
        raw=pred.sum(-1)*0
        if not active.any():
            self.last_diagnostics={}
            return self.loss_weight*weight_reduce_loss(raw,weight=weight,reduction=reduction,avg_factor=avg_factor)
        if self._context is None:
            raise ValueError('FrSIoU needs image metadata and GT annotation rows from its FCOS adapter')
        if image_indices is None or level_indices is None:
            raise ValueError('FrSIoU requires per-positive image and level indices')
        if self.semantic_enabled and (feature_maps is None or feature_strides is None):
            raise ValueError('FrSIoU requires feature maps and strides for semantic supervision')
        p,t=pred[active],target[active]
        if not torch.isfinite(p).all() or not torch.isfinite(t).all() or (t[:,2:4]<=0).any():
            raise ValueError('Nonfinite boxes or nonpositive GT')
        p=torch.cat((p[:,:2],p[:,2:4].clamp_min(self.min_size),p[:,4:]),-1)
        images,levels=image_indices[active],level_indices[active]
        metas,gt_boxes=self._context
        # keys 长度为 G；groups 长度为有效框数，每个元素都是 0..G-1 的组索引。
        keys,groups=object_history_keys(t,images,levels,metas,gt_boxes)
        if self.modulation_scope=='per_box':
            # No tangent transport, so no canonical coordinates or flip Jacobian.
            # 新版直接保留像素框，跳过旧版坐标标准化及梯度变换。
            z,jac=p,None
        else:
            z,jac=canonical_box_coordinates(p,t,images,metas)
        u=geometric_iou(p,t,self.min_size)
        if self.semantic_enabled:
            s,s1,s2=semantic_scores(p,t,feature_maps,images,levels,feature_strides,
                self.alpha,self.beta,self.grid_size,self.context_scale,self.cosine_eps)
        else:
            s=s1=s2=u*0
        # 融合分数对应式（7）；当前 u、s、q 都可微，未对当前语义分数 detach。
        q=s+(1-s)*u
        w=p.new_ones(len(p)) if weight is None else weight[active]
        # 先以零补齐历史：lq:[G,M]；lg:[G,M,5] 仅供旧代理分支使用。
        lq=p.new_zeros((len(keys),self.memory_steps))
        lg=p.new_zeros((len(keys),self.memory_steps,5))
        hits=0
        # 逐组查缓存，最近记录放 lag=0；命中计数按组统计，不是按预测框统计。
        for index,key in enumerate(keys):
            records=self.history.get(key,[])
            if records:
                hits+=1
            for lag,(score,gradient) in enumerate(records[:self.memory_steps]):
                lq[index,lag]=score
                if self.history_gradient=='box_tangent':
                    lg[index,lag]=p.new_tensor(gradient)
        # values 是逐框损失；meanq 仅用于本次训练完成后的历史提交。
        values,meanq,meanz,derivative,modulation=fractional_terms(
            q,z,groups,w,lq,lg,self.coefficients,self.modulation_lambda,self.history_gradient,
            modulation_scope=self.modulation_scope)
        if not torch.isfinite(values).all():
            raise FloatingPointError('Nonfinite fractional semantic objective')
        if learning:
            # 此分支只为旧实验计算框梯度代理；当前新版走 else，历史梯度槽位为零。
            if self.history_gradient=='box_tangent' and self.memory_steps:
                gp,=torch.autograd.grad(q.sum(),p,retain_graph=True,create_graph=False)
                gz=gp.detach()*jac
                norm=w.new_zeros(len(keys)).index_add(0,groups,w).clamp_min(1e-12)
                gradient=p.new_zeros((len(keys),5)).index_add(0,groups,w[:,None]*gz)/norm[:,None]
            else:
                gradient=p.new_zeros((len(keys),5))
            if not torch.isfinite(gradient).all():
                raise FloatingPointError('Nonfinite history box gradient')
            # 组均值 detach 后转为 CPU 数值，只存历史状态，不跨训练步保留计算图。
            # 此处只暂存，FrSIoUMemoryHook 会在 optimizer.step 之后提交。
            self._pending=list(zip(keys,meanq.detach().cpu().tolist(),gradient.detach().cpu().tolist()))
            self.total_history_hits+=hits
        # 把有效样本损失放回原位置，随后按传入 GT centerness 加权。
        raw=raw.index_copy(0,active.nonzero().flatten(),values)
        # 以下诊断统计不参与反向；命中率用于确认历史机制是否被实际使用。
        with torch.no_grad():
            self.last_diagnostics=dict(semantic_mean=float(s.mean()),intra_similarity_mean=float(s1.mean()),
                context_similarity_mean=float(s2.mean()),iou_mean=float(u.mean()),
                semantic_min=float(s.min()),semantic_max=float(s.max()),nonoverlap_fraction=float((u<=1e-7).float().mean()),
                groups=len(keys),history_hit_groups=hits,history_hit_fraction=hits/max(len(keys),1),
                cached_groups=len(self.history),committed_updates=self.committed_updates,
                q_mean=float(q.mean()),fractional_derivative_mean=float(derivative.mean()),
                modulation_mean=float(modulation.mean()),modulation_min=float(modulation.min()),modulation_max=float(modulation.max()),
                transported_history_gradient_l2=float((lg*p.new_tensor(self.coefficients[1:])[None,:,None]).sum(1).norm()))
        # 本项目 bbox loss=sum(centerness_i*L_i)/sum(centerness_i)，外乘 loss_weight。
        return self.loss_weight*weight_reduce_loss(raw,weight=weight,reduction=reduction,avg_factor=avg_factor)


# 当前 12 轮训练入口：每框当前分数独立调制，历史为冻结的组均值。
# 与论文完整复现的差异包括历史组定义及历史导数缺失；不声称实现式（20）。
@ROTATED_LOSSES.register_module()
class FCOSFrSIoUForwardLoss(FCOSFrSIoUEngineeringLoss):
    """Per-box Eq.(10),(12)-(14), with the full current semantic gradient.

    Past scores are frozen image/GT/FPN group means from previous visits, not
    a paper-specified per-prediction trajectory. This is a forward-formula
    implementation under that stated convention, NOT a reproduction of Eq.(20)
    or of the authors' unreleased training code. No historical tangent is used.
    """
    def __init__(self,**kwargs):
        if kwargs.get('history_gradient','values')!='values':
            raise ValueError('Forward loss does not support historical tangent gradients')
        if kwargs.get('modulation_scope','per_box')!='per_box':
            raise ValueError('Forward loss requires per-box modulation')
        # 强制锁定新版行为；不会因继承旧基类而进入组级权重或历史切线代理。
        kwargs.update(history_gradient='values',modulation_scope='per_box')
        super().__init__(**kwargs)
